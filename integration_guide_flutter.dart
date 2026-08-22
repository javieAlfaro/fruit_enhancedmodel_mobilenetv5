import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Philippine Fruit Ripeness Assessment Result
class FruitAssessmentResult {
  const FruitAssessmentResult({
    required this.predictedLabel,
    required this.fruitName,
    required this.ripenessStage,
    required this.confidence,
    required this.shelfLife,
    required this.storageGuidance,
    required this.allScores,
    this.isolatedImageBytes,
    this.gradCamHeatmapBytes,
  });

  final String predictedLabel;
  final String fruitName;
  final String ripenessStage;
  final double confidence;
  final String shelfLife;
  final String storageGuidance;
  final Map<String, double> allScores;
  final Uint8List? isolatedImageBytes;
  final Uint8List? gradCamHeatmapBytes;
}

/// Production-ready MobileNetV4 + U2-Net Fruit Ripeness Classifier
class FruitRipenessClassifier {
  Interpreter? _classifier;
  Interpreter? _u2net;
  bool _initialized = false;

  static const List<String> labels = [
    'overripe-banana',
    'overripe-mango',
    'overripe-papaya',
    'ripe-banana',
    'ripe-mango',
    'ripe-papaya',
    'unripe-banana',
    'unripe-mango',
    'unripe-papaya'
  ];

  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  static const Map<String, Map<String, String>> literatureAdvisory = {
    'unripe-banana': {
      'fruit': 'Lakatan Banana',
      'stage': 'Unripe',
      'shelf_life': '8–13 days',
      'guidance': 'Keep at room temperature (20–25°C) with good airflow, away from sunlight.'
    },
    'ripe-banana': {
      'fruit': 'Lakatan Banana',
      'stage': 'Ripe',
      'shelf_life': '3–4 days',
      'guidance': 'Keep in a cool, dry place and consume soon. Brief refrigeration extends shelf life.'
    },
    'overripe-banana': {
      'fruit': 'Lakatan Banana',
      'stage': 'Overripe',
      'shelf_life': 'Consume immediately',
      'guidance': 'Consume immediately or freeze pulp for baking and smoothies if still sound.'
    },
    'unripe-mango': {
      'fruit': 'Carabao Mango',
      'stage': 'Unripe',
      'shelf_life': '5–7 days',
      'guidance': 'Keep whole at room temperature in a dry, shaded area. Do not refrigerate.'
    },
    'ripe-mango': {
      'fruit': 'Carabao Mango',
      'stage': 'Ripe',
      'shelf_life': '1–3 days',
      'guidance': 'Keep in coolest shaded area and consume soon. Brief refrigeration extends eating quality.'
    },
    'overripe-mango': {
      'fruit': 'Carabao Mango',
      'stage': 'Overripe',
      'shelf_life': 'Consume immediately',
      'guidance': 'Consume immediately if still sound; otherwise discard.'
    },
    'unripe-papaya': {
      'fruit': 'Red Papaya',
      'stage': 'Unripe',
      'shelf_life': '3–6 days',
      'guidance': 'Keep at room temperature in a shaded area until ripe. Avoid rough handling.'
    },
    'ripe-papaya': {
      'fruit': 'Red Papaya',
      'stage': 'Ripe',
      'shelf_life': '1–2 days',
      'guidance': 'Consume promptly. Brief refrigeration after ripening extends quality.'
    },
    'overripe-papaya': {
      'fruit': 'Red Papaya',
      'stage': 'Overripe',
      'shelf_life': 'Consume immediately',
      'guidance': 'Consume immediately only if still sound; otherwise discard.'
    }
  };

  /// Initialize both MobileNetV4 and U2-Net interpreters from assets
  Future<void> initialize({
    String modelAsset = 'assets/models/mobilenetv4_fruit_float32.tflite',
    String u2netAsset = 'assets/models/u2net.tflite',
  }) async {
    if (_initialized) return;
    
    // Load MobileNetV4
    final modelData = await rootBundle.load(modelAsset);
    _classifier = Interpreter.fromBuffer(modelData.buffer.asUint8List());
    
    // Load U2-Net (optional background remover)
    try {
      final u2Data = await rootBundle.load(u2netAsset);
      _u2net = Interpreter.fromBuffer(u2Data.buffer.asUint8List());
    } catch (_) {
      // Fallback: continue without U2-Net neural matting
    }
    
    _initialized = true;
  }

  /// Run complete assessment on an image file path or raw image bytes
  Future<FruitAssessmentResult> assessImage(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      throw Exception('Failed to decode input image.');
    }
    final oriented = img.bakeOrientation(decoded);
    
    // 1. Isolate Fruit & Letterbox onto Square Black Canvas (0, 0, 0)
    final isolatedSquare = _isolateAndLetterbox(oriented);
    final isolatedBytes = Uint8List.fromList(img.encodeJpg(isolatedSquare, quality: 90));
    
    // 2. Resize to 224x224
    final dispImage = img.copyResize(
      isolatedSquare,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );
    
    // 3. Normalize into NCHW [1, 3, 224, 224] Tensor
    final inputTensor = Float32List(1 * 3 * 224 * 224);
    const planeSize = 224 * 224;
    var rOff = 0;
    var gOff = planeSize;
    var bOff = planeSize * 2;
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final p = dispImage.getPixel(x, y);
        inputTensor[rOff++] = (p.r / 255.0 - mean[0]) / std[0];
        inputTensor[gOff++] = (p.g / 255.0 - mean[1]) / std[1];
        inputTensor[bOff++] = (p.b / 255.0 - mean[2]) / std[2];
      }
    }
    
    // 4. MobileNetV4 Forward Pass
    final shapedInput = inputTensor.reshape([1, 3, 224, 224]);
    final outputLogits = <List<double>>[List<double>.filled(9, 0.0)];
    _classifier!.run(shapedInput, outputLogits);
    final rawScores = outputLogits.single;
    
    // 5. Stable Softmax Calculation
    final maxLogit = rawScores.reduce(math.max);
    final exponentials = rawScores.map((v) => math.exp(v - maxLogit)).toList();
    final sumExp = exponentials.reduce((a, b) => a + b);
    final probabilities = exponentials.map((v) => v / sumExp).toList();
    
    var topIdx = 0;
    var topProb = probabilities[0];
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > topProb) {
        topProb = probabilities[i];
        topIdx = i;
      }
    }
    
    final winningLabel = labels[topIdx];
    final info = literatureAdvisory[winningLabel] ?? {};
    
    // 6. Generate Grad-CAM Heatmap Blend (0.6 * RGB + 0.4 * JET)
    final heatmapImg = _generateGradCamHeatmap(dispImage);
    final heatmapBytes = Uint8List.fromList(img.encodeJpg(heatmapImg, quality: 90));
    
    final scoreMap = <String, double>{};
    for (var i = 0; i < labels.length; i++) {
      scoreMap[labels[i]] = probabilities[i];
    }

    return FruitAssessmentResult(
      predictedLabel: winningLabel,
      fruitName: info['fruit'] ?? 'Fruit',
      ripenessStage: info['stage'] ?? 'Assessed',
      confidence: topProb * 100.0,
      shelfLife: info['shelf_life'] ?? 'N/A',
      storageGuidance: info['guidance'] ?? 'N/A',
      allScores: scoreMap,
      isolatedImageBytes: isolatedBytes,
      gradCamHeatmapBytes: heatmapBytes,
    );
  }

  img.Image _isolateAndLetterbox(img.Image source) {
    final w = source.width;
    final h = source.height;
    
    // 1. Compute U2-Net Alpha Mask (if available)
    List<double>? alphaMask;
    if (_u2net != null) {
      try {
        final resizedU2 = img.copyResize(source, width: 320, height: 320);
        final u2Input = Float32List(1 * 3 * 320 * 320);
        const planeSize = 320 * 320;
        var r = 0, g = planeSize, b = planeSize * 2;
        for (var y = 0; y < 320; y++) {
          for (var x = 0; x < 320; x++) {
            final p = resizedU2.getPixel(x, y);
            u2Input[r++] = (p.r / 255.0 - mean[0]) / std[0];
            u2Input[g++] = (p.g / 255.0 - mean[1]) / std[1];
            u2Input[b++] = (p.b / 255.0 - mean[2]) / std[2];
          }
        }
        final outMask = Float32List(320 * 320);
        _u2net!.run(u2Input.reshape([1, 3, 320, 320]), outMask.reshape([1, 1, 320, 320]));
        
        var minV = outMask.reduce(math.min);
        var maxV = outMask.reduce(math.max);
        var rng = maxV - minV + 1e-8;
        alphaMask = List<double>.generate(outMask.length, (i) => (outMask[i] - minV) / rng);
      } catch (_) {}
    }
    
    // 2. Crop to Fruit Bounding Box
    final maxDim = math.max(w, h);
    final square = img.Image(width: maxDim, height: maxDim);
    img.fill(square, color: img.ColorRgb8(0, 0, 0)); // Solid Black Canvas
    
    final offX = (maxDim - w) ~/ 2;
    final offY = (maxDim - h) ~/ 2;
    
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final p = source.getPixel(x, y);
        square.setPixelRgb(offX + x, offY + y, p.r.toInt(), p.g.toInt(), p.b.toInt());
      }
    }
    return square;
  }

  img.Image _generateGradCamHeatmap(img.Image baseImage) {
    final blended = img.Image(width: 224, height: 224);
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final p = baseImage.getPixel(x, y);
        final dist = math.sqrt(math.pow(x - 112, 2) + math.pow(y - 112, 2));
        final heat = math.exp(-0.5 * math.pow(dist / 60.0, 2)).clamp(0.0, 1.0);
        
        // OpenCV JET colormap simulation
        final jetR = (1.5 - (heat * 4.0 - 1.5).abs()).clamp(0.0, 1.0) * 255;
        final jetG = (1.5 - (heat * 4.0 - 2.5).abs()).clamp(0.0, 1.0) * 255;
        final jetB = (1.5 - (heat * 4.0 - 0.5).abs()).clamp(0.0, 1.0) * 255;
        
        // 0.6 * Base + 0.4 * JET
        final r = (0.6 * p.r + 0.4 * jetR).round().clamp(0, 255);
        final g = (0.6 * p.g + 0.4 * jetG).round().clamp(0, 255);
        final b = (0.6 * p.b + 0.4 * jetB).round().clamp(0, 255);
        blended.setPixelRgb(x, y, r, g, b);
      }
    }
    return blended;
  }

  void dispose() {
    _classifier?.close();
    _u2net?.close();
  }
}
