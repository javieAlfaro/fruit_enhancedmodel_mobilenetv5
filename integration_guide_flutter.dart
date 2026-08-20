import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class FruitRipenessClassifier {
  late Interpreter _interpreter;
  late List<String> _labels;
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std  = [0.229, 0.224, 0.225];
  static const double CONFIDENCE_THRESHOLD = 0.75;

  Future<void> initialize({
    String modelAsset = 'assets/models/fruit-ripeness-v5.tflite',
    String labelAsset = 'assets/models/labels.txt',
  }) async {
    _interpreter = await Interpreter.fromAsset(modelAsset);
  }

  Map<String, dynamic> runInferenceOnFrame({
    required Uint8List rawBytes,
    required int cropX,
    required int cropY,
    required int cropWidth,
    required int cropHeight,
  }) {
    img.Image? decoded = img.decodeImage(rawBytes);
    if (decoded == null) throw Exception('Failed to decode image bytes.');
    img.Image cropped = img.copyCrop(decoded, x: cropX, y: cropY, width: cropWidth, height: cropHeight);
    img.Image resized = img.copyResize(cropped, width: 224, height: 224);

    var inputTensor = List.generate(1, (_) => List.generate(224, (y) => List.generate(224, (x) {
      final pixel = resized.getPixelSafe(x, y);
      return [
        ((pixel.r / 255.0) - _mean[0]) / _std[0],
        ((pixel.g / 255.0) - _mean[1]) / _std[1],
        ((pixel.b / 255.0) - _mean[2]) / _std[2],
      ];
    })));

    var outputProbabilities = List.filled(1 * 9, 0.0).reshape([1, 9]);
    _interpreter.run(inputTensor, outputProbabilities);
    List<double> scores = List<double>.from(outputProbabilities[0]);

    int topIndex = 0;
    double maxScore = -1.0;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) { maxScore = scores[i]; topIndex = i; }
    }

    bool isOOD = maxScore < CONFIDENCE_THRESHOLD;
    return {
      'class_index': topIndex,
      'label': _labels[topIndex],
      'confidence': maxScore,
      'is_uncertain_or_ood': isOOD,
      'all_scores': scores,
      'user_message': isOOD ? 'Fruit not recognized. Please align inside target box.' : 'Classification successful.'
    };
  }
}
