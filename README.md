# MobileNetV4 Fruit Ripeness Detection Model Integration (v5)

## Overview
This repository provides the trained MobileNetV4 (v5) models, schema definitions, and Flutter/Dart integration contracts for Philippine Fruit Ripeness Detection (Banana, Mango, Papaya).

## Repository File Index
| File | Description |
| :--- | :--- |
| `fruit-ripeness-v5.tflite` | Primary Model: High-speed 9-class Softmax classifier for mobile inference. |
| `fruit-ripeness-v5-heatmap.tflite` | Explainability Model: Dual-output TFLite exporting probabilities + 7x7 spatial feature map. |
| `labels.txt` | Alphabetical label ordering (Index 0 to 8). |
| `model_contract.json` | Detailed tensor shapes, data types, and normalization formulas. |
| `preprocess.py` | Standalone Python reference script mirroring Google Colab transformations. |
| `integration_guide_flutter.dart` | Production-ready Dart service class implementing crop, resize, normalization, and inference. |
| `fixtures/` | Golden input image and expected JSON predictions for unit testing. |

## Model Input Contract
* **Input Tensor:** `[1, 224, 224, 3]` (float32, RGB)
* **Viewfinder Square Crop:** Raw frame must be cropped to the UI Target Box prior to resizing to prevent aspect-ratio distortion.
* **Pixel Normalization Formula:** `Normalized Pixel = ((Raw Pixel / 255.0) - Mean) / Std`
  * **Mean (RGB):** `[0.485, 0.456, 0.406]`
  * **Std (RGB):** `[0.229, 0.224, 0.225]`

## Class Index Mapping (`labels.txt`)
```text
0: overripe-banana
1: overripe-mango
2: overripe-papaya
3: ripe-banana
4: ripe-mango
5: ripe-papaya
6: unripe-banana
7: unripe-mango
8: unripe-papaya
```

## Out-of-Distribution (OOD) & Fallback Rules
1. **Confidence Gate:** If max(scores) < 0.75, the app must not declare a final ripeness stage.
2. **User Guidance:** Display: *'Fruit not recognized or uncentered. Please align inside target box.'*
3. **Heatmap Overlay (Optional):** If using `fruit-ripeness-v5-heatmap.tflite`, take the second output `[1, 1, 7, 7]`, bilinear-upsample to 224x224, apply a colormap gradient, and blend over the viewfinder with 50% opacity.
