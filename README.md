# MobileNetV4 Fruit Ripeness & Shelf-Life Detection Model Bundle (v5)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-v2.x-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/lite)
[![Flutter](https://img.shields.io/badge/Flutter-Ready-02569B.svg?style=flat&logo=flutter)](https://flutter.dev)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.56%25-brightgreen.svg)]()

Production-ready model weights, neural foreground matting segmentation, Grad-CAM explainability, and Flutter/Dart integration contracts for **Philippine Fruit Ripeness Assessment** (*Lakatan Banana*, *Carabao Mango*, and *Red Papaya*).

---

## 🏗️ Architecture Pipeline

```mermaid
graph TD
    A[Raw Input Photo / Camera Frame] --> B[U2-Net Neural Matting (u2net.tflite)]
    B --> C[Alpha Mask Extraction]
    C --> D[Tight Bounding-Box Crop]
    D --> E[Aspect-Ratio Square Black Canvas Letterbox]
    E --> F[Bilinear Resize to 224x224 & ImageNet Norm]
    F --> G[MobileNetV4 Classifier (mobilenetv4_fruit_float32.tflite)]
    G --> H[Stable Softmax Probabilities]
    G --> I[Grad-CAM JET Heatmap Visualization]
    H --> J[Postharvest Shelf-Life & Storage Advisory]
    I --> K[Multi-View UI: Isolated / Original / Grad-CAM]
    J --> K
```

---

## 📦 Repository Contents & File Index

| File | Description |
| :--- | :--- |
| `mobilenetv4_fruit_float32.tflite` | **Primary Model**: MobileNetV4 float32 classifier (9 classes, NCHW `[1, 3, 224, 224]`). |
| `u2net.tflite` | **Neural Matting Model**: U2-Net on-device background remover (`[1, 3, 320, 320]`). |
| `labels.txt` | Alphabetical label ordering (Index 0 to 8). |
| `model_contract.json` | JSON schema contract defining tensor shapes, normalization formulas, and shelf-life rules. |
| `preprocess.py` | Standalone Python reference script implementing the full Colab inference pipeline. |
| `integration_guide_flutter.dart` | Production-ready Flutter/Dart service class for on-device inference. |
| `CODEX_IMPLEMENTATION_GUIDE.md` | **Codex Guide**: Step-by-step instructions & prompts for integrating via Codex. |
| `fixtures/` | Golden input image and expected JSON predictions for verification. |

---

## 🎯 Target Classes & Postharvest Shelf-Life Reference

| Index | Class Name | Target Fruit | Ripeness Stage | Estimated Shelf Life | Storage Guidance |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | `overripe-banana` | Lakatan Banana | Overripe | Consume immediately | Consume immediately for eating, or freeze pulp for smoothies and baking. |
| **1** | `overripe-mango` | Carabao Mango | Overripe | Consume immediately | Consume immediately if still sound; otherwise discard. |
| **2** | `overripe-papaya` | Red Papaya | Overripe | Consume immediately | Consume immediately only if still sound; otherwise discard. |
| **3** | `ripe-banana` | Lakatan Banana | Ripe | 3–4 days | Keep in a cool, dry place and consume soon. Brief refrigeration extends eating quality. |
| **4** | `ripe-mango` | Carabao Mango | Ripe | 1–3 days | Keep in the coolest shaded area available and consume soon. Brief refrigeration is reasonable. |
| **5** | `ripe-papaya` | Red Papaya | Ripe | 1–2 days | Consume promptly. Brief refrigeration after ripening extends edible quality. |
| **6** | `unripe-banana` | Lakatan Banana | Unripe | 8–13 days | Keep at room temperature (20–25°C) with good airflow, away from direct sunlight. |
| **7** | `unripe-mango` | Carabao Mango | Unripe | 5–7 days | Keep whole at room temperature in a dry, ventilated, shaded area. Do not refrigerate. |
| **8** | `unripe-papaya` | Red Papaya | Unripe | 3–6 days | Keep at room temperature in a dry, shaded area until ripe. Avoid rough handling. |

---

## 🧪 Quick Test (Python)

To test the model bundle locally on your computer:
```bash
pip install pillow numpy ai-edge-litert opencv-python
python preprocess.py path/to/fruit_photo.jpg
```

---

## 📱 Flutter Integration

Refer to **[`CODEX_IMPLEMENTATION_GUIDE.md`](./CODEX_IMPLEMENTATION_GUIDE.md)** for detailed copy-paste instructions and prompts for the **Codex** AI assistant.
