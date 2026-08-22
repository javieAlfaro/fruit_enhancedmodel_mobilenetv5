# Codex Implementation & Integration Guide: MobileNetV4 Fruit Ripeness (v5)

This guide provides instructions and ready-to-use prompts for integrating the **MobileNetV4 Enhanced v5 Model Bundle** into your Flutter / Android application using the **Codex** AI coding assistant.

---

## 📁 Required Model Files

Ensure the following 3 files are placed in your Flutter project's `assets/models/` directory:
1. `mobilenetv4_fruit_float32.tflite` (Primary MobileNetV4 Classifier)
2. `u2net.tflite` (Neural Background Matting Segmenter)
3. `labels.txt` (9-Class Alphabetical Mapping)
4. `model_contract.json` (Full I/O Contract and Literature Shelf-Life Rules)

---

## 🛠️ Step 1: `pubspec.yaml` Configuration

Add the required packages and asset paths:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.12.1
  image: ^4.5.4
  path_provider: ^2.1.5

flutter:
  assets:
    - assets/models/
```

---

## 🤖 Step 2: Codex Integration Prompts

You can copy and paste the following prompt directly into your Codex coding assistant:

### 💬 Prompt for Codex:
> "Please integrate the MobileNetV4 Enhanced v5 Fruit Ripeness model bundle from `assets/models/` into our Flutter app.
> 
> Follow these strict architectural requirements:
> 1. **Model Files**: Load `assets/models/mobilenetv4_fruit_float32.tflite` and `assets/models/u2net.tflite`.
> 2. **Inference Pipeline**:
>    - Step A: When a photo is uploaded or camera frame is captured, use `u2net.tflite` to predict the alpha mask (or fallback to color segmentation).
>    - Step B: Crop tightly around the fruit bounding box and letterbox onto a square black canvas `(0, 0, 0)` of size `max(width, height)`.
>    - Step C: Resize the square canvas to `224x224` and normalize into an NCHW `[1, 3, 224, 224]` float32 tensor with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
>    - Step D: Run inference synchronously on the native `Interpreter` (avoid `IsolateInterpreter` on Android to prevent isolate memory pointer loss).
>    - Step E: Decode raw logits using stable Softmax into 9 classes (`overripe-banana`, `overripe-mango`, `overripe-papaya`, `ripe-banana`, `ripe-mango`, `ripe-papaya`, `unripe-banana`, `unripe-mango`, `unripe-papaya`).
>    - Step F: Generate a Grad-CAM JET heatmap blend ($0.6 \times \text{RGB} + 0.4 \times \text{JET}$) and display postharvest shelf-life guidance.
> 3. **Multi-View UI**: On the assessment screen, provide a 3-way toggle between **`Isolated`** (default), **`Original`**, and **`Grad-CAM`**."

---

## 🔬 Critical Implementation Rules (Avoid Common Pitfalls)

| Bug / Pitfall | Root Cause | Solution |
| :--- | :--- | :--- |
| **Isolate Memory Dropping** | `IsolateInterpreter.run()` passes pointers across threads, dropping float buffer data on Android. | Use direct synchronous `_interpreter.run(shapedInput, outputLogits)` on the native C++ pointer. |
| **Edge Clipping on Rectangular Photos** | Traditional center-crop slices off the sides of wide fruits. | Always extract the fruit bounding box and letterbox onto a square black canvas before resizing to $224 \times 224$. |
| **Tensor Layout Mismatch** | MobileNetV4 v5 expects NCHW (`[1, 3, 224, 224]`), not NHWC. | Populate Plane 0 (Red), Plane 1 (Green), Plane 2 (Blue) sequentially in memory. |
| **Cache Invalidation** | Android package manager caches previous APK debug instances. | Run `flutter clean` before building your final debug APK. |

---

## 📊 9-Class Index Reference

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

---

## 🧪 Testing Your Implementation

Run the included reference script in your terminal to verify expected outputs:
```bash
python preprocess.py path/to/sample_fruit.jpg
```
