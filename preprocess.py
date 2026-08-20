"""Reference Preprocessing Implementation (Python)"""
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_input, crop_rect=None):
    img = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input.convert("RGB")
    if crop_rect is not None:
        x, y, w, h = crop_rect
        img = img.crop((x, y, x + w, y + h))
    else:
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    img_resized = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img_normalized, axis=0).astype(np.float32)
