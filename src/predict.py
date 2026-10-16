"""Predicción sobre una o varias imágenes con el modelo entrenado.

Uso:
    python -m src.predict ruta/imagen.jpg [otra.png ...] [--top-k 3]
"""
import argparse
import json
from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image

from src import config
from src.data import preprocess_pil_image


@lru_cache(maxsize=1)
def load_model(model_path=str(config.MODEL_PATH)):
    return tf.keras.models.load_model(model_path)


def load_class_names():
    if config.CLASS_NAMES_PATH.exists():
        with open(config.CLASS_NAMES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return config.CLASS_NAMES


def predict_image(img: Image.Image, model=None, class_names=None):
    """Devuelve (clase_predicha, confianza, vector_probabilidades)."""
    model = model or load_model()
    class_names = class_names or load_class_names()
    probs = model.predict(preprocess_pil_image(img), verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def main():
    p = argparse.ArgumentParser(description="Clasifica imágenes con la CNN de CIFAR-10")
    p.add_argument("images", nargs="+")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    class_names = load_class_names()
    for path in args.images:
        label, conf, probs = predict_image(Image.open(path), class_names=class_names)
        print(f"\n{path}\n  -> {label} ({conf:.2%})")
        for i in np.argsort(probs)[::-1][: args.top_k]:
            print(f"     {class_names[i]:<10} {probs[i]:.2%}")


if __name__ == "__main__":
    main()
