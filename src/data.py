"""Carga, preprocesamiento y división del dataset CIFAR-10."""
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config

HF_URL = ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
          "plain_text/{split}-00000-of-00001.parquet")
CACHE_FILE = Path.home() / ".keras" / "datasets" / "cifar10_uint8.npz"


def _load_from_huggingface():
    """Descarga CIFAR-10 desde el espejo oficial en Hugging Face (uoft-cs/cifar10)."""
    import pandas as pd

    def read(split):
        df = pd.read_parquet(HF_URL.format(split=split))
        x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                      for d in df["img"]])
        return x.astype("uint8"), df["label"].to_numpy().astype("int64")

    return read("train"), read("test")


def load_cifar10():
    """Descarga (si es necesario) y devuelve CIFAR-10 en formato numpy.

    Usa una caché local; si no existe intenta el espejo de Hugging Face
    (más rápido) y, si falla, ``tf.keras.datasets.cifar10``.

    Returns:
        (x_train_full, y_train_full), (x_test, y_test) con imágenes uint8
        (N, 32, 32, 3) y etiquetas enteras (N,).
    """
    if CACHE_FILE.exists():
        d = np.load(CACHE_FILE)
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    try:
        (x_train, y_train), (x_test, y_test) = _load_from_huggingface()
    except Exception as exc:  # sin pandas/pyarrow o sin acceso a HF
        print(f"Espejo HF no disponible ({exc}); usando keras.datasets...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train, y_test = y_train.flatten(), y_test.flatten()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    return (x_train, y_train), (x_test, y_test)


def normalize(images: np.ndarray) -> np.ndarray:
    """Normaliza píxeles uint8 [0, 255] -> float32 [0, 1]."""
    return images.astype("float32") / 255.0


def split_train_val(x, y, val_size=config.VAL_SIZE, seed=config.SEED):
    """División estratificada entrenamiento / validación."""
    return train_test_split(x, y, test_size=val_size, stratify=y, random_state=seed)


def get_splits():
    """Devuelve los tres conjuntos ya normalizados.

    - Entrenamiento: 45 000 imágenes
    - Validación:     5 000 imágenes (estratificado, sacado de train)
    - Prueba:        10 000 imágenes (conjunto de test oficial)
    """
    (x_full, y_full), (x_test, y_test) = load_cifar10()
    x_train, x_val, y_train, y_val = split_train_val(x_full, y_full)
    return (
        (normalize(x_train), y_train),
        (normalize(x_val), y_val),
        (normalize(x_test), y_test),
    )


def make_dataset(x, y, batch_size=config.BATCH_SIZE, shuffle=False):
    """Crea un tf.data.Dataset eficiente (shuffle + batch + prefetch)."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=config.SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Prepara una imagen arbitraria para la CNN.

    Convierte a RGB, redimensiona a 32x32 y normaliza a [0, 1].
    Devuelve un array de forma (1, 32, 32, 3).
    """
    img = img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
    arr = normalize(np.asarray(img))
    return np.expand_dims(arr, axis=0)
