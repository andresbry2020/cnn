"""Configuración global del proyecto (rutas, hiperparámetros y clases)."""
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "cnn_cifar10.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = REPORTS_DIR / "history.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Datos ---------------------------------------------------------------
IMG_SIZE = 32          # CIFAR-10: imágenes RGB de 32x32
NUM_CHANNELS = 3
NUM_CLASSES = 10
VAL_SIZE = 5000        # imágenes de entrenamiento reservadas para validación
SEED = 42

CLASS_NAMES = [
    "avión", "automóvil", "pájaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camión",
]
CLASS_NAMES_EN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# --- Entrenamiento -------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
