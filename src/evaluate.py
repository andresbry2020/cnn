"""Funciones de visualización y evaluación reutilizables."""
import json

import matplotlib
matplotlib.use("Agg")  # backend sin ventana (scripts / servidores)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import config


def plot_history(history: pd.DataFrame, path):
    """Curvas de pérdida y exactitud (entrenamiento vs. validación)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = np.arange(1, len(history) + 1)
    axes[0].plot(epochs, history["loss"], label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], label="Validación")
    axes[0].set_title("Pérdida (cross-entropy)")
    axes[1].plot(epochs, history["accuracy"], label="Entrenamiento")
    axes[1].plot(epochs, history["val_accuracy"], label="Validación")
    axes[1].set_title("Exactitud (accuracy)")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión" + (" (normalizada por fila)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_predictions(x, y_true, probs, path, n=20, seed=config.SEED):
    """Cuadrícula de n predicciones (verde = acierto, rojo = error)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax, i in zip(axes.flat, idx):
        pred = int(np.argmax(probs[i]))
        ok = pred == y_true[i]
        ax.imshow(x[i])
        ax.set_title(f"Real: {config.CLASS_NAMES[y_true[i]]}\n"
                     f"Pred: {config.CLASS_NAMES[pred]} ({probs[i, pred]:.0%})",
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compute_metrics(y_true, probs):
    """Métricas globales y por clase sobre un conjunto etiquetado."""
    y_pred = probs.argmax(axis=1)
    top3 = np.mean([y in p for y, p in zip(y_true, np.argsort(probs, axis=1)[:, -3:])])
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": float(top3),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc_ovr_macro": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "per_class": {k: v for k, v in report.items() if k in config.CLASS_NAMES},
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
