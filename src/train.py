"""Entrenamiento de la CNN sobre CIFAR-10.

Uso:
    python -m src.train                 # entrenamiento completo
    python -m src.train --epochs 5      # prueba rápida
"""
import argparse
import json
import time

import pandas as pd
import tensorflow as tf

from src import config
from src.data import get_splits, make_dataset
from src.evaluate import (compute_metrics, plot_confusion_matrix, plot_history,
                          plot_predictions, save_json)
from src.model import build_cnn, compile_model


def parse_args():
    p = argparse.ArgumentParser(description="Entrena la CNN de CIFAR-10")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--no-augment", action="store_true", help="Desactiva Data Augmentation")
    p.add_argument("--model-path", default=str(config.MODEL_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(config.SEED)
    for d in (config.MODELS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Datos ------------------------------------------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_splits()
    print(f"Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")
    train_ds = make_dataset(x_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(x_val, y_val, args.batch_size)

    # 2. Modelo -----------------------------------------------------------
    model = compile_model(build_cnn(augment=not args.no_augment), lr=args.lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_path, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-5, verbose=1),
    ]

    # 3. Entrenamiento ----------------------------------------------------
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    print(f"Tiempo de entrenamiento: {(time.time() - t0) / 60:.1f} min")

    hist = pd.DataFrame(history.history)
    hist.to_csv(config.HISTORY_PATH, index_label="epoch")
    plot_history(hist, config.FIGURES_DIR / "training_curves.png")

    # 4. Evaluación en el conjunto de prueba -----------------------------
    model = tf.keras.models.load_model(args.model_path)  # mejor checkpoint
    probs = model.predict(x_test, batch_size=256, verbose=0)
    metrics = compute_metrics(y_test, probs)
    save_json(metrics, config.METRICS_PATH)
    plot_confusion_matrix(y_test, probs.argmax(1), config.FIGURES_DIR / "confusion_matrix.png")
    plot_predictions(x_test, y_test, probs, config.FIGURES_DIR / "predictions.png", n=20)

    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(config.CLASS_NAMES, f, ensure_ascii=False)

    print("\n=== Resultados en TEST ===")
    for k, v in metrics.items():
        if k != "per_class":
            print(f"{k:>20}: {v:.4f}")
    print(f"\nModelo guardado en: {args.model_path}")


if __name__ == "__main__":
    main()
