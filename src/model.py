"""Definición de la arquitectura CNN."""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src import config


def build_augmentation():
    """Capas de Data Augmentation (solo activas durante el entrenamiento)."""
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            layers.RandomRotation(0.05, fill_mode="reflect"),
            layers.RandomZoom(0.1, fill_mode="reflect"),
        ],
        name="data_augmentation",
    )


def _conv_block(x, filters, dropout, wd):
    """Dos convoluciones 3x3 + BatchNorm + ReLU, MaxPooling y Dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_cnn(num_classes=config.NUM_CLASSES, augment=True, wd=config.WEIGHT_DECAY):
    """Construye la CNN tipo VGG para imágenes 32x32x3 normalizadas en [0, 1]."""
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS),
                          name="imagen")
    x = build_augmentation()(inputs) if augment else inputs

    x = _conv_block(x, 32, 0.20, wd)    # 32x32 -> 16x16
    x = _conv_block(x, 64, 0.30, wd)    # 16x16 -> 8x8
    x = _conv_block(x, 128, 0.40, wd)   # 8x8   -> 4x4

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilidades")(x)

    return models.Model(inputs, outputs, name="cnn_cifar10")


def compile_model(model, lr=config.LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model
