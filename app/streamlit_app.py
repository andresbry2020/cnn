"""Aplicación Streamlit: clasificación de imágenes con la CNN entrenada en CIFAR-10.

Ejecutar desde la raíz del repositorio:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import preprocess_pil_image  # noqa: E402

SAMPLES_DIR = ROOT / "app" / "sample_images"

st.set_page_config(page_title="Clasificador CIFAR-10 (CNN)", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def get_model():
    import tensorflow as tf
    return tf.keras.models.load_model(config.MODEL_PATH)


# --- Barra lateral -------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Acerca del modelo")
    st.markdown(
        "Red Neuronal Convolucional entrenada sobre **CIFAR-10** "
        "(60 000 imágenes RGB de 32×32 en 10 clases)."
    )
    st.markdown("**Clases:** " + ", ".join(config.CLASS_NAMES))
    metrics_file = config.METRICS_PATH
    if metrics_file.exists():
        import json
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        st.metric("Accuracy en test", f"{m['accuracy']:.2%}")
        st.metric("F1 macro en test", f"{m['f1_macro']:.3f}")
    st.caption("La imagen se redimensiona a 32×32 px antes de la predicción. "
               "Funciona mejor con un único objeto centrado.")

# --- Contenido principal -------------------------------------------------
st.title("🧠 Clasificador de imágenes con CNN")
st.write("Sube una imagen (avión, automóvil, pájaro, gato, ciervo, perro, rana, "
         "caballo, barco o camión) y el modelo predecirá su clase.")

if not config.MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en `{config.MODEL_PATH}`. "
             "Entrénalo primero con `python -m src.train`.")
    st.stop()

model = get_model()

uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

samples = sorted(SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
sample_choice = None
if samples:
    sample_choice = st.selectbox("...o prueba con una imagen de ejemplo del conjunto de test",
                                 ["(ninguna)"] + [p.name for p in samples])

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif sample_choice and sample_choice != "(ninguna)":
    image = Image.open(SAMPLES_DIR / sample_choice)

if image is not None:
    probs = model.predict(preprocess_pil_image(image), verbose=0)[0]
    idx = int(np.argmax(probs))
    label, conf = config.CLASS_NAMES[idx], float(probs[idx])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image.convert("RGB"), caption="Imagen cargada", use_container_width=True)
    with col2:
        st.subheader("Resultado")
        st.markdown(f"### **{label.capitalize()}**")
        st.metric("Confianza", f"{conf:.2%}")
        st.progress(conf)
        if conf < 0.5:
            st.warning("Confianza baja: la imagen podría no pertenecer a ninguna de las 10 clases.")

    st.subheader("Probabilidad por clase")
    df = (pd.DataFrame({"Clase": config.CLASS_NAMES, "Probabilidad": probs})
          .sort_values("Probabilidad", ascending=False))
    st.bar_chart(df.set_index("Clase"), horizontal=True)
    st.dataframe(df.style.format({"Probabilidad": "{:.2%}"}), hide_index=True,
                 use_container_width=True)
else:
    st.info("👆 Sube una imagen para obtener la predicción.")
