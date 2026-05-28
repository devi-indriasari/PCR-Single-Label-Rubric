import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


MODEL_FILE = "RF_Count_Vectors.sav"
VECTORIZER_FILE = "count_vect_model.sav"
ENCODER_FILE = "encoder.sav"
STOPWORD_FILE = "stopword.txt"


def remove_upper_case(text):
    text = str(text)
    words = text.split()
    stripped = [w.title() if w.isupper() else w for w in words]
    return " ".join(stripped)


def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub("", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub("", text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def text_to_word_sequence_like_keras(text):
    """
    Fungsi ringan untuk meniru tensorflow.keras.preprocessing.text.text_to_word_sequence.
    Ini dipakai agar aplikasi Streamlit tidak perlu menginstall TensorFlow.
    """
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = text.lower()

    for char in filters:
        text = text.replace(char, " ")

    return text.split()


@st.cache_data
def load_stopwords():
    path = Path(STOPWORD_FILE)
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    return set(stopwords)


@st.cache_resource
def load_artifacts():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)

    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)

    return model, vectorizer, encoder


def preprocess_text(text, stopwords):
    text = remove_upper_case(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)

    tokens = text_to_word_sequence_like_keras(text)
    tokens = [token for token in tokens if token not in stopwords]

    return " ".join(tokens)


def predict_category(text):
    stopwords = load_stopwords()
    model, vectorizer, encoder = load_artifacts()

    clean_text = preprocess_text(text, stopwords)
    vectorized_text = vectorizer.transform([clean_text])

    prediction = model.predict(vectorized_text)

    try:
        label = encoder.inverse_transform(prediction)[0]
    except Exception:
        label = prediction[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vectorized_text)[0]

        try:
            class_labels = encoder.inverse_transform(model.classes_)
        except Exception:
            class_labels = model.classes_

        probabilities = pd.DataFrame({
            "Label": class_labels,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

    return label, clean_text, probabilities


st.set_page_config(
    page_title="Peer Code Review Feedback Classifier",
    page_icon="💬",
    layout="centered"
)

st.title("Peer Code Review Feedback Classifier")
st.write(
    "Aplikasi ini mengklasifikasikan komentar peer code review mahasiswa "
    "ke dalam kategori rubrik kualitas kode menggunakan model Random Forest dan Count Vectorizer."
)

input_text = st.text_area(
    "Masukkan komentar peer code review:",
    height=180,
    placeholder="Contoh: Penamaan variabel sudah cukup jelas, tetapi masih bisa dibuat lebih konsisten."
)

if st.button("Klasifikasikan"):
    if not input_text.strip():
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        try:
            label, clean_text, probabilities = predict_category(input_text)

            st.subheader("Hasil Prediksi")
            st.success(f"Kategori: {label}")

            with st.expander("Lihat hasil preprocessing"):
                st.write(clean_text)

            if probabilities is not None:
                st.subheader("Probabilitas Tiap Label")
                st.dataframe(probabilities, use_container_width=True)
                st.bar_chart(probabilities.set_index("Label"))

        except FileNotFoundError as e:
            st.error(f"File tidak ditemukan: {e}")
            st.info(
                "Pastikan RF_Count_Vectors.sav, count_vect_model.sav, "
                "encoder.sav, dan stopword.txt sudah berada dalam folder yang sama."
            )
        except Exception as e:
            st.error("Terjadi error saat melakukan prediksi.")
            st.exception(e)