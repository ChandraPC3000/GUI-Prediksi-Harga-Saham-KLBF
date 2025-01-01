import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model, predict

# Styling halaman
st.markdown("""
<style>
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown('<h1 class="center">Visualisasi Grafik Prediksi Harga Saham PT Kalbe Farma Tbk (KLBF)</h1>', unsafe_allow_html=True)
st.write("Halaman ini menampilkan grafik prediksi harga saham berdasarkan data input.")

# Pilihan model
MODELS = ["Model XGBoost Default", "Model XGBoost GridSearch", "Model XGBoost PSO",
          "Model LSTM Adam", "Model LSTM RMSprop"]

# Dropdown untuk memilih model
selected_model_name = st.selectbox("Pilih Model Prediksi", MODELS)

# Load model sesuai pilihan
model = load_model(selected_model_name)

# Upload file data atau masukkan data manual
uploaded_file = st.file_uploader(
    "Upload File CSV dengan kolom [Date, Open, High, Low, Close, Volume]", type=["csv"])

if uploaded_file:
    # Membaca file CSV
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.write("Data yang diunggah:")
    st.dataframe(df.head())

    # Prediksi untuk setiap baris
    predictions = []
    for index, row in df.iterrows():
        pred_close = predict(
            model,
            row["Open"],
            row["High"],
            row["Low"],
            row["Close"],
            selected_model_name
        )
        predictions.append(pred_close)

    # Menambahkan kolom prediksi ke data
    df["Predicted_Close"] = predictions

    # Visualisasi data aktual dan prediksi
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df["Close"], label="Actual Close Price", color="blue")
    ax.plot(df["Date"], df["Predicted_Close"], label="Predicted Close Price", color="orange")
    ax.set_title("Grafik Prediksi Harga Saham", fontsize=16)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Saham")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Unggah file data untuk melihat visualisasi grafik prediksi.")
