import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model, predict

# Load daftar model
MODELS = ["Model XGBoost Default", "Model XGBoost GridSearch", "Model XGBoost PSO",
          "Model LSTM Adam", "Model LSTM RMSprop"]

# Halaman Visualisasi Grafik Prediksi
st.title("Visualisasi Grafik Prediksi Saham Kalbe Farma (KLBF)")
st.write("Halaman ini menampilkan visualisasi grafik prediksi harga saham berdasarkan model yang dipilih.")

# Dropdown untuk memilih model
selected_model_name = st.selectbox("Pilih Model Prediksi", MODELS)

# Load model berdasarkan pilihan
model = load_model(selected_model_name)

# Input data untuk prediksi
st.sidebar.header("Input Data Prediksi")
open_prices = st.sidebar.text_area("Harga Open (Pisahkan dengan koma)", "1530, 1540, 1550")
high_prices = st.sidebar.text_area("Harga High (Pisahkan dengan koma)", "1550, 1560, 1570")
low_prices = st.sidebar.text_area("Harga Low (Pisahkan dengan koma)", "1500, 1510, 1520")
close_prices = st.sidebar.text_area("Harga Close (Pisahkan dengan koma)", "1510, 1520, 1530")

# Konversi input ke array numpy
try:
    open_prices = np.array([float(x) for x in open_prices.split(",")])
    high_prices = np.array([float(x) for x in high_prices.split(",")])
    low_prices = np.array([float(x) for x in low_prices.split(",")])
    close_prices = np.array([float(x) for x in close_prices.split(",")])
except ValueError:
    st.error("Pastikan semua nilai input valid dan dipisahkan dengan koma.")

# Prediksi harga penutupan
if st.sidebar.button("Generate Predictions"):
    if len(open_prices) == len(high_prices) == len(low_prices) == len(close_prices):
        predictions = []
        for open_price, high_price, low_price, close_price in zip(open_prices, high_prices, low_prices, close_prices):
            prediction = predict(model, open_price, high_price, low_price, close_price, selected_model_name)
            predictions.append(prediction)

        # Membuat DataFrame untuk visualisasi
        data = pd.DataFrame({
            "Index": range(1, len(predictions) + 1),
            "Harga Aktual": close_prices,
            "Harga Prediksi": predictions
        })

        # Visualisasi menggunakan matplotlib
        fig, ax = plt.subplots()
        ax.plot(data["Index"], data["Harga Aktual"], label="Harga Aktual", marker='o')
        ax.plot(data["Index"], data["Harga Prediksi"], label="Harga Prediksi", marker='x')
        ax.set_xlabel("Index")
        ax.set_ylabel("Harga")
        ax.set_title(f"Visualisasi Prediksi - {selected_model_name}")
        ax.legend()

        # Tampilkan grafik
        st.pyplot(fig)

        # Tampilkan data
        st.write("Data Prediksi:")
        st.dataframe(data)
    else:
        st.error("Jumlah nilai pada input harga tidak sama.")
