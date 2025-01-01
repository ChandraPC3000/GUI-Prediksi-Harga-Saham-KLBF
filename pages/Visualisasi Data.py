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
try:
    model = load_model(selected_model_name)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Upload file data atau masukkan data manual
uploaded_file = st.file_uploader(
    "Upload File CSV dengan kolom [Date, Open, High, Low, Close, Volume]", type=["csv"])

if uploaded_file:
    try:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        st.write("Data yang diunggah:")
        st.dataframe(df.head())

        # Validasi data
        required_columns = ["Date", "Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"File harus memiliki kolom: {', '.join(required_columns)}")
            st.stop()

        # Prediksi untuk setiap baris
        predictions = []
        for index, row in df.iterrows():
            try:
                pred_close = predict(
                    model,
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    selected_model_name
                )
                predictions.append(pred_close)
            except Exception as e:
                st.error(f"Error saat memproses baris ke-{index + 1}: {e}")
                predictions.append(None)

        # Menambahkan kolom prediksi ke data
        df["Predicted_Close"] = predictions

        # Visualisasi data aktual dan prediksi
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Date"], df["Close"], label="Actual Close Price", color="blue", linewidth=2)
        ax.plot(df["Date"], df["Predicted_Close"], label="Predicted Close Price", color="orange", linewidth=2)
        ax.set_title("Grafik Prediksi Harga Saham", fontsize=16)
        ax.set_xlabel("Tanggal", fontsize=12)
        ax.set_ylabel("Harga Saham", fontsize=12)
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Unggah file data untuk melihat visualisasi grafik prediksi.")
