import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model, predict
from datetime import datetime, timedelta

# Load daftar model
MODELS = ["Model XGBoost Default", "Model XGBoost GridSearchCV", "Model XGBoost PSO"]

# Halaman Visualisasi Grafik Prediksi
st.title("📊 Visualisasi Grafik Prediksi Saham Kalbe Farma (KLBF)")
st.write("Halaman ini menampilkan visualisasi grafik prediksi harga saham berdasarkan model yang dipilih.")

# Dropdown untuk memilih model
selected_model_name = st.selectbox("📌 Pilih Model Prediksi", MODELS)

# Load model berdasarkan pilihan
model = load_model(selected_model_name)

# Bagian input
st.sidebar.header("🔢 Input Data Prediksi")
input_method = st.sidebar.radio("📌 Pilih Metode Input", ["Manual", "Upload CSV"])

if input_method == "Manual":
    # Input manual
    open_prices = st.sidebar.text_area("📌 Harga Open (Pisahkan dengan koma)", "1530, 1540, 1550")
    high_prices = st.sidebar.text_area("📌 Harga High (Pisahkan dengan koma)", "1550, 1560, 1570")
    low_prices = st.sidebar.text_area("📌 Harga Low (Pisahkan dengan koma)", "1500, 1510, 1520")
    close_prices = st.sidebar.text_area("📌 Harga Close (Pisahkan dengan koma)", "1510, 1520, 1530")

    try:
        # Konversi input ke array numpy
        open_prices = np.array([float(x.strip()) for x in open_prices.split(",")])
        high_prices = np.array([float(x.strip()) for x in high_prices.split(",")])
        low_prices = np.array([float(x.strip()) for x in low_prices.split(",")])
        close_prices = np.array([float(x.strip()) for x in close_prices.split(",")])
        last_date = datetime.today()
    except ValueError:
        st.error("⚠️ Pastikan semua nilai input valid dan dipisahkan dengan koma.")
        open_prices, high_prices, low_prices, close_prices, last_date = [], [], [], [], datetime.today()

elif input_method == "Upload CSV":
    # Input melalui file CSV
    uploaded_file = st.sidebar.file_uploader("📤 Upload File CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # ✅ Pastikan file memiliki kolom yang dibutuhkan
            required_columns = ["Date", "Open", "High", "Low", "Close"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"⚠️ File CSV harus memiliki kolom: {', '.join(missing_columns)}")
                st.stop()

            # ✅ Konversi data ke tipe numerik
            df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")

            # Hapus baris yang memiliki NaN setelah konversi
            df.dropna(inplace=True)

            if df.empty:
                st.error("⚠️ Data setelah konversi tidak valid. Periksa kembali file CSV Anda.")
                st.stop()

            # Ambil data terakhir dari CSV untuk prediksi
            last_row = df.iloc[-1]
            open_prices = np.array([last_row["Open"]])
            high_prices = np.array([last_row["High"]])
            low_prices = np.array([last_row["Low"]])
            close_prices = np.array([last_row["Close"]])

            # Konversi tanggal dengan aman
            try:
                last_date = pd.to_datetime(last_row["Date"])
            except Exception:
                st.error("⚠️ Format tanggal pada file CSV tidak dikenali. Pastikan menggunakan format YYYY-MM-DD atau DD/MM/YYYY.")
                last_date = datetime.today()

        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat membaca file CSV: {e}")
            open_prices, high_prices, low_prices, close_prices, last_date = [], [], [], [], datetime.today()
    else:
        open_prices, high_prices, low_prices, close_prices, last_date = [], [], [], [], datetime.today()

# Prediksi harga penutupan
if st.sidebar.button("📈 Generate Predictions"):
    if len(open_prices) == len(high_prices) == len(low_prices) == len(close_prices) and len(open_prices) > 0:
        predictions = []
        for open_price, high_price, low_price, close_price in zip(open_prices, high_prices, low_prices, close_prices):
            prediction = predict(model, open_price, high_price, low_price, close_price)
            predictions.append(prediction)

        # ✅ Membuat DataFrame untuk visualisasi
        data = pd.DataFrame({
            "Date": [last_date - timedelta(days=i) for i in range(len(close_prices))][::-1],
            "Harga Aktual": close_prices,
            "Harga Prediksi": predictions
        })

        # ✅ Visualisasi menggunakan matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data["Date"], data["Harga Aktual"], label="Harga Aktual", marker='o', color="blue")
        ax.plot(data["Date"], data["Harga Prediksi"], label="Harga Prediksi", marker='x', color="orange")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        ax.set_title(f"📊 Visualisasi Prediksi - {selected_model_name}")
        ax.legend()

        # ✅ Tampilkan grafik
        st.pyplot(fig)

        # ✅ Tampilkan data dalam tabel
        st.write("📊 **Data Prediksi:**")
        st.dataframe(data)

        # ✅ Simpan hasil prediksi ke CSV
        prediction_csv_path = f"/content/prediksi_{selected_model_name.replace(' ', '_').lower()}.csv"
        data.to_csv(prediction_csv_path, index=False)
        st.write(f"📂 **Hasil prediksi telah disimpan di:** `{prediction_csv_path}`")

    else:
        st.error("⚠️ Jumlah nilai pada input harga tidak sama atau data kosong.")
