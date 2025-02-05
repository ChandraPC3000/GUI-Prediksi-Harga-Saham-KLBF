import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Konstanta nilai minimum dan maksimum berdasarkan penjelasan
MIN_VALUE = 784.91
MAX_VALUE = 2271.54

# Fungsi Normalisasi
def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return 1 - ((value - min_value) / (max_value - min_value))

# Fungsi Denormalisasi
def denormalize(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return max_value - (value * (max_value - min_value))

# Path CSV untuk model XGBoost yang berisi parameter
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_gridsearchcv_params.csv",
    "Model XGBoost PSO": "models/xgboost_pso_params.csv",
}

def load_model(model_name):
    """Membaca parameter XGBoost dari CSV dan membuat model."""
    
    st.write("🔍 Model yang diminta:", model_name)
    
    # Cek apakah model ada dalam dictionary MODELS_PATH
    if model_name not in MODELS_PATH:
        raise ValueError(f"Model '{model_name}' tidak ditemukan. Periksa apakah nama model cocok dengan yang ada di MODELS_PATH.")
    
    model_path = MODELS_PATH[model_name]

    if model_path.endswith(".csv"):
        try:
            model_params = pd.read_csv(model_path)

            # Debug: Lihat isi file CSV
            st.write("📄 Isi CSV model:", model_params.head())

            # Konversi ke dictionary
            params_dict = model_params.set_index("Parameter")["Value"].to_dict()

            # Debug: Pastikan semua parameter sudah terbaca
            st.write("✅ Parameter model:", params_dict)

            # Konversi tipe data
            params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
            params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
            params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
            params_dict["gamma"] = float(params_dict.get("gamma", 0))
            params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
            params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
            params_dict["subsample"] = float(params_dict.get("subsample", 1))
            params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))

            # Debug: Print parameter yang telah dikonversi
            st.write("🔢 Parameter yang digunakan:", params_dict)

            # Buat model dengan parameter yang telah diperbaiki
            model = XGBRegressor(**params_dict)

            return model

        except Exception as e:
            raise ValueError(f"⚠️ Terjadi kesalahan saat membaca model dari {model_path}:\n{str(e)}")

    else:
        raise ValueError("Model tidak ditemukan atau format tidak didukung.")

def predict(model, open_price, high_price, low_price, close_price):
    """Melakukan prediksi harga saham menggunakan model XGBoost."""

    # Normalisasi input data
    input_data = np.array([
        [
            custom_min_max_scaler(open_price),
            custom_min_max_scaler(high_price),
            custom_min_max_scaler(low_price),
            custom_min_max_scaler(close_price)
        ]
    ])
    
    # Prediksi harga saham (dalam skala normalisasi)
    prediction = model.predict(input_data)
    prediction_close = prediction[0]

    # Denormalisasi hasil prediksi
    prediction_close = denormalize(prediction_close)
    
    return prediction_close

# --- Streamlit UI ---
st.title("Prediksi Harga Saham KLBF dengan XGBoost")

# Pilih model dari CSV
model_option = st.selectbox("Pilih Model XGBoost:", list(MODELS_PATH.keys()))

if st.button("Muat Model"):
    model = load_model(model_option)
    st.success(f"Model {model_option} berhasil dimuat!")

# Input harga saham
open_price = st.number_input("Harga Open:", min_value=0.0, format="%.2f")
high_price = st.number_input("Harga High:", min_value=0.0, format="%.2f")
low_price = st.number_input("Harga Low:", min_value=0.0, format="%.2f")
close_price = st.number_input("Harga Close:", min_value=0.0, format="%.2f")

if st.button("Prediksi Harga Close"):
    if 'model' in locals():  # Pastikan model sudah dimuat
        prediksi_harga = predict(model, open_price, high_price, low_price, close_price)
        st.success(f"Prediksi Harga Close (Denormalisasi): {prediksi_harga:.2f}")
    else:
        st.error("Harap pilih dan muat model terlebih dahulu.")
