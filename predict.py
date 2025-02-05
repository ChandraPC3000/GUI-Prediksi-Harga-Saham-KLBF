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

# Fungsi untuk memastikan dataset memiliki MultiIndex dan normalisasi
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data[('Close', 'KLBF.JK')]
else:
    raise ValueError("Dataset tidak memiliki MultiIndex yang sesuai.")

# Pastikan kolom 'Close' adalah numerik
close_prices = pd.to_numeric(close_prices, errors='coerce')  # Konversi ke numerik

# Hapus nilai NaN (jika ada)
close_prices.dropna(inplace=True)

# Normalisasi kolom 'Close'
normalized_close = close_prices.apply(custom_min_max_scaler)

# Tambahkan hasil normalisasi ke dataset
data[('Normalized_Close', 'KLBF.JK')] = normalized_close

# Path CSV untuk model XGBoost yang berisi parameter
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_gridsearchcv_params.csv",
    "Model XGBoost PSO": "models/xgboost_pso_params.csv",
}

def load_model(model_name):
    """Membaca parameter XGBoost dari CSV dan membuat model."""
    
    model_path = MODELS_PATH.get(model_name)

    if model_path is None:
        raise ValueError(f"⚠️ Model '{model_name}' tidak ditemukan dalam MODELS_PATH.")

    st.write(f"🔍 Membaca model dari: {model_path}")

    try:
        model_params = pd.read_csv(model_path)
        st.write(f"📄 Isi CSV {model_name}:")
        st.write(model_params.head())

        if "Parameter" not in model_params.columns or "Value" not in model_params.columns:
            raise ValueError(f"⚠️ File {model_path} tidak memiliki kolom 'Parameter' dan 'Value'. Periksa formatnya.")

        params_dict = model_params.set_index("Parameter")["Value"].to_dict()

        params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
        params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
        params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
        params_dict["gamma"] = float(params_dict.get("gamma", 0))
        params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
        params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
        params_dict["subsample"] = float(params_dict.get("subsample", 1))
        params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))

        st.write("🔢 Parameter yang digunakan untuk model:", params_dict)

        model = XGBRegressor(**params_dict)
        return model

    except Exception as e:
        raise ValueError(f"⚠️ Terjadi kesalahan saat membaca model dari {model_path}:\n{str(e)}")

def predict(model, open_price, high_price, low_price, close_price):
    """Melakukan prediksi harga saham menggunakan model XGBoost."""
    input_data = np.array([
        [
            custom_min_max_scaler(open_price),
            custom_min_max_scaler(high_price),
            custom_min_max_scaler(low_price),
            custom_min_max_scaler(close_price)
        ]
    ])
    
    prediction = model.predict(input_data)
    return prediction[0]

# --- Streamlit UI ---
st.title("Prediksi Harga Saham KLBF dengan XGBoost")

model_option = st.selectbox("Pilih Model XGBoost:", list(MODELS_PATH.keys()))

if st.button("Muat Model"):
    model = load_model(model_option)
    st.success(f"Model {model_option} berhasil dimuat!")

open_price = st.number_input("Harga Open:", min_value=0.0, format="%.2f")
high_price = st.number_input("Harga High:", min_value=0.0, format="%.2f")
low_price = st.number_input("Harga Low:", min_value=0.0, format="%.2f")
close_price = st.number_input("Harga Close:", min_value=0.0, format="%.2f")

if st.button("Prediksi Harga Close"):
    if 'model' in locals():
        prediksi_harga = predict(model, open_price, high_price, low_price, close_price)
        st.success(f"Prediksi Harga Close (Normalisasi): {prediksi_harga:.2f}")
    else:
        st.error("Harap pilih dan muat model terlebih dahulu.")
