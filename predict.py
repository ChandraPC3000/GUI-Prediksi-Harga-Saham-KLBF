import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Konstanta nilai minimum dan maksimum berdasarkan penjelasan
MIN_VALUE = 784.91
MAX_VALUE = 2271.54

# Path model dari CSV
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_model_gridsearchcv_params.csv",
    "Model XGBoost PSO": "models/xgboost_model_pso_params.csv"
}

# Fungsi Normalisasi
def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return 1 - ((value - min_value) / (max_value - min_value))

# Fungsi untuk memuat model dari parameter CSV
def load_model(model_name):
    model_path = MODELS_PATH.get(model_name)
    if not model_path:
        raise ValueError("Model tidak ditemukan dalam daftar MODELS_PATH.")
    
    try:
        model_params = pd.read_csv(model_path)
        params_dict = model_params.set_index("Parameter")["Value"].to_dict()
        
        params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
        params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
        params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
        params_dict["gamma"] = float(params_dict.get("gamma", 0))
        params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
        params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
        params_dict["subsample"] = float(params_dict.get("subsample", 1))
        params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))
        
        model = XGBRegressor(**params_dict)
        return model
    except Exception as e:
        raise ValueError(f"❌ Terjadi kesalahan saat membaca parameter model dari CSV: {str(e)}")

# Fungsi prediksi
def predict(model, open_price, high_price, low_price, close_price):
    # Normalisasi input data
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
st.title("Prediksi Harga Saham KLBF dengan Model XGBoost")

model_option = st.selectbox("Pilih Model:", list(MODELS_PATH.keys()))

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
        st.success(f"Prediksi Harga Close: {prediksi_harga:.2f}")
    else:
        st.error("Harap pilih dan muat model terlebih dahulu.")
