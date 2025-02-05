import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import streamlit as st

# Konstanta nilai minimum dan maksimum berdasarkan penjelasan
MIN_VALUE = 784.91
MAX_VALUE = 2271.54

# Fungsi Normalisasi
def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return 1 - ((value - min_value) / (max_value - min_value))

# Fungsi Denormalisasi
def denormalize(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return max_value - (value * (max_value - min_value))

# Path ke file CSV model
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_gridsearchcv_params.csv",
    "Model XGBoost PSO": "models/xgboost_pso_params.csv",
}

# Load dataset latih untuk memastikan model bisa fit
TRAIN_DATA_PATH = "models/train_data.csv"  # Pastikan ada file dataset latih
TARGET_COLUMN = "Next_Day_Close"  # Sesuaikan dengan nama target

def load_model(model_name):
    """Membaca parameter XGBoost dari CSV dan membuat model yang sudah dilatih."""
    
    model_path = MODELS_PATH.get(model_name)

    if model_path is None:
        raise ValueError(f"⚠️ Model '{model_name}' tidak ditemukan dalam MODELS_PATH.")

    st.write(f"🔍 Membaca model dari: {model_path}")

    try:
        # Baca file CSV
        model_params = pd.read_csv(model_path)

        # Debug: Tampilkan isi CSV
        st.write(f"📄 Isi CSV {model_name}:")
        st.write(model_params.head())

        # Pastikan kolom yang dibutuhkan ada
        if "Parameter" not in model_params.columns or "Value" not in model_params.columns:
            raise ValueError(f"⚠️ File {model_path} tidak memiliki kolom 'Parameter' dan 'Value'. Periksa formatnya.")

        # Konversi CSV ke dictionary
        params_dict = model_params.set_index("Parameter")["Value"].to_dict()

        # Konversi tipe data
        params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
        params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
        params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
        params_dict["gamma"] = float(params_dict.get("gamma", 0))
        params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
        params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
        params_dict["subsample"] = float(params_dict.get("subsample", 1))
        params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))

        st.write("🔢 Parameter yang digunakan untuk model:", params_dict)

        # Buat model dengan parameter yang telah diperbaiki
        model = XGBRegressor(**params_dict)

        # Load data latih agar model bisa difit sebelum prediksi
        train_data = pd.read_csv(TRAIN_DATA_PATH)

        # Pisahkan fitur dan target
        X_train = train_data.drop(columns=[TARGET_COLUMN])
        y_train = train_data[TARGET_COLUMN]

        # Latih model dengan data latih
        st.write("🔄 Melatih model dengan dataset latih...")
        model.fit(X_train, y_train)

        st.write("✅ Model berhasil dilatih dan siap digunakan.")

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
