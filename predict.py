import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import streamlit as st

# Path ke file CSV model dan dataset latih
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_gridsearchcv_params.csv",
    "Model XGBoost PSO": "models/xgboost_pso_params.csv",
}

TRAIN_DATA_PATH = "models/train_data.csv"  # Dataset latih
TARGET_COLUMN = "Next_Day_Close"
FEATURES = ["Open", "High", "Low", "Close", "Volume"]  # Fitur yang digunakan dalam model

# Konstanta nilai min & max berdasarkan dataset
MIN_VALUE = 784.91
MAX_VALUE = 2271.54

# ✅ Fungsi Normalisasi
def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return 1 - ((value - min_value) / (max_value - min_value))

# ✅ Fungsi Denormalisasi
def denormalize(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return max_value - (value * (max_value - min_value))

def load_model(model_name):
    """Membaca parameter XGBoost dari CSV dan melatih model sebelum digunakan."""
    
    model_path = MODELS_PATH.get(model_name)

    if model_path is None:
        raise ValueError(f"⚠️ Model '{model_name}' tidak ditemukan dalam MODELS_PATH.")

    st.write(f"🔍 Membaca model dari: {model_path}")

    try:
        # Baca file CSV model
        model_params = pd.read_csv(model_path)

        # Pastikan file memiliki kolom yang sesuai
        if "Parameter" not in model_params.columns or "Value" not in model_params.columns:
            raise ValueError(f"⚠️ File {model_path} tidak memiliki kolom 'Parameter' dan 'Value'.")

        # Konversi CSV ke dictionary
        params_dict = model_params.set_index("Parameter")["Value"].to_dict()

        # Konversi tipe data ke float/int sesuai kebutuhan
        params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
        params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
        params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
        params_dict["gamma"] = float(params_dict.get("gamma", 0))
        params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
        params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
        params_dict["subsample"] = float(params_dict.get("subsample", 1))
        params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))

        st.write("🔢 Parameter yang digunakan:", params_dict)

        # ✅ Buat model dengan parameter yang telah diperbaiki
        model = XGBRegressor(**params_dict)

        # ✅ Pastikan model sudah dilatih sebelum digunakan
        train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)

        # Pastikan indeks dalam format datetime
        train_data.index = pd.to_datetime(train_data.index)

        # Pisahkan fitur dan target berdasarkan data latih yang sudah dibagi sebelumnya
        total_data = len(train_data)
        train_size = int(total_data * 0.8)  # 80% training
        test_size = total_data - train_size  # 20% testing

        X_train = train_data[FEATURES].iloc[:train_size].copy()
        y_train = train_data[TARGET_COLUMN].iloc[:train_size].copy()

        X_test = train_data[FEATURES].iloc[train_size:].copy()
        y_test = train_data[TARGET_COLUMN].iloc[train_size:].copy()

        st.write(f"✅ Jumlah data total: {total_data}")
        st.write(f"✅ Jumlah data training: {len(X_train)}")
        st.write(f"✅ Jumlah data testing: {len(X_test)}")

        st.write(f"📅 Tanggal data training: {X_train.index.min().date()} - {X_train.index.max().date()}")
        st.write(f"📅 Tanggal data testing: {X_test.index.min().date()} - {X_test.index.max().date()}")

        # ✅ Latih model dengan dataset latih
        st.write("🔄 Melatih model dengan dataset latih...")
        model.fit(X_train, y_train)  

        st.write("✅ Model berhasil dilatih dan siap digunakan.")

        return model

    except Exception as e:
        raise ValueError(f"⚠️ Terjadi kesalahan saat membaca model dari {model_path}:\n{str(e)}")

def predict(model, open_price, high_price, low_price, close_price):
    """Melakukan prediksi harga saham dengan model XGBoost."""
    try:
        # ✅ Normalisasi input
        open_price = custom_min_max_scaler(open_price)
        high_price = custom_min_max_scaler(high_price)
        low_price = custom_min_max_scaler(low_price)
        close_price = custom_min_max_scaler(close_price)

        # ✅ Siapkan input data dalam bentuk array
        input_data = pd.DataFrame([[open_price, high_price, low_price, close_price]], columns=FEATURES[:-1])

        # ✅ Lakukan prediksi dengan model
        prediction_normalized = model.predict(input_data)[0]

        # ✅ Denormalisasi hasil prediksi
        prediction_denormalized = denormalize(prediction_normalized)

        return prediction_denormalized

    except Exception as e:
        raise ValueError(f"⚠️ Terjadi kesalahan saat melakukan prediksi:\n{str(e)}")
