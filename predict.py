import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Konstanta nilai minimum dan maksimum berdasarkan penjelasan
MIN_VALUE = 784.91
MAX_VALUE = 2271.54

# Fungsi Normalisasi
def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    st.write(f"🔍 Debugging - custom_min_max_scaler: value={value}, min_value={min_value}, max_value={max_value}")
    
    if value is None:
        raise ValueError("⚠️ Nilai yang diberikan ke custom_min_max_scaler() adalah None.")
    
    if not isinstance(value, (int, float)):
        raise TypeError(f"⚠️ Nilai harus berupa angka (int/float), bukan {type(value).__name__}: {value}")

    if min_value == max_value:
        raise ValueError("⚠️ min_value dan max_value tidak boleh sama karena akan menyebabkan pembagian oleh nol.")
    
    return 1 - ((value - min_value) / (max_value - min_value))

# Fungsi Denormalisasi
def denormalize(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return max_value - (value * (max_value - min_value))

# Fungsi untuk melatih dan menyimpan model XGBoost
def train_and_save_xgboost_model():
    """Melatih model XGBoost dengan parameter dari CSV dan menyimpannya ke file."""
    try:
        params_df = pd.read_csv("best_xgboost_model_params.csv")
        params_dict = params_df.set_index("Parameter")["Nilai Parameter"].to_dict()
        
        params_dict["max_depth"] = int(float(params_dict["max_depth"]))
        params_dict["gamma"] = float(params_dict["gamma"])
        params_dict["reg_lambda"] = float(params_dict["reg_lambda"])
        params_dict["learning_rate"] = float(params_dict["learning_rate"])
        params_dict["min_child_weight"] = float(params_dict["min_child_weight"])
        params_dict["subsample"] = float(params_dict["subsample"])
        params_dict["colsample_bytree"] = float(params_dict["colsample_bytree"])
        params_dict["random_state"] = int(float(params_dict["random_state"]))
        
        model = XGBRegressor(**params_dict)
        X_train = np.random.rand(100, 4)
        y_train = np.random.rand(100)
        model.fit(X_train, y_train)
        model.save_model("models/best_xgboost_model.json")
        st.write("✅ Model berhasil dilatih dan disimpan sebagai 'models/best_xgboost_model.json'")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat melatih dan menyimpan model: {str(e)}")

# Fungsi untuk memuat model yang telah dilatih
def load_trained_xgboost_model():
    """Memuat model XGBoost yang telah dilatih dan disimpan."""
    try:
        model = XGBRegressor()
        model.load_model("models/best_xgboost_model.json")
        st.write("✅ Model XGBoost berhasil dimuat dari 'models/best_xgboost_model.json'")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# --- Streamlit UI ---
st.title("Prediksi Harga Saham KLBF dengan XGBoost")

if st.button("Latih Ulang Model"):
    train_and_save_xgboost_model()

if st.button("Muat Model"):
    model = load_trained_xgboost_model()
    if model:
        st.success("✅ Model XGBoost berhasil dimuat!")
    else:
        st.error("❌ Model belum tersedia. Silakan latih ulang model terlebih dahulu.")

open_price = st.number_input("Harga Open:", min_value=0.0, format="%.2f")
high_price = st.number_input("Harga High:", min_value=0.0, format="%.2f")
low_price = st.number_input("Harga Low:", min_value=0.0, format="%.2f")
close_price = st.number_input("Harga Close:", min_value=0.0, format="%.2f")

if st.button("Prediksi Harga Close"):
    if 'model' in locals() and model is not None:
        try:
            input_data = np.array([
                [
                    custom_min_max_scaler(open_price),
                    custom_min_max_scaler(high_price),
                    custom_min_max_scaler(low_price),
                    custom_min_max_scaler(close_price)
                ]
            ])
            prediction = model.predict(input_data)
            prediction_close = denormalize(prediction[0])
            st.success(f"Prediksi Harga Close (Denormalisasi): {prediction_close:.2f}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
    else:
        st.error("Harap pilih dan muat model terlebih dahulu.")
