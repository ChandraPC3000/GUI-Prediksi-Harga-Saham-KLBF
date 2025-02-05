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
