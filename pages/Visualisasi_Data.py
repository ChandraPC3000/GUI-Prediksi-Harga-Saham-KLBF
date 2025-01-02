import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import load_model, predict
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Title for the page
st.title("Visualisasi Data dan Prediksi Saham Kalbe Farma (KLBF)")

# File uploader for CSV
uploaded_file = st.file_uploader(
    "Upload File CSV dengan header Date, Close, Open, High, Low, Volume", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])

    # Show the uploaded data
    st.write("### Data dari File CSV")
    st.write(df)

    # Plot the actual data
    st.write("### Visualisasi Data Aktual")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Harga Close (Aktual)", color="blue")

    # Predict future values
    model_name = "Model XGBoost PSO"  # Default model
    model = load_model(model_name)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[["Open", "High", "Low", "Close"]])

    # Generate predictions for 5-10 years ahead
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1], periods=252 * 10, freq="B")  # Business days for 10 years
    predictions = []
    last_data = data_scaled[-1, :]  # Last row of scaled data

    for _ in range(len(future_dates)):
        prediction = model.predict(last_data.reshape(1, -1))[0]
        predictions.append(prediction)
        # Update last_data with the predicted value for chaining predictions
        last_data = np.roll(last_data, -1)
        last_data[-1] = prediction

    # Rescale predictions back to original scale
    predictions_rescaled = scaler.inverse_transform(
        np.column_stack([np.zeros_like(predictions)] * 3 + [predictions]))[:, -1]

    # Plot predictions
    ax.plot(future_dates, predictions_rescaled, label="Harga Close (Prediksi)", color="orange", linestyle="--")

    # Customize plot
    ax.set_title("Visualisasi Harga Saham dan Prediksi", fontsize=16)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga Close", fontsize=12)
    ax.legend()
    st.pyplot(fig)

    # Option to display more data
    st.write("### Statistik Data")
    st.write(df.describe())
else:
    st.info("Silakan upload file CSV untuk melihat visualisasi dan prediksi.")

# Footer or any additional information
st.markdown("""
---
**Note:** Pastikan file CSV memiliki header kolom yang sesuai dengan format: **Date, Close, Open, High, Low, Volume**.
Prediksi menggunakan model XGBoost PSO dan hanya bersifat simulasi.
""")
