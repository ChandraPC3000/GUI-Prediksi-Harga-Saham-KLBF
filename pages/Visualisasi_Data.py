import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import load_model, predict

# Title for the page
st.title("Visualisasi Data Prediksi Saham Kalbe Farma (KLBF)")

# File uploader for CSV
uploaded_file = st.file_uploader(
    "Upload File CSV dengan header Date, Close, Open, High, Low, Volume", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])

    # Show the uploaded data
    st.write("### Data dari File CSV")
    st.write(df)

    # Plot the line chart
    st.write("### Visualisasi Data")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Harga Close", color="blue")
    ax.set_title("Visualisasi Harga Saham", fontsize=16)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga Close", fontsize=12)
    ax.legend()
    st.pyplot(fig)

    # Option to display more data
    st.write("### Statistik Data")
    st.write(df.describe())
else:
    st.info("Silakan upload file CSV untuk melihat visualisasi.")

# Footer or any additional information
st.markdown("""
---
**Note:** Pastikan file CSV memiliki header kolom yang sesuai dengan format: **Date, Close, Open, High, Low, Volume**.
""")
