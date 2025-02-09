import streamlit as st
import pandas as pd

# Styling for layout adjustments
st.markdown("""
<style>
.center {
    text-align: center;
}
.adjusted-left {
    margin-left: 20%; /* Geser sedikit ke tengah kiri */
    margin-top: 20px;
    line-height: 1.8;
    text-align: center;
}
.adjusted-right {
    margin-right: 20%; /* Geser sedikit ke tengah kanan */
    margin-top: 20px;
    line-height: 1.8;
    text-align: center;
}
.space-below {
    margin-bottom: 5px;
}
.space-between {
    margin-top: 20px; /* Space between sections */
}
.space-name {
    margin-top: 10px; /* Tambahkan jarak antara judul dan nama */
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# Logo placement
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('logo.png', width=200,)

with col3:
    st.write(' ')

# Title and Application Introduction
st.markdown('<h1 class="center">Sistem Prediksi Harga Saham PT Kalbe Farma Tbk</h1>', unsafe_allow_html=True)

st.markdown("""
### Selamat Datang di Aplikasi Prediksi Harga Saham

Aplikasi ini dikembangkan untuk membantu dalam analisis dan prediksi harga penutupan saham PT Kalbe Farma Tbk (**KLBF**) menggunakan model **Extreme Gradient Boosting (XGBoost)**. Dengan antarmuka berbasis **Streamlit**, aplikasi ini menyediakan kemudahan bagi pengguna dalam melakukan analisis data saham tanpa perlu keahlian teknis mendalam.

**Fitur Utama:**
- **Prediksi Harga Saham** menggunakan model XGBoost (Default, GridSearchCV, PSO).
- **Input Data** melalui metode manual atau unggahan file CSV.
- **Visualisasi Data** dalam bentuk grafik interaktif.
- **Evaluasi Model** untuk melihat performa prediksi.

""", unsafe_allow_html=True)

# User Guide Section
st.markdown("""
### Panduan Penggunaan Aplikasi

1. **Navigasi Menu**
   - **Prediksi Harga Saham** → Untuk melakukan prediksi harga saham berdasarkan input manual atau data CSV.
   - **Visualisasi Data** → Untuk melihat grafik prediksi harga saham.
   - **Evaluasi Model** → Menampilkan metrik evaluasi model XGBoost.

2. **Cara Melakukan Prediksi**
   - Pilih model prediksi yang diinginkan (*XGBoost Default, GridSearchCV, atau PSO*).
   - Input data harga saham secara manual atau unggah file CSV dengan format yang sesuai.
   - Klik tombol **Predict** untuk melihat hasil prediksi.

3. **Cara Mengunggah File CSV**
   - Pastikan file CSV memiliki kolom: `Date`, `Open`, `High`, `Low`, dan `Close`.
   - Format tanggal yang didukung: **YYYY-MM-DD**.
   - Ukuran file tidak boleh terlalu besar untuk menghindari keterlambatan proses.

4. **Menafsirkan Hasil Prediksi**
   - Hasil prediksi akan ditampilkan dalam bentuk angka dan grafik interaktif.
   - Gunakan informasi ini untuk membantu dalam analisis tren harga saham.

5. **Tips Penggunaan Optimal**
   - Gunakan data historis yang relevan agar hasil prediksi lebih akurat.
   - Jika terjadi error, periksa kembali format data sebelum mengunggah.
""", unsafe_allow_html=True)

# Footer
st.markdown("""
---
📌 *Untuk informasi lebih lanjut mengenai aplikasi ini, silakan hubungi pengembang atau lihat dokumentasi yang tersedia.*
""", unsafe_allow_html=True)
