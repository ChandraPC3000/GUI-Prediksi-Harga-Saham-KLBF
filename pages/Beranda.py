import streamlit as st

# Styling untuk mempercantik tampilan
st.markdown("""
<style>
    .title {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        color: #2E3B55;
    }
    .subtitle {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        color: #2E3B55;
        margin-top: 10px;
    }
    .description {
        font-size: 16px;
        text-align: justify;
        line-height: 1.8;
        color: #4A4A4A;
    }
    .container {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .section {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Judul utama
st.markdown('<h1 class="title">Prediksi Harga Saham PT Kalbe Farma Tbk</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Menggunakan Model Extreme Gradient Boosting (XGBoost)</h2>', unsafe_allow_html=True)

# Deskripsi aplikasi
st.markdown('<div class="container section">', unsafe_allow_html=True)
st.markdown("""
<p class="description">
Aplikasi ini dikembangkan untuk membantu analisis harga saham PT Kalbe Farma Tbk (KLBF) dengan menggunakan model **Extreme Gradient Boosting (XGBoost)**. 
Dengan antarmuka yang interaktif dan mudah digunakan, pengguna dapat melakukan prediksi harga saham berdasarkan data historis 
serta menganalisis pergerakan harga dengan berbagai metode optimasi, seperti **GridSearchCV dan Particle Swarm Optimization (PSO)**.
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Panduan Penggunaan
st.markdown('<div class="container section">', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Panduan Penggunaan</h2>', unsafe_allow_html=True)
st.markdown("""
<p class="description">
1. **Pilih Model Prediksi**: Pada sidebar, pengguna dapat memilih salah satu model prediksi, yaitu **XGBoost Default**, **XGBoost GridSearchCV**, atau **XGBoost PSO**.<br>
2. **Input Data**: Data dapat dimasukkan secara manual dengan mengisi harga open, high, low, dan close atau dengan mengunggah file CSV.<br>
3. **Lakukan Prediksi**: Setelah data diinput, klik tombol **"Predict"** untuk mendapatkan hasil prediksi harga penutupan saham.<br>
4. **Visualisasi Data**: Pengguna dapat melihat grafik pergerakan harga saham dan membandingkan hasil prediksi dengan data aktual pada halaman visualisasi.<br>
5. **Analisis Lebih Lanjut**: Data hasil prediksi dapat diunduh untuk analisis lebih lanjut.
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Kesimpulan
st.markdown('<div class="container section">', unsafe_allow_html=True)
st.markdown("""
<p class="description">
Aplikasi ini memberikan kemudahan dalam memprediksi harga saham KLBF secara real-time dan membantu pengguna dalam pengambilan keputusan investasi 
dengan data yang lebih akurat dan berbasis machine learning.
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
