import streamlit as st
import pandas as pd

# Styling for center alignment and layout adjustments
st.markdown("""
<style>
.center {
    text-align: center;
}
.left {
    text-align: left;
    margin-left: 50px;
    margin-top: 20px;
}
.right {
    text-align: right;
    margin-right: 50px;
    margin-top: 20px;
}
.space-below {
    margin-bottom: 5px;
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

# Title and information
st.markdown('<h1 class="center">Prediksi Harga Saham PT Kalbe Farma Tbk (KLBF) Menggunakan Metode XGBoost dan LSTM</h1>', unsafe_allow_html=True)
st.markdown('<p class="center space-below">Oleh:</p>', unsafe_allow_html=True)
st.markdown('<p class="center space-below"><b>Chandra Putra Ciptaningtyas</b></p>', unsafe_allow_html=True)
st.markdown('<p class="center">NIM: 123456789</p>', unsafe_allow_html=True)

# Display dosen pembimbing information
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<p class="left"><b>Dosen Pembimbing 1:</b><br>Dr. Triastuti Wuryandari, S.Si., M.Si.<br><i>NIP. 197109061998032001</i></p>', unsafe_allow_html=True)

with col_right:
    st.markdown('<p class="right"><b>Dosen Pembimbing 2:</b><br>Miftahul Jannah, S.Si., M.Si.<br><i>NIP. H.7.199804242023072001</i></p>', unsafe_allow_html=True)
