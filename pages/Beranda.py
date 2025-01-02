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
    margin-top: 30px; /* Adjusted for extra space */
    line-height: 1.8; /* Adjust spacing for better readability */
}
.right {
    text-align: right;
    margin-right: 50px;
    margin-top: 30px; /* Adjusted for extra space */
    line-height: 1.8; /* Adjust spacing for better readability */
}
.space-below {
    margin-bottom: 5px;
}
.space-between {
    margin-top: 20px; /* Space between NIM and dosen pembimbing */
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
st.markdown('<p class="center space-below">NIM. 24050121140106</p>', unsafe_allow_html=True)

# Additional space between NIM and dosen pembimbing
st.markdown('<div class="space-between"></div>', unsafe_allow_html=True)

# Display dosen pembimbing information
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<p class="left"><b>Dosen Pembimbing 1:</b><br><br>Dr. Triastuti Wuryandari, S.Si., M.Si.<br>NIP. 197109061998032001</p>', unsafe_allow_html=True)

with col_right:
    st.markdown('<p class="right"><b>Dosen Pembimbing 2:</b><br><br>Miftahul Jannah, S.Si., M.Si.<br>NIP. H.7.199804242023072001</p>', unsafe_allow_html=True)