import streamlit as st
import numpy as np
import pickle
from adaboost_model import AdaBoostR2  # kelas custom

# === Load scaler dan model ===
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("adaboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Prediksi Harga Rumah Boston", layout="centered")
st.title("🏠 Prediksi Harga Rumah Boston (MEDV) dengan AdaBoostR2")
st.markdown("Masukkan nilai fitur berikut untuk memprediksi harga rumah (dalam ribuan USD):")

# === Daftar fitur & deskripsi ===
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

feature_descriptions = {
    'CRIM': 'Tingkat kejahatan per kapita menurut kota',
    'ZN': 'Persentase tanah untuk rumah dengan ukuran lebih dari 25.000 sq.ft',
    'INDUS': 'Proporsi area bisnis non-retail per kota',
    'CHAS': 'Dummy variabel: 1 jika berbatasan dengan Sungai Charles, 0 jika tidak',
    'NOX': 'Konsentrasi oksida nitrogen (bagian per 10 juta)',
    'RM': 'Rata-rata jumlah kamar per rumah',
    'AGE': 'Persentase unit hunian yang dibangun sebelum 1940',
    'DIS': 'Jarak tertimbang ke lima pusat bisnis Boston',
    'RAD': 'Indeks aksesibilitas ke jalan raya',
    'TAX': 'Tarif pajak properti per $10.000',
    'PTRATIO': 'Rasio siswa terhadap guru menurut kota',
    'B': '1000(Bk - 0.63)^2, di mana Bk adalah proporsi orang kulit hitam',
    'LSTAT': 'Persentase status sosial ekonomi rendah di populasi'
}

user_inputs = []
col1, col2 = st.columns(2)

for i, feat in enumerate(feature_names):
    desc = feature_descriptions[feat]
    with (col1 if i % 2 == 0 else col2):
        st.markdown(f"**{feat}** - {desc}")
        value = st.number_input(
            f"Masukkan nilai {feat}",
            value=0.000,
            format="%.5f",  # agar pakai titik, bukan koma
            step=0.001
        )
        user_inputs.append(value)

if st.button("📊 Prediksi Harga Rumah"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]

    st.success(f"💰 Prediksi MEDV (harga rumah): **${pred:.2f} ribu dolar**")
