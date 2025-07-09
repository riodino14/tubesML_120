
# Prediksi Harga Rumah Boston dengan Model Machine Learning (from Scratch)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%2C%20NumPy%2C%20Streamlit-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Proyek ini merupakan implementasi *end-to-end* untuk memprediksi harga rumah di Boston menggunakan dataset klasik "Boston House Prices". Fokus utama proyek ini adalah membangun model regresi **Decision Tree, Random Forest, dan AdaBoost dari awal (from scratch)** hanya dengan menggunakan Python, NumPy, dan Pandas untuk memahami mekanisme internal algoritma tersebut.

Model terbaik kemudian di-deploy ke dalam sebuah aplikasi web interaktif menggunakan **Streamlit**.

###  Demo Aplikasi Interaktif

Aplikasi web memungkinkan pengguna untuk memasukkan nilai fitur-fitur rumah dan mendapatkan prediksi harga secara *real-time*.


*(Catatan: Ganti gambar di atas dengan screenshot aplikasi Anda sendiri)*

---

## 📋 Daftar Isi
- [Tujuan Proyek](#-tujuan-proyek)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Hasil dan Evaluasi Model](#-hasil-dan-evaluasi-model)
- [Struktur File](#-struktur-file)
- [Cara Menjalankan Proyek Secara Lokal](#-cara-menjalankan-proyek-secara-lokal)
- [Anggota Tim](#-anggota-tim)
- [Lisensi](#-lisensi)

---

## 🎯 Tujuan Proyek
Tujuan utama dari proyek ini adalah:
1.  **Implementasi dari Awal:** Membangun algoritma Decision Tree, Random Forest, dan AdaBoost Regressor tanpa *library* siap pakai seperti Scikit-learn untuk memperdalam pemahaman fundamental.
2.  **Analisis & Pra-pemrosesan:** Melakukan analisis data eksploratif (EDA), menangani outlier, dan melakukan *feature scaling*.
3.  **Optimisasi Model:** Menemukan hyperparameter terbaik menggunakan metode *Grid Search* dan *K-Fold Cross-Validation* untuk mendapatkan model yang robust.
4.  **Evaluasi Komparatif:** Membandingkan kinerja ketiga model berdasarkan metrik R² Score dan RMSE.
5.  **Deployment:** Membuat antarmuka pengguna (UI) yang interaktif dengan Streamlit untuk mendemonstrasikan kegunaan model.

---

## 📊 Dataset
Dataset yang digunakan adalah **Boston House Prices** yang bersumber dari [Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices). Dataset ini berisi 506 baris data dengan 14 atribut, termasuk variabel target `MEDV` (Median value of owner-occupied homes in $1000s).

Beberapa fitur utama yang digunakan:
- **CRIM:** Tingkat kejahatan per kapita.
- **RM:** Rata-rata jumlah kamar per rumah.
- **LSTAT:** Persentase populasi berstatus ekonomi rendah.
- **DIS:** Jarak ke pusat bisnis Boston.
- **PTRATIO:** Rasio murid-guru di kota.

---

## 🛠️ Metodologi
Alur kerja proyek ini dibagi menjadi beberapa tahap:

1.  **Pra-pemrosesan Data:**
    - Memuat dataset dan memberikan nama kolom yang sesuai.
    - Melakukan analisis data eksploratif (EDA) untuk memahami distribusi data, korelasi antar fitur, dan mendeteksi outlier.
    - Menerapkan **StandardScaler** untuk menstandarisasi skala fitur agar memiliki mean 0 dan varians 1.

2.  **Implementasi Model (from Scratch):**
    - **Decision Tree Regressor:** Dibangun dengan logika rekursif untuk membagi data berdasarkan *variance reduction*.
    - **Random Forest Regressor:** Merupakan ansambel dari beberapa Decision Tree yang dilatih pada sampel data *bootstrap*.
    - **AdaBoost Regressor:** Menggunakan Decision Tree sebagai *weak learner* dan secara iteratif melatih model baru untuk memperbaiki kesalahan model sebelumnya.

3.  **Hyperparameter Tuning:**
    - Sebuah *grid* parameter didefinisikan untuk setiap model.
    - **Grid Search** manual dikombinasikan dengan **5-Fold Cross-Validation** dilakukan untuk mencari kombinasi hyperparameter terbaik yang memberikan R² Score tertinggi pada data validasi.

---

## 📈 Hasil dan Evaluasi Model
Setelah melalui proses tuning dan validasi, **AdaBoost Regressor** terpilih sebagai model terbaik dengan performa paling unggul dan stabil.

#### Parameter Terbaik untuk AdaBoost:
```python
{
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 1
}

Metrik Evaluasi Final:
Metrik	Nilai
Test R² Score	0.87
Test RMSE	3.10 (setara ~$3,100)
Rata-rata R² (Cross-Val)	0.85
Train R² Score	0.97
Train RMSE	1.63

Analisis:

Model akhir mampu menjelaskan 87% varians pada data uji yang belum pernah dilihat sebelumnya.

Rata-rata kesalahan prediksi (RMSE) pada data uji adalah $3,100, yang merupakan hasil yang solid mengingat rentang harga rumah di dataset.

Performa yang kuat pada data validasi (rata-rata R² 0.85) menunjukkan bahwa model ini robust dan tidak mengalami overfitting yang parah.

📁 Struktur File
Generated code
.
├── app.py              # File utama aplikasi Streamlit
├── adaboost_model.py   # Berisi kelas custom untuk DecisionTree dan AdaBoost
├── adaboost_model.pkl  # File model akhir yang telah dilatih
├── scaler.pkl          # File scaler yang digunakan untuk transformasi data
├── requirements.txt    # Daftar dependensi/library yang dibutuhkan
└── README.md           # File dokumentasi ini
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
🚀 Cara Menjalankan Proyek Secara Lokal

Untuk menjalankan aplikasi ini di komputer Anda, ikuti langkah-langkah berikut:

Clone Repositori

Generated bash
git clone https://github.com/[USERNAME_ANDA]/[NAMA_REPO_ANDA].git
cd [NAMA_REPO_ANDA]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Buat Virtual Environment (Direkomendasikan)

Generated bash
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Instal Dependensi
Pastikan Anda memiliki file requirements.txt dengan isi sebagai berikut, lalu jalankan perintah di bawah.

Generated code
# requirements.txt
streamlit
numpy
pandas
scikit-learn
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Jalankan Aplikasi Streamlit

Generated bash
streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Aplikasi akan terbuka secara otomatis di browser Anda.

🧑‍💻 Anggota Tim

Proyek ini dikerjakan oleh:

Riodino Raihan - Profil GitHub (Ganti dengan link profil yang benar)

Faiq Misbah Yazdi - Profil GitHub (Ganti dengan link profil yang benar)

Puguh Aiman - Profil GitHub (Ganti dengan link profil yang benar)

📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.

