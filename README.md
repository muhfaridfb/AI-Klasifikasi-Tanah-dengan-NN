# 📌 Projek AI: Klasifikasi Kualitas Tanah untuk Rekomendasi Jenis Tanaman

Projek ini bertujuan untuk melakukan **klasifikasi kualitas tanah** menggunakan metode **Neural Network**. Hasil klasifikasi digunakan untuk merekomendasikan jenis tanaman yang sesuai berdasarkan karakteristik tanah. Model yang dibangun berhasil mencapai akurasi sebesar **91,41%** pada data uji.

---

## 📂 Struktur Folder & Deskripsi File

| Nama File / Folder               | Deskripsi                                                                 |
|:--------------------------------|:--------------------------------------------------------------------------|
| `model/`                         | Berisi file Python source code untuk proses pelatihan model neural network. |
| `Modelling.ipynb`                | Notebook Jupyter berisi proses modeling neural network secara interaktif. |
| `my_model.h5`                    | File model hasil pelatihan yang disimpan dalam format H5.                 |
| `soil.xls`                       | Dataset kualitas tanah yang digunakan dalam proses pelatihan dan pengujian.|
| `test.py`                        | Script Python untuk menjalankan proses prediksi menggunakan model terlatih.|
| `PPT Projek AI.pdf`              | File presentasi hasil projek AI.                                          |
| `Tugas Project AI_Kelompok 6.pdf`| Laporan lengkap hasil projek AI.                                          |
| `read.txt`                       | Catatan atau dokumentasi tambahan.                                        |

---

## 📑 Cara Menjalankan Projek

1. **Pastikan environment telah ter-install library berikut:**
   - TensorFlow
   - Pandas
   - NumPy
   - scikit-learn
   - openpyxl (jika membaca file `.xls/.xlsx`)

2. **Jalankan file `Modelling.ipynb`** untuk melihat proses training model.

3. **Gunakan `test.py`** untuk melakukan prediksi menggunakan file model `my_model.h5` yang sudah dilatih.

---

## 📊 Hasil Model
- Akurasi data uji: **91,41%**

---

## 📌 Credit
Projek dikembangkan oleh:  
**Kelompok 6 (Muh Farid FB, Sindi Aprilianti, Zhafran Agus, Azriel Teddy Muhammad**  
Mata Kuliah *Kecerdasan Buatan*  
Institut Pertanian Bogor  
2025
