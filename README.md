# GSLC Sesi 5: Advanced CNNs dan Transfer Learning

NIM: 2802477383
Nama: Samuel Rafin Djunaidi
Repository: [link GitHub menyusul]

## Ringkasan

GSLC ini membandingkan beberapa strategi untuk klasifikasi 10 ras hewan dari dataset Oxford-IIIT Pet. Ada model ResNet-18 yang dilatih dari nol (B1), feature extraction dengan backbone beku (B2), progressive fine-tuning layer4 dengan differential learning rate (B3), ablation augmentasi (B4), dan perbandingan arsitektur dengan EfficientNet-B0 (B5). Selain itu ada failure analysis pada prediksi yang salah dan visualisasi Grad-CAM.

Jawaban teori ada di `jawaban_bagian_A.md`, sedangkan laporan eksperimen lengkap ada di `laporan.md` (dan versi PDF-nya di `laporan.pdf`).

## Kelas yang dipakai

Kucing (5): Abyssinian, Bengal, Birman, Persian, Siamese
Anjing (5): Beagle, Pug, Samoyed, Shiba Inu, Chihuahua

Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet`, lalu hanya 10 kelas di atas yang diambil (total 1997 gambar).

## Pembagian data

Split dilakukan secara stratified 70/15/15 dengan seed 42, menggunakan `train_test_split` dari scikit-learn pada level indeks sebelum transformasi apa pun dipasang. Hasilnya 1397 gambar train, 300 validation, dan 300 test. Augmentasi hanya diterapkan ke data train. Pemilihan model dan early stopping memakai validation loss, data test baru disentuh di akhir. Indeks lengkap tiap split disimpan di `outputs/results/split_indices.json` supaya bisa dicek ulang.

## Cara menjalankan di Google Colab

1. Buka `Tugas05_Notebook.ipynb` di Google Colab.
2. Pilih Runtime > Change runtime type > GPU (T4).
3. Pastikan `SMOKE = False` di sel konfigurasi. Mode `SMOKE = True` hanya untuk uji cepat memakai data sintetis dan hasilnya tidak bisa dipakai.
4. Jalankan Runtime > Run all. Di T4, waktu training kelima kondisi totalnya sekitar 20 menit, belum termasuk unduh dataset dan Grad-CAM.
5. Semua hasil tersimpan di `/content/outputs/`, dengan subfolder `logs/`, `results/`, `figures/`, `checkpoints/`, dan `data/`. Sel terakhir membuat `outputs.zip`, tapi zip itu ikut membawa `data/` dan `checkpoints/` sehingga ukurannya besar. Untuk pengumpulan cukup unduh `logs/`, `results/`, dan `figures/`.

Kalau ingin menjalankan secara lokal, install dependensi dengan `pip install -r requirements.txt`, lalu buka notebook dengan Jupyter. Saat ini `requirements.txt` baru mengunci torch dan torchvision, karena hanya dua versi itu yang tercatat di `environment.json`. Versi numpy, pandas, scikit-learn, matplotlib, dan Pillow bisa diambil dari Colab dengan perintah yang tertulis di file tersebut. GPU tetap disarankan.

## Struktur file pengumpulan

```
Tugas05_Notebook.ipynb     notebook utama (B1 sampai B5, failure analysis, Grad-CAM)
laporan.md                 laporan Bagian C
laporan.pdf                laporan Bagian C versi PDF
jawaban_bagian_A.md        jawaban analisis konseptual Bagian A
jawaban_bagian_A.pdf       jawaban Bagian A versi PDF
outputs/
  logs/                    history per epoch dan prediksi test untuk tiap kondisi
  results/                 raw_metrics.csv/json, tabel_konsolidasi.csv/md,
                           failure_analysis.csv, split_indices.json, environment.json
  figures/                 kurva training, gap B4, confusion matrix,
                           failure analysis, Grad-CAM sebelum/sesudah
README.md
requirements.txt
.gitignore
```

Folder `outputs/data/` (dataset hasil unduhan) dan `outputs/checkpoints/` (bobot model `.pt`) sengaja tidak ikut dikumpulkan karena ukurannya besar dan keduanya bisa dibuat ulang dengan menjalankan notebook.

## Hasil

Tabel perbandingan semua kondisi ada di `outputs/results/tabel_konsolidasi.md`, angka mentahnya di `outputs/results/raw_metrics.csv`. Secara singkat, accuracy test B1 hanya 0.52, sementara semua model transfer learning ada di kisaran 0.95 sampai 0.98. EfficientNet-B0 (B5) mendapat hasil tertinggi, yaitu accuracy test 0.9767 dan macro F1 0.9767. Pembahasan lengkapnya ada di `laporan.md`.

## Lingkungan

Eksperimen dijalankan di Google Colab dengan GPU Tesla T4, Python 3.13.15, torch 2.11.0+cu130, dan torchvision 0.26.0+cu130. Detailnya tercatat di `outputs/results/environment.json`.
