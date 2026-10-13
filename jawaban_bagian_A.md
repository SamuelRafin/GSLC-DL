# Bagian A: Analisis Konseptual

Nama: Samuel Rafin Djunaidi
NIM: 2802477383
Tugas: GSLC Sesi 5, Advanced CNNs and Transfer Learning

---

## 1. Efek Regularisasi pada Batch Normalization (Slide 6–8)

Saat training, Batch Normalization menormalkan aktivasi memakai mean dan varians mini-batch, bukan seluruh dataset (Ioffe & Szegedy, 2015). Isi batch diacak setiap epoch dan statistiknya selalu bergeser sedikit. Keluaran untuk satu gambar pun ikut bergantung pada 31 gambar lain di batch yang sama (batch size 32). Fluktuasi tersebut bekerja seperti noise pada aktivasi, dan model menjadi lebih sulit menghafal sampel tertentu. Saat inferensi noise itu hilang, karena BN beralih ke running mean dan running variance yang tetap.

Asumsi bahwa Dropout tidak diperlukan lagi bisa gagal. Pada batch yang sangat kecil, estimasi statistiknya terlalu berisik dan training menjadi tidak stabil. Pada batch yang sangat besar, noise-nya hampir hilang sehingga efek regularisasinya lemah. Head fully-connected tanpa BN juga tetap bisa overfitting. Terakhir, Dropout yang diletakkan sebelum BN menggeser varians antara mode train dan test (variance shift), sehingga running statistics tidak cocok lagi dan performa turun (Li dkk., 2019).

## 2. Degradasi Jaringan Dalam dan Formulasi Residual (Slide 10–12, Sesi 02)

Degradation problem bukan overfitting, karena yang naik justru training error (He dkk., 2016a). Secara teori, lapisan tambahan cukup mempelajari fungsi identitas agar model dalam setara dengan model dangkal. Kenyataannya, solver SGD sulit mendekati identitas lewat tumpukan konvolusi dan ReLU non-linear. Persoalannya optimasi, bukan kapasitas.

Skip connection mengubah target belajar menjadi residual: y = x + F(x). Kalau identitas memang optimal, jaringan cukup mendorong F(x) ke nol, dan itu jauh lebih mudah. Saat backpropagation, turunannya menjadi ∂L/∂x = ∂L/∂y · (I + ∂F/∂x). Suku I memberi gradien jalur langsung ke lapisan awal tanpa dikalikan bobot (He dkk., 2016b).

Di Sesi 02, vanishing gradient dijelaskan sebagai perkalian berantai turunan bernilai kurang dari satu, sehingga gradien menyusut eksponensial; jalur aditif tadi memutus rantai itu. Meski begitu, He dkk. (2016a) mencatat bahwa dengan BN gradien plain network tidak benar-benar lenyap; manfaat utama residual terletak pada optimasi yang lebih mudah.

## 3. Feature Extraction vs. Fine-Tuning (Slide 14–16)

Feature extraction hanya melatih head. Di B2 jumlahnya 5.130 dari 11.181.642 parameter, sedangkan fine-tuning layer4 dan fc pada B3 membuka 8.398.858 parameter. Konsekuensinya, feature extraction cukup dengan data sedikit, sementara fine-tuning butuh data lebih banyak agar jutaan bobot itu tidak overfitting. Di atas kertas fine-tuning lebih berat per epoch, sebab aktivasi dan gradien blok yang dibuka harus disimpan. Pada T4 selisihnya kecil (13,93 vs 14,19 detik per epoch) karena waktu lebih banyak habis di pipeline augmentasi CPU. Risiko destructive gradient updates hampir nol pada feature extraction, tetapi nyata pada fine-tuning, terutama bila head masih acak dan learning rate besar (Yosinski dkk., 2014).

Feature extraction cocok ketika data sangat terbatas, domain mirip ImageNet, atau komputasi sempit. Fine-tuning dengan learning rate kecil lebih tepat bila data cukup, domain berbeda, atau feature extraction mulai underfitting. Kornblith dkk. (2019) menunjukkan fine-tuning umumnya mengungguli fitur beku, terutama pada dataset fine-grained.

## 4. Kaidah Penerapan Data Augmentation (Slide 20, 21, 23)

Augmentasi adalah regularisasi yang hanya relevan selama model belajar (Shorten & Khoshgoftaar, 2019). Validation dan test set berfungsi sebagai alat ukur: keduanya harus deterministik dan mewakili citra yang ditemui saat model dipakai. Kalau transformasi acak ikut diterapkan, val loss berubah-ubah antar epoch, padahal angka itu dipakai untuk early stopping dan pemilihan checkpoint. Hasil test juga tidak reproducible. Karena itu notebook memakai eval transform tetap (Resize 256, CenterCrop 224, normalisasi), dan split dilakukan sebelum transformasi.

Contoh yang tidak layak untuk pet breeds adalah hue shift yang kuat atau grayscale acak. Pada Oxford-IIIT Pet (Parkhi dkk., 2012), warna bulu sering menjadi ciri ras, misalnya bulu kemerahan Abyssinian atau pola colorpoint Siamese. Mengubah warna secara drastis berarti mengubah informasi yang menentukan label. Label tidak lagi invarian terhadap transformasi itu, sehingga augmentasi justru menyuntikkan label noise. Itulah alasan ColorJitter di eksperimen ini hanya mengubah brightness dan contrast.

## 5. Mitigasi Catastrophic Forgetting (Slide 28–29)

Catastrophic forgetting muncul karena bobot pretrained hanya dijaga oleh loss pada data baru, bukan oleh tugas lamanya (Kirkpatrick dkk., 2017). Di awal fine-tuning head masih acak, dan gradien besar mengalir ke backbone. Dengan learning rate tinggi, bobot yang menyimpan fitur umum ImageNet bergeser jauh, padahal loss pada sekitar 1.400 citra latih tidak memberi insentif untuk mempertahankannya. Representasi lama akhirnya tertimpa fitur sempit yang mudah overfitting.

Strategi pertama adalah differential learning rate. Lapisan awal menyimpan fitur umum seperti tepi dan tekstur, lapisan atas lebih spesifik terhadap tugas (Yosinski dkk., 2014). Backbone diberi learning rate kecil dan head lebih besar, seperti di B3 (1e-4 untuk layer4, 1e-3 untuk fc), agar fitur umum hanya bergeser sedikit (Howard & Ruder, 2018).

Strategi kedua adalah gradual unfreezing. Head dilatih dulu sampai gradiennya bermakna, lalu blok dibuka bertahap dari atas ke bawah (Howard & Ruder, 2018), sehingga gradien acak dari head tidak sempat merusak backbone.

## Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016a). Deep Residual Learning for Image Recognition. CVPR.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016b). Identity Mappings in Deep Residual Networks. ECCV.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., dkk. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13).
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? CVPR.
- Li, X., Chen, S., Hu, X., & Yang, J. (2019). Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift. CVPR.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. CVPR.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6(60).
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? NeurIPS.
