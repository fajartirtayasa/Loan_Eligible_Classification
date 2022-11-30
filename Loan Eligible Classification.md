

# Laporan Proyek Machine Learning Terapan : Predictive Analytics



## [Problem Domain]

Perusahaan *Dream Housing Finance* menangani semua pinjaman rumah. Mereka memiliki riwayat pemberian pinjaman untuk semua daerah  perkotaan, semi-perkotaan, dan pedesaan. 

Perusahaan ingin mengotomatisasi proses kelayakan pinjaman (secara real-time) berdasarkan detail pelanggan yang diberikan saat mengisi formulir aplikasi online. Rincian tersebut adalah Jenis Kelamin, Status Perkawinan, Pendidikan, Jumlah Tanggungan, Penghasilan, Jumlah Pinjaman, Riwayat Kredit, dan lain-lain. Untuk mengotomatisasi proses ini, mereka telah memberikan masalah untuk mengidentifikasi segmen pelanggan, yang  berhak atas jumlah pinjaman sehingga mereka dapat secara khusus  menargetkan pelanggan tersebut.

Otomatisasi proses pinjaman ini menjadi hal penting karena kualitas pinjaman serta prosedur pinjaman menjadi dua di antara faktor-faktor yang memberikan pengaruh signifikan terhadap keputusan pelanggan dalam pengambilan pinjaman di suatu perusahaan tertentu. [1]

Berdasarkan permasalahan tersebut, penulis ingin membuat sistem yang dapat memprediksi kelayakan pelanggan untuk mendapat pinjaman agar proses validasi dapat berjalan lebih efektif dan efisien.



## [Business Understanding]

#### Problem Statement

Berdasarkan *problem domain* di atas, maka diperoleh *problem statement* pada proyek ini, yaitu:

- Bagaimana *data preparation* yang perlu dilakukan pada dataset Loan Eligible Classification untuk membangun model *machine learning* yang baik?
- Bagaimana cara memilih atau membuat model *machine learning* tebaik untuk memprediksi kelayakan calon peminjam di perusahaan *Dream Housing Finance*?

#### Goals

Berdasarkan *problem statement* di atas, maka diperoleh tujuan dari proyek ini, yaitu:

- Untuk melakukan tahap persiapan data atau *data preparation*, agar data yang digunakan dapat dipakai untuk melatih model *machine learning* dengan baik.
- Untuk membuat model *machine learning* dalam memprediksi kelayakan calon peminjam di perusahaan *Dream Housing Finance* dengan performa ketepatan klasifikasi cukup baik.

#### Solution Statement

Berdasarkan *problem statement* dan *goals* di atas, maka berikut beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek  ini, yaitu:

- Melakukan proses *Data Preparation* untuk memastikan data siap digunakan untuk pemodelan. Beberapa tahapan yang dapat dilakukan, antara lain *drop column* yang tidak diperlukan, *handling missing values*, mengganti tipe data, *handling outliers* pada kolom dengan tipe data numerikal, *feature engineering*, *encoding categorical label*, *handling imbalance data* pada fitur target, *splitting data*, serta standarisasi.
- Cara memilih atau membuat model machine learning terbaik akan dilakukan dengan beberapa tahapan, sebagai berikut:
  - Menggunakan LazyPredict untuk membandingkan 20+ algoritma machine learning untuk kasus klasifikasi.
  - Memilih 5 algoritma yang konsisten memberikan performa yang baik terhadap data latih.
  - Melakukan improvement pada baseline 5 model dengan *hyperparameter tuning*.
  - Mengevaluasi performa 5 model yang telah dilatih menggunakan *Confusion Matrix* dan *Classification Report*.



## [Data Understanding]

Data yang digunakan dalam *predictive analytics* ini merupakan data yang yang bersumber dari kaggle. Dataset ini berisikan detail informasi setiap calon peminjam yang diberikan saat mengisi aplikasi online milik perusahaan *Dream Housing Finance*.

*Source Dataset*: [Loan Eligible Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset?select=loan-train.csv)

#### Variabel-Variabel pada Loan Eligible Dataset

- **Loan_ID**: ID unik pinjaman.
- **Gender**: Male/Female.
- **Married**: Status pernikahan peminjam (Yes/No).
- **Dependents**: Banyaknya tanggungan peminjam.
- **Education**: Status pendidikan peminjam (Graduate/Under Graduate).
- **Self_Employed**: Status wiraswasta (Yes/No).
- **ApplicantIncome**: Pendapatan peminjam.
- **CoapplicantIncome**: Pendapatan pendamping peminjam.
- **LoanAmount**: Banyaknya pinjaman (dalam ribuan).
- **Loan_Amount_Term**: Jangka waktu pinjaman (dalam bulan).
- **Credit_History**: Riwayat credit memenuhi pedoman {1:'Good', 0:'Bad'}.
- **Property_Area**: Urban/Semi-Urban/Rural.
- **Loan_Status**: Status penerimaan pinjaman (Yes/No).

#### Descriptive Statistic

Tabel 1. Descriptive Statistic Data Numerikal

|       | ApplicantIncome | CoapplicantIncome | LoanAmount | Loan_Amount_Term | Credit_History |
| ----: | --------------: | ----------------: | ---------: | ---------------: | -------------: |
| count |          614.00 |            614.00 |     592.00 |           600.00 |         564.00 |
|  mean |         5403.46 |           1621.25 |     146.41 |           342.00 |           0.84 |
|   std |         6109.04 |           2926.25 |      85.59 |            65.12 |           0.36 |
|   min |          150.00 |              0.00 |       9.00 |            12.00 |           0.00 |
|   25% |         2877.50 |              0.00 |     100.00 |           360.00 |           1.00 |
|   50% |         3812.50 |           1188.50 |     128.00 |           360.00 |           1.00 |
|   75% |         5795.00 |           2297.25 |     168.00 |           360.00 |           1.00 |
|   max |        81000.00 |          41667.00 |     700.00 |           480.00 |           1.00 |

Tabel 2. Descriptive Statistic Data Kategorikal

|        |  Loan_ID | Gender | Married | Dependents | Education | Self_Employed | Property_Area | Loan_Status |
| -----: | -------: | -----: | ------: | ---------: | --------: | ------------: | ------------: | ----------: |
|  count |      614 |    601 |     611 |        599 |       614 |           582 |           614 |         614 |
| unique |      614 |      2 |       2 |          4 |         2 |             2 |             3 |           2 |
|    top | LP001002 |   Male |     Yes |          0 |  Graduate |            No |     Semiurban |           Y |
|   freq |        1 |    489 |     398 |        345 |       480 |           500 |           233 |         422 |

Beberapa informasi yang dapat diperoleh dari tahapan Descriptive Statistic berdasarkan Tabel 1 dan Tabel 2 di atas adalah:

- Dataset ini terdiri atas 614 baris dan 13 kolom (5 kolom dengan tipe data numerik, dan 8 kolom dengan tipe data object).
- Beberapa kolom dengan tipe data numerikal terdindikasi memiliki *outlier*.
- Terdapat beberapa kolom yang memiliki *missing value*, yakni  kolom 'Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount',  dan 'Credit_History'. Dengan demikian, nantinya perlu adanya proses *handling missing value*.
- Tidak terdapat data yang duplikat pada dataset.
- Terdapat beberapa kolom yang baiknya dilakukan proses perubahan tipe data menjadi *integer*, yakni untuk kolom 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', dan 'Credit_History'.
- Sebelum dilakukan pemodelan, kolom 'Loan_ID' dapat di-drop karena tidak dibutuhkan.

#### Univariate Analysis

<img src='https://drive.google.com/uc?export=view&id=1B7JLvuEjHvRaxHSafiWUDq7to3p6ZfCI'>

Gambar 1. Count Plot untuk Kolom 'Loan_Status'

<img src='https://drive.google.com/uc?export=view&id=1dtjelcD9bpxBP4N_j_1avm-dJvJGoNNa'>

Gambar 2. Histogram untuk Kolom dengan Tipe Data Numerikal

<img src='https://drive.google.com/uc?export=view&id=1vhKCPxd2gDBXXv7Ek7vEYBl0p3jHf7Ag'>

Gambar 3. Boxplot untuk Kolom dengan  Tipe Data Numerikal



Beberapa informasi yang dapat diperoleh dari tahapan Univariate Analysis berdasarkan Gambar 1, Gambar 2, dan Gambar 3 di atas adalah:

- Secara umum, untuk kolom dengan tipe data kategorikal, hampir semua  kolom memiliki kategori dominan yang ditunjukkan dengan persentase  >50%. Secara khusus untuk kolom 'Loan_Status', perbandingan  kategorinya adalah 69:31 (imbalance), artinya perlu dilakukan perlakuan  khusus agar nantinya model yang dibuat tidak ada kecenderungan prediksi  berlebih ke kategori tertentu.
- Terdapat outlier pada kolom-kolom dengan tipe data numerik, yakni  kolom 'ApplicantIncome', 'CoapplicantIncome', dan 'LoanAmount'. Nantinya akan dilakukan proses ***handling outlier*** pada tahap *data preparation*.

#### Multivariate Analysis

<img src='https://drive.google.com/uc?export=view&id=1bOdogO0urVUdDuyUg0CUWHsakhkqiaIX'>

Gambar 4. Bar Chart untuk Banyak Status Pinjaman berdasarkan Riwayat Kredit Pinjaman

<img src='https://drive.google.com/uc?export=view&id=1IP4BVn3kpUuHvfDpdS3-v9Fxg_lNTGIU'>

Gambar 5. Heatmap untuk Kolom dengan Tipe Data Numerikal



Beberapa informasi yang dapat diperoleh dari tahapan Multivariate Analysis berdasarkan Gambar 4 dan Gambar 5 di atas adalah:

- Riwayat peminjam yang buruk cenderung ditolak pengajuan pinjamannya. Karena missing value dari riwayat peminjam didominasi oleh yang  diterima pinjamannya, maka nantinya missing value ini akan diisi dengan nilai 1 (Good).
- Terdapat korelasi yang sedang antara kolom 'LoanAmount' dengan 'ApplicantIncome'.



## [Data Preparation]

Berikut adalah beberapa teknik yang digunakan dalam Data Preparation:

- ***Handling Missing Values***. Beberapa kolom yang memiliki *missing value*, yakni kolom 'Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount',  dan 'Credit_History'. Untuk kolom dengan tipe data kategorikal akan diisi dengan value yang paling banyak muncul di kolom tersebut, sedangkan untuk kolom dengan tipe data numerikal akan diisi dengan nilai median untuk menghindari bias karena adanya outlier.
- **Mengganti Tipe Data**. Beberapa kolom yang baiknya dilakukan proses perubahan tipe data menjadi *integer*, yakni untuk kolom 'CoapplicantIncome', 'LoanAmount', dan 'Credit_History'.
- ***Handling Outliers***. Outlier dapat ditemukan pada kolom kolom 'ApplicantIncome', 'CoapplicantIncome', dan 'LoanAmount'. Hal ini harus diatasi karena adanya outlier akan membuat model yang dibangun memiliki performa yang buruk. Proses *handling* akan dilakukan dengan metode ***Interquartile Range*** dengan formula sebagai berikut.
- ***Feature Engineering***. Setelah proses *handling outliers* ternyata didapati bahwa distribusi data pada kolom numerikal masih cenderung *skew* ke arah tertentu. Proses *feature engineering* yang akan dilakukan adalah **Transformasi** data dengan fungsi tertentu dengan tujuan mengurangi skew dan distribusi data terlihat lebih normal.
- ***Encoding Categorical Label***. Model yang akan dibangun tidak dapat menerima data dengan tipe kategorikal. Oleh karena itu proses ini perlu dilakukan untuk memecah kategori-kategori dari suatu kolom menjadi beberapa kolom terpisah dengan value 0 atau 1.
- ***Handling Imbalance Data***. Kategori pada kolom 'Loan_Status' memiliki perbandingan 69:31 (imbalance), artinya perlu dilakukan perlakuan  khusus agar nantinya model yang dibuat tidak ada kecenderungan prediksi berlebih ke kategori tertentu. Proses *handling* ini akan dilakukan dengan metode **SMOTE**.
- ***Splitting Data***. Ini merupakan proses membagi dataset menjadi data latih (train) dan data uji (test). Nantinya data train akan dilatih untuk membangun model, sedangkan data test akan digunakan untuk menguji seberapa baik generalisasi model terhadap data baru.
- ***Standardization***. Proses ini dilakukan untuk *scaling* data pada kolom numerik. Fungsi yang akan digunakan adalah **StandardScaler()**. Fungsi ini akan memetakan setiap nilai pada suatu distribusi menjadi nilai pada distribusi normal.



## [Modelling]

Pada tahap ini, akan dibandingkan 20+ algoritma *Machine Learning* untuk kasus klasifikasi menggunakan metode **LazyPredict**. Kemudian akan dipilih 5 algoritma yang konsisten memberikan performa yang baik terhadap data latih.  Selanjutnya, beberapa baseline model akan diberi *improvement* dengan *hyperparameter tuning* dengan tujuan agar peneliti tau parameter terbaik yang dapat digunakan untuk model.

Tabel 3. Hasil LazyPredict (Pelatihan Ke-5)

|             Model             | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken |
| :---------------------------: | -------: | ----------------: | ------: | -------: | ---------: |
|           LinearSVC           |     0.85 |              0.85 |    0.85 |     0.85 |       0.04 |
|    CalibratedClassifierCV     |     0.85 |              0.84 |    0.84 |     0.85 |       0.12 |
|     KNeighborsClassifier      |     0.85 |              0.84 |    0.84 |     0.85 |       0.09 |
|              SVC              |     0.85 |              0.84 |    0.84 |     0.84 |       0.03 |
|             NuSVC             |     0.84 |              0.84 |    0.84 |     0.84 |       0.04 |
|    RandomForestClassifier     |     0.83 |              0.83 |    0.83 |     0.83 |       0.22 |
|        RidgeClassifier        |     0.83 |              0.83 |    0.83 |     0.83 |       0.02 |
|  LinearDiscriminantAnalysis   |     0.83 |              0.83 |    0.83 |     0.83 |       0.02 |
|        LGBMClassifier         |     0.83 |              0.83 |    0.83 |     0.83 |       0.08 |
|     ExtraTreesClassifier      |     0.83 |              0.83 |    0.83 |     0.83 |       0.14 |
|      LogisticRegression       |     0.83 |              0.82 |    0.82 |     0.83 |       0.02 |
|         XGBClassifier         |     0.82 |              0.82 |    0.82 |     0.82 |       0.08 |
|       RidgeClassifierCV       |     0.83 |              0.82 |    0.82 |     0.83 |       0.02 |
|      AdaBoostClassifier       |     0.82 |              0.82 |    0.82 |     0.82 |       0.10 |
|        NearestCentroid        |     0.81 |              0.80 |    0.80 |     0.80 |       0.02 |
|          BernoulliNB          |     0.81 |              0.80 |    0.80 |     0.80 |       0.01 |
|       BaggingClassifier       |     0.80 |              0.80 |    0.80 |     0.80 |       0.03 |
|    DecisionTreeClassifier     |     0.80 |              0.80 |    0.80 |     0.80 |       0.02 |
|        LabelSpreading         |     0.79 |              0.79 |    0.79 |     0.79 |       0.04 |
|       LabelPropagation        |     0.79 |              0.79 |    0.79 |     0.79 |       0.03 |
|          Perceptron           |     0.79 |              0.79 |    0.79 |     0.79 |       0.01 |
|  PassiveAggressiveClassifier  |     0.78 |              0.78 |    0.78 |     0.78 |       0.01 |
|      ExtraTreeClassifier      |     0.76 |              0.76 |    0.76 |     0.76 |       0.01 |
|         SGDClassifier         |     0.73 |              0.73 |    0.73 |     0.73 |       0.02 |
| QuadraticDiscriminantAnalysis |     0.69 |              0.70 |    0.70 |     0.68 |       0.01 |
|          GaussianNB           |     0.50 |              0.53 |    0.53 |     0.37 |       0.01 |
|        DummyClassifier        |     0.48 |              0.50 |    0.50 |     0.31 |       0.01 |

Berikut adalah 5 model yang konsisten memberikan performa yang baik terhadap data latih.

1. **CalibratedClassifierCV**
2. **SVC**
3. **LogisticRegression**
4. **RandomForestClassifier**
5. **LinearSVC**



#### CalibratedClassifierCV

CalibratedClassifierCV menggunakan pendekatan cross-validation untuk memastikan data yang tidak bias selalu digunakan agar sesuai dengan kalibrator. Data dibagi menjadi k (train_set, test_set) pasangan (sebagaimana ditentukan oleh cv). Ketika ensemble=True (default), prosedur berikut diulangi secara independen untuk setiap pemisahan cross-validation: tiruan dari base_estimator pertama kali dilatih pada subset latih. Kemudian prediksinya pada subset uji digunakan agar sesuai dengan kalibrator (baik sigmoid ataupun isotonik regressor). Ini menghasilkan ansambel pasangan k (pengklasifikasi, kalibrator) di mana setiap kalibrator memetakan output dari pengklasifikasi yang sesuai ke [0, 1]. Setiap pasangan diekspos dalam atribut calibrated_classifiers_ , di mana setiap entri adalah classifier yang dikalibrasi dengan metode predict_proba yang menghasilkan probabilitas yang dikalibrasi. Output dari predict_proba untuk instance utama CalibratedClassifierCV sesuai dengan rata-rata probabilitas prediksi k estimator dalam daftar calibrated_classifiers_. Keluaran dari predict adalah kelas yang memiliki probabilitas tertinggi. [2]

#### SVC

SVM Classifier bekerja dengan membuat decision boundary atau sebuah bidang yang mampu memisahkan dua atau lebih kelas. Skemanya:

Pertama SVM mencari support vector pada setiap kelas. Support vector adalah sampel dari masing-masing kelas yang memiliki jarak paling dekat dengan sampel kelas lainnya. Setelah support vector ditemukan, SVM menghitung margin. Margin bisa kita anggap sebagai jalan yang memisahkan dua kelas. Margin dibuat  berdasarkan support vector di mana support vector bekerja sebagai batas tepi jalan, atau sering kita kenal sebagai bahu jalan. SVM mencari  margin terbesar atau jalan terlebar yang mampu memisahkan kedua kelas. Setelah menemukan jalan terlebar, decision boundary lalu digambar berdasarkan jalan tersebut. Decision boundary adalah garis yang membagi jalan atau margin menjadi 2  bagian yang sama besar.  [3]

Pada data non-linear, decision boundary yang dihitung algoritma SVM bukan berbentuk garis lurus. Meski cukup rumit dalam menentukan decision boundary pada kasus ini, tapi kita juga mendapatkan keuntungan, yaitu bisa menangkap lebih banyak relasi kompleks dari setiap data poin yang  tersebar. Untuk data non-linear, Support Vector Classifier menggunakan sebuah metode yaitu “kernel trick” sehingga data dapat dipisahkan secara linier. Kernel trick adalah sebuah metode untuk mengubah data pada dimensi tertentu (misal 2D) ke dalam dimensi yang lebih tinggi  (3D) sehingga dapat menghasilkan hyperplane yang optimal. [3]

Berikut adalah ilustrasi bagaimana kernel trick bekerja.

<img src='https://drive.google.com/uc?export=view&id=1_474Dk8A076KvDYYI1N-TpR8CKtqXpT5'>

Gambar 6. Ilustrasi Kernel Trick

#### LogisticRegression

Logistic regression dikenal juga sebagai logit regression, maximum-entropy classification, dan log-linear classification merupakan salah satu metode yang umum digunakan untuk klasifikasi. Pada kasus  klasifikasi, logistic regression bekerja dengan menghitung probabilitas  kelas dari sebuah sampel. [3]

Sesuai namanya, logistic regression menggunakan fungsi logistik seperti di bawah untuk menghitung probabilitas kelas dari sebuah sampel. Dalam kasus *predictive analytics* ini, apabila kelayakan calon peminjam diterima memiliki probabilitas 82%, maka calon peminjam tersebut masuk ke dalam kelas orang yang diterima pengajuan pinjamannya. Kemudian apabila kelayakan calon peminjam memiliki probabilitas <50%, maka calon peminjam tersebut masuk ke dalam kelas orang yang ditolak pengajuan pinjamannya.

Berikut adalah ilustrasi dari fungsi logistik.

<img src='https://drive.google.com/uc?export=view&id=1cASbNnfz15OrYaqZat0AqnzQyKcBMzhh'>

Gambar 7. Grafik Fungsi Logistik

#### RandomForestClassifier

Random Forest adalah perluasan dari metode bagging karena menggunakan bagging dan fitur acak untuk membuat Decision Tree yang tidak berkorelasi. Keacakan fitur, juga dikenal sebagai *feature bagging*, menghasilkan subset fitur acak, yang memastikan korelasi rendah di antara Decision Tree. Ini adalah perbedaan utama antara Decision Tree dan Random Forest. Dengan kata lain, Decision Tree mempertimbangkan semua pemisahan fitur yang mungkin, sedangkan Random Forest hanya memilih subset dari fitur tersebut. [4]

Berikut adalah ilustrasi dari Random Forest.

<img src='https://drive.google.com/uc?export=view&id=1AnszscxbG4HeiabmWPqh-HHK0No_4VGI'>

Gambar 8. Ilustrasi Random Forest



Kelebihan Random Forest:

- Menghasilkan eror yang lebih rendah.
- Memberikan hasil yang bagus dalam klasifikasi.
- Dapat mengatasi data training dalam jumlah sangat besar secara efisien.
- Dapat memperkiraan variabel apa yang penting dalam klasifikasi.
- Menyediakan metode eksperimental untuk mendeteksi interaksi variabel.

Kekurangan Random Forest:

- Waktu pemrosesan yang lama karena menggunakan data yang banyak dan  membangun model tree yang banyak pula untuk membentuk random trees  karena menggunakan single processor.
- Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
- Ketika digunakan untuk regresi, mereka tidak dapat memprediksi di luar  kisaran dalam data percobaan, hal ini di mungkinkan data terlalu cocok  dengan kumpulan data pengganggu (noisy).

#### LinearSVC

Tujuan LinearSVC (Support Vector Classifier) adalah untuk melatih data yang diberikan, kemudian mengembalikan hyperplane "paling cocok" untuk membagi atau mengkategorikan data tersebut. Secara konsep dasar, LinearSVC memiliki kesamaan dengan SVC. Adapun perbedaan di antara keduanya antara lain LinearSVC menggunakan 'garis lurus' untuk pengklasifikasian linier, sedangkan SVC memungkinkan kita memilih berbagai kernel non-linier.



Berikut adalah hasil akurasi dari pelatihan 5 model di atas.

Tabel 4. Akurasi Train dan Tes Model

|         Model          | Akurasi Train | Akurasi Tes |
| :--------------------: | ------------: | ----------: |
| CalibratedClassifierCV |          0.84 |        0.83 |
|          SVC           |          0.97 |        0.87 |
|   LogisticRegression   |          0.84 |        0.83 |
| RandomForestClassifier |          0.96 |        0.83 |
|       LinearSVC        |          0.84 |        0.85 |



## [Evaluation]

Model yang digunakan adalah kasus *predictive analytics* ini merupakan model berjenis klasifikasi. Oleh karena itu, metrics evaluasi yang akan digunakan adalah ***Confusion Matrix*** dan ***Classification Report***.

#### Confusion Matrix

Confusion Matrix merupakan metode evaluasi yang dapat digunakan untuk menghitung kinerja atau tingkat kebenaran dari proses klasifikasi. Confusion Matrix adalah tabel dengan 4 kombinasi berbeda dari nilai prediksi dan nilai aktual.

Ada empat istilah yang merupakan representasi hasil proses klasifikasi pada Confusion Matrix yaitu True Positive (TP), True Negative (TN), False  Positive (FP), dan False Negative (FN).

Berikut adalah tabel Confusion Matrix.

<img src='https://drive.google.com/uc?export=view&id=1p3VITyFH8DQRReQVvoKtPQVh1q4cZpFu'>

Gambar 9. Confusion Matrix untuk Target dengan 2 Kategori

Keterangan :

- TP (True Positive) terjadi apabila peminjam diprediksi model sebagai kategori yang layak diberi pinjaman (Positive), dan kondisi aktual peminjam memang layak diberi pinjaman (Positive).
- FN (False Negative) terjadi apabila peminjam diprediksi model sebagai kategori yang tidak layak diberi pinjaman (Negative), akan tetapi kondisi aktual peminjam layak diberi pinjaman (Positive).
- FP (False Positive) terjadi apabila peminjam diprediksi model sebagai kategori yang layak diberi pinjaman (Positive), akan tetapi kondisi aktual peminjam tidak layak diberi pinjaman (Negative).
- TN (True Negative) terjadi apabila peminjam diprediksi model sebagai kategori yang tidak layak diberi pinjaman (Negative), dan kondisi aktual peminjam memang tidak layak diberi pinjaman (Negative).

Berikut adalah Confusion Matrix dari 5 model yang telah dilatih.

```
Confusion Matrix Model CalibratedClassifierCV:
 [[ 77  28]
 [  9 107]]

Confusion Matrix Model SVC:
 [[ 84  21]
 [  8 108]]

Confusion Matrix Model LogisticRegression:
 [[ 77  28]
 [  9 107]]

Confusion Matrix Model RandomForestClassifier:
 [[87 18]
 [19 97]]

Confusion Matrix Model LinearSVC:
 [[ 77  28]
 [  6 110]]
```

#### **Classification Report**

Berdasarkan Confusion Matrix, berikut adalah beberapa metrics yang dapat diukur.

- **Akurasi**

<img src="https://latex.codecogs.com/gif.latex?Accuracy&space;=&space;\frac{TP&plus;TN}{TP&plus;FP&plus;FN&plus;TN}" title="Accuracy = \frac{TP+TN}{TP+FP+FN+TN}" />

- **Presisi**

<img src="https://latex.codecogs.com/gif.latex?Precision&space;=&space;\frac{TP}{TP&plus;FP}" title="Precision = \frac{TP}{TP+FP}" />

- **Recall**

<img src="https://latex.codecogs.com/gif.latex?Recall&space;=&space;\frac{TP}{TP&plus;FN}" title="Recall = \frac{TP}{TP+FN}" />

- **F1 Score**

<img src="https://latex.codecogs.com/gif.latex?F1&space;Score&space;=&space;\frac{2\times(Recall&space;\times&space;Precission)}{Recall&space;&plus;&space;Precission}" title="F1 Score = \frac{2\times(Recall \times Precission)}{Recall + Precission}" />

Berikut adalah Classification Report dari 5 model yang telah dilatih.

```
Classification Report Model CalibratedClassifierCV:
               precision    recall  f1-score   support

           0       0.90      0.73      0.81       105
           1       0.79      0.92      0.85       116

    accuracy                           0.83       221
   macro avg       0.84      0.83      0.83       221
weighted avg       0.84      0.83      0.83       221


Classification Report Model SVC:
               precision    recall  f1-score   support

           0       0.91      0.80      0.85       105
           1       0.84      0.93      0.88       116

    accuracy                           0.87       221
   macro avg       0.88      0.87      0.87       221
weighted avg       0.87      0.87      0.87       221


Classification Report Model LogisticRegression:
               precision    recall  f1-score   support

           0       0.90      0.73      0.81       105
           1       0.79      0.92      0.85       116

    accuracy                           0.83       221
   macro avg       0.84      0.83      0.83       221
weighted avg       0.84      0.83      0.83       221


Classification Report Model RandomForestClassifier:
               precision    recall  f1-score   support

           0       0.82      0.83      0.82       105
           1       0.84      0.84      0.84       116

    accuracy                           0.83       221
   macro avg       0.83      0.83      0.83       221
weighted avg       0.83      0.83      0.83       221


Classification Report Model LinearSVC:
               precision    recall  f1-score   support

           0       0.93      0.73      0.82       105
           1       0.80      0.95      0.87       116

    accuracy                           0.85       221
   macro avg       0.86      0.84      0.84       221
weighted avg       0.86      0.85      0.84       221
```

Dalam kasus *predictive analytics* ini, selain model yang dibangun diharapkan mampu memberikan prediksi klasifikasi kelayakan peminjam secara True, akan lebih diterima apabila letak error model lebih banyak berada pada False Negative (FN) dibanding dengan False Positive (FP). Hal ini dikarenakan apabila kita melihat dari sisi bisnis, lebih baik menolak calon peminjam yang benar-benar layak (peneliti dapat melakukan improvement kembali pada model), dibandingkan memberikan pinjaman kepada calon peminjam yang sebetulnya tidak layak diberi pinjaman.



Kesimpulan yang diperoleh dari hasil analisis dan pemodelan *machine learning* untuk kasus ini yakni model yang digunakan untuk memprediksi kelayakan calon peminjam pada perusahaan *Dream Housing Finance* adalah model dengan algoritma **RandomForestClassifier**. Model ini memiliki performa ketepatan klasifikasi yang cukup baik. Namun meskipun demikian, masih perlu adanya *improvement* lebih lanjut untuk meminimalisasi kesalahan klasifikasi untuk tipe FP (False Positive).



## **[Daftar Referensi]**

[1] Huda. B, et al., "Pengaruh Kualitas Pelayanan, Prosedur Kredit, dan Tingkat Suku Bunga terhadap Keputusan Nasabah Dalam Mengambil Kredit pada PT. Bank Perkreditan Rakyat Sukowono Arthajaya Jember". Jurnal Ilmiah Ilmu Pendidikan, Ilmu Ekonomi, dan Ilmu Sosial, Vol 13, No.1. 87-93, 2019.

[2] Scikit-Learn. "Probability Calibration". https://scikit-learn.org/stable/modules/calibration.html [accesed Nov. 30 2022]

[3] Dicoding. "Belajar Machine Learning untuk Pemula". https://www.dicoding.com/academies/184/tutorials/8447 [accesed Nov. 30 2022]

[4] IBM. "Random Forest". https://www.ibm.com/cloud/learn/random-forest [accesed Nov. 30 2022]

