# 🔬 HIGGS Dataset — Machine Learning Pipeline

> Parçacık fiziği sinyal sınıflandırması için uçtan uca makine öğrenmesi boru hattı  
> **Üsküdar Üniversitesi | Yapay Zeka ve Uygulamaları**

---

## 📋 Proje Özeti

Bu proje, [HIGGS veri seti](https://archive.ics.uci.edu/ml/datasets/HIGGS) üzerinde kapsamlı bir makine öğrenmesi pipeline'ı uygulamaktadır. Orijinal 11 milyon örnekten **100.000 rastgele örnek** seçilerek dört farklı algoritma, iç içe çapraz doğrulama (Nested Cross-Validation) mimarisiyle karşılaştırılmıştır.

---

## 🗂️ Proje Yapısı

```
├── lastmlfinal.ipynb       # Ana Jupyter Notebook (tüm pipeline)
├── HIGGS.csv.gz            # Veri seti (otomatik indirilir)
├── features_target_boxplots.png   # Box plot grafikleri
├── models_roc_curves.png          # ROC eğrisi karşılaştırması
└── README.md
```

---

## ⚙️ Pipeline Adımları

```
Ham Veri (11M örnek)
       ↓
100.000 Rastgele Örnekleme
       ↓
Eksik Veri Kontrolü → 0 eksik veri
       ↓
Aykırı Değer Analizi (IQR) → Winsorization
       ↓
Feature Scaling → MinMaxScaler [0, 1]
       ↓
Korelasyon Analizi (Pearson Heatmap)
       ↓
Feature Selection → ANOVA F-score (28 → 15 özellik)
       ↓
Nested Cross-Validation (Outer: 5-fold / Inner: 3-fold)
       ↓
GridSearchCV → En İyi Hiperparametreler
       ↓
Model Değerlendirme (Accuracy, F1, ROC-AUC)
```

---

## 🤖 Modeller ve Hiperparametreler

| Model | Hiperparametreler |
|-------|-------------------|
| **XGBoost** | `max_depth`: [3, 5] · `learning_rate`: [0.1, 0.2] · `n_estimators`: 50 |
| **MLP** | `hidden_layer_sizes`: [(50,), (100,)] · `activation`: [relu, tanh] · `early_stopping`: True |
| **KNN** | `n_neighbors`: [3, 7, 11] |
| **SVM** | `LinearSVC` · `C`: [0.1, 1, 10] |

> ⚠️ **SVM Notu:** RBF kernel (`SVC`) 100.000 örnek üzerinde O(n²)–O(n³) karmaşıklığı nedeniyle Google Colab ortamında sistemi kilitlemektedir. Bu yüzden `LinearSVC` kullanılmıştır. Bu bir algoritmik tercih değil, hesaplama ortamının getirdiği bir kısıttır.

---

## 📊 Sonuçlar

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 0.7154 | 0.7298 | 0.7375 | 0.7336 | **0.7918** |
| MLP | 0.7135 | 0.7337 | 0.7241 | 0.7286 | 0.7828 |
| KNN | 0.6695 | 0.6737 | 0.7330 | 0.7021 | 0.7260 |
| SVM (Linear) | 0.6431 | 0.6429 | 0.7384 | 0.6874 | 0.6808 |

### ROC Eğrileri

![ROC Curves](models_roc_curves.png)

---

## 🏆 En Başarılı Kombinasyon

```
MinMaxScaler + Winsorization + ANOVA F-score (15 özellik) + XGBoost
ROC-AUC: 0.7918
```

**Neden XGBoost?**  
HIGGS gibi yapılandırılmış (tabular) veri kümelerinde gradyan artırımlı karar ağaçları, öznitelikler arasındaki **doğrusal olmayan karmaşık ilişkileri** yakalamada üstündür. ANOVA ile boyut indirgeme ise gereksiz korelasyonları temizleyerek XGBoost'un genelleme yeteneğini artırmıştır.

---

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Çalıştırma

```bash
# Jupyter Notebook
jupyter notebook lastmlfinal.ipynb
```

> 📌 Veri seti ilk çalıştırmada otomatik olarak UCI deposundan indirilir (~2.6 GB).  
> Google Colab'da çalıştırılması önerilir (RAM yoğun işlemler nedeniyle).

---

## 📦 Kullanılan Kütüphaneler

| Kütüphane | Kullanım |
|-----------|----------|
| `numpy` / `pandas` | Veri işleme |
| `matplotlib` / `seaborn` | Görselleştirme |
| `scikit-learn` | Ön işleme, CV, metrikler |
| `xgboost` | XGBoost sınıflandırıcı |

---

## 📁 Veri Seti

- **Kaynak:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)
- **Toplam Örnek:** ~11 milyon
- **Kullanılan Örnek:** 100.000 (rastgele)
- **Özellik Sayısı:** 28 → 15 (ANOVA seçimi sonrası)
- **Hedef:** Binary (0: Arka plan, 1: Higgs sinyali)
