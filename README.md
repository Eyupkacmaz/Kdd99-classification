# KDD99 Veri Seti ile Makine Öğrenimi Model Karşılaştırması

Bu proje, KDD99 veri setini kullanarak çeşitli makine öğrenimi modellerinin ağ saldırılarını tespit etme performanslarını karşılaştırmayı amaçlamaktadır. Proje, veri ön işleme adımlarından model eğitimi ve değerlendirmesine kadar kapsamlı bir süreç içermektedir.

---

## 📚 İçindekiler
1. [Proje Açıklaması](#proje-açıklaması)
2. [Veri Seti](#veri-seti)
3. [Kurulum](#kurulum)
4. [Kullanım](#kullanım)
5. [Çıktılar](#çıktılar)
6. [Lisans](#lisans)

---

## 📌 Proje Açıklaması

Bu projede, KDD99 veri seti üzerinde aşağıdaki makine öğrenimi modelleri eğitilmiş ve değerlendirilmiştir:
- **K-Nearest Neighbors**
- **Decision Tree**
- **Random Forest**
- **Naive Bayes**
- **Support Vector Machine**
- **XGBoost**

### Özellikler:
- Veri ön işleme:
  - Eksik ve gereksiz sütunların kaldırılması
  - Kategorik değişkenlerin encode edilmesi
  - Özelliklerin ölçeklenmesi
- Performans değerlendirme:
  - Accuracy, Precision, Recall, F1 Score gibi metrikler
  - Karışıklık matrisi görselleştirme
- Eğitim ve test veri setlerinin sınıf dağılımının kontrolü

---

## 📂 Veri Seti

KDD99 veri seti, ağ saldırılarını ve anormal aktiviteleri içeren geniş bir veri kümesidir. Bu veri seti, makine öğrenimi algoritmalarının performansını değerlendirmek için kullanılmıştır. 

**Tam veri seti boyutu nedeniyle GitHub'a yüklenememiştir. Ancak aşağıdaki bağlantıdan erişebilirsiniz:**
- [KDD99 Veri Seti ](https://www.kaggle.com/datasets/toobajamal/kdd99-dataset)

---

## 🔧 Kurulum

Projeyi çalıştırmak için aşağıdaki bağımlılıkları kurmanız gerekmektedir. Python 3.8 veya üzeri bir sürüm gereklidir.

### Gerekli Kütüphaneler:
```bash
pip install pandas scikit-learn xgboost matplotlib
```
 ## 🚀 Kullanım

Proje dosyasını çalıştırmak için şu adımları takip edin:

```bash kdd99_classification.py ```  dosyasını indirin.

Çalıştırmak için aşağıdaki komutu kullanın:
```bash
python kdd99_classification.py
```
Çalıştırma sonunda, aşağıdaki sonuçlar üretilecektir:

- Model metrikleri

- Karışıklık matrisi

- Eğitim ve test veri setinin sınıf dağılımı
