import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# KDD99 veri setini yükleme (yolunuzu güncellemeniz gerekebilir)
df = pd.read_csv("kddcup99_csv.csv")

# === Normalizasyon İşlemleri ===

# 1. Boyut Azaltma ('Packets not Found' alanı oluşturulması)
# Sütun isimlerini kontrol et
print("Mevcut sütunlar:", df.columns)

# packets_looked_up ve packets_matched sütunlarının varlığını kontrol et
if 'packets_looked_up' in df.columns and 'packets_matched' in df.columns:
    df['packets_not_found'] = df['packets_looked_up'] - df['packets_matched']
else:
    print("Gerekli sütunlar (packets_looked_up veya packets_matched) mevcut değil.")


# 2. Az Örneğe Sahip Sınıfı Çıkarma
if 'label' in df.columns:
    df = df[df['label'] != 'Overflow']  # Sınıf 5 (Overflow) çıkarılıyor

# 3. Gereksiz Özelliklerin Çıkarılması
drop_columns = ['table_id', 'max_size', 'is_valid']
drop_columns = [col for col in drop_columns if col in df.columns]  # Var olan sütunları filtrele
df.drop(columns=drop_columns, inplace=True, errors='ignore')

# 4. Değeri Sabit Olan Özelliklerin Çıkarılması
df = df.loc[:, df.nunique() > 1]  # Sadece farklı değerlere sahip sütunları koru

# 5. Ordinal Encoding (Kategorik Veriler İçin)
categorical_columns = df.select_dtypes(include=["object"]).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# === Hedef Değişken ve Özelliklerin Ayrılması ===
y = df["label"]
x = df.drop(columns=["label"])

# 6. Ölçekleme (StandardScaler)
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Eğitim ve test veri setlerindeki sınıf dağılımını kontrol etme
print("Eğitim setindeki sınıf dağılımı:")
print(y_train.value_counts())
print("\nTest setindeki sınıf dağılımı:")
print(y_test.value_counts())

# === Model Tanımlama ===
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True),  # ROC AUC için probability=True kullanıyoruz
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Sonuçları saklamak için bir sözlük
results = {}

# === Model Eğitimi ve Değerlendirme ===
for model_name, model in models.items():
    # Modeli eğitme
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Gerçek ve tahmin edilen sınıfların kontrolü
    print(f"\n{model_name} Modeli İçin:")
    print(f"Gerçek sınıflar: {set(y_test)}")
    print(f"Tahmin edilen sınıflar: {set(y_pred)}")

    # Skor hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # ROC AUC hesaplama (ROC AUC yalnızca binary veya uygun veri tiplerinde hesaplanabilir)
    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class='ovr')
    except:
        roc_auc = "N/A"

    # Sonuçları saklama
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Balanced Accuracy": balanced_acc,
        "MCC": mcc,
        "ROC AUC": roc_auc
    }

# === Sonuçları Yazdırma ===
for model_name, metrics in results.items():
    print(f"\n{model_name} Modeli Sonuçları:")
    print(f"Accuracy: {metrics['Accuracy']*100:.5f}%")
    print(f"Precision: {metrics['Precision']*100:.5f}%")
    print(f"Recall: {metrics['Recall']*100:.5f}%")
    print(f"F1 Score: {metrics['F1 Score']*100:.5f}%")
    print(f"Balanced Accuracy: {metrics['Balanced Accuracy']*100:.5f}%")
    print(f"MCC: {metrics['MCC']:.5f}")
    print(f"ROC AUC: {metrics['ROC AUC']}")

# === Karışıklık Matrisi ===
ConfusionMatrixDisplay.from_predictions(y_test, models["Random Forest"].predict(x_test), xticks_rotation="vertical")
plt.title("Random Forest Confusion Matrix")
plt.show()
