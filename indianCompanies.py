import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
df = pd.read_csv('indian companies complete data 2025.csv')

# İlk 5 satırı gösterme (Verinin ne olduğunu anlamak için)
print("İlk 5 Satır:")
print(df.head())

# Veri seti hakkında genel bilgi alma
print("\nVeri Seti Bilgisi:")
print(df.info())
# Eksik verilerin toplam sayısını kontrol etme
print("\nEksik Veri Sayıları:")
print(df.isnull().sum())

# 1. Kopya oluşturma 
df_clean = df.copy()
# 2. 'RATING' sütununu temizleme
# Sayısal olmayan değerleri NaN yapar, sonra temizleriz
df_clean['RATING'] = pd.to_numeric(df_clean['RATING'], errors='coerce')

# 3. 'YEAR OLD' sütununu temizleme 
# Regex ile sadece sayıları alıyoruz
df_clean['YEAR OLD'] =df_clean['YEAR OLD'].str.extract('(\d+)').astype(float)

# 4. 'REVIEWS'sütununu temizleme (k = 1000, L = 100000 dönüşümü)
def clean_reviews(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace('(', '').replace(')', '')
    if 'L' in x:
        return float(x.replace('L', '')) * 100000
    elif 'k' in x:
        return float( x.replace('k', '')) * 1000
    else:
        try:
            return float(x)
        except:
            return np.nan
df_clean['REVIEWS'] = df_clean['REVIEWS'].apply(clean_reviews)

# 5.'PACKAGE' sütununu temizleme (9.4L -> 940000 gibi)
# Sadece sayısal kısmı alıp 'L' varsayımıyla çarparız
def clean_package(x):
    if  pd.isna(x):
        return np.nan
    x = str(x)
    if 'L' in x:
        return float(x.replace('L', '')) * 100000
    else:
        try:
            return float(x) # Eğer L yoksa direkt sayı olabilir
        except:
            return np.nan

df_clean['PACKAGE'] = df_clean['PACKAGE'].apply(clean_package)

# 6. Gereksiz veya çok eksik sütunları atma
# CONS ve COMPANY TYPE çok eksikti, CONS'u atalım çünkü metin analizi zor olabilir
# COMPANY TYPE için 'Unknown' dolduralım.
df_clean = df_clean.drop(columns=['CONS'])
df_clean['COMPANY TYPE'] = df_clean['COMPANY TYPE'].fillna('Unknown')

# Kalan eksik verileri doldurma stratejileri:
# Sayısal sütunlar için ortalama (mean) veya medyan
df_clean['RATING'] = df_clean['RATING'].fillna(df_clean['RATING'].mean())
df_clean['REVIEWS'] = df_clean['REVIEWS'].fillna(df_clean['REVIEWS'].median())
df_clean['PACKAGE'] = df_clean['PACKAGE'].fillna(df_clean['PACKAGE'].median())
df_clean['YEAR OLD'] = df_clean['YEAR OLD'].fillna(df_clean['YEAR OLD'].median())

# Kategorik sütunlar için "Unknown" veya en sık gecen değer (Mode)
df_clean['INDIA HQ'] = df_clean['INDIA HQ'].fillna('Unknown')
df_clean['BENEFITS'] = df_clean['BENEFITS'].fillna('Not Specified')
df_clean['TOTAL EMPLOYEES'] = df_clean['TOTAL EMPLOYEES'].fillna('Unknown')
df_clean['BRANCHES'] = df_clean['BRANCHES'].fillna('Unknown')

# Son durum görelim
print("Temizlenmiş Veri Bilgisi:")
print(df_clean.info())
print("\nTemizlenmiş İlk 5 Satır:")
print(df_clean.head())

# Grafik stili ayarlama
sns.set_style("whitegrid")
plt.figure(figsize=(20, 15))

# 1. Grafik: Şirket Puanlarının (Rating) Dağılımı
plt.subplot(2, 2, 1)
sns.histplot(df_clean['RATING'], kde=True, color='skyblue', bins=20)
plt.title('Şirket Puanlarının Dağılımı', fontsize=15)
plt.xlabel('Puan (Rating)')
plt.ylabel('Frekans')

# 2. Grafik: En Yaygın 10 Endüstri
plt.subplot(2, 2, 2)
top_industries = df_clean['INDUSTRY'].value_counts().head(10)
sns.barplot(y=top_industries.index, x=top_industries.values, palette='viridis')
plt.title('En Yaygın 10 Endüstri', fontsize=15)
plt.xlabel('Şirket Sayısı')

# 3. Grafik: Maaş (Package) ve Puan (Rating) İlişkisi
# Verinin daha okunabilir olması için çok yüksek maaşları (outliers) filtreleyebiliriz görselde
plt.subplot(2, 2, 3)
sns.scatterplot(data=df_clean, x='PACKAGE', y='RATING', alpha=0.5, color='coral')
plt.title('Maaş vs Şirket Puanı', fontsize=15)
plt.xlabel('Maaş (Package - Yıllık)')
plt.ylabel('Puan (Rating)')
plt.xscale('log') # Maaşlar çok değişken olduğu için logaritmik ölçek kullanıyoruz

# 4. Grafik: Korelasyon Isı Haritası (Heatmap)
plt.subplot(2, 2, 4)
# Sadece sayısal sütunları seçiyoruz
numeric_cols = df_clean[['RATING', 'REVIEWS', 'PACKAGE', 'YEAR OLD']]
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Değişkenler Arası Korelasyon', fontsize=15)
plt.tight_layout()
plt.savefig('visualization_output.png') # Dosyayı kaydet
print("Görseller oluşturuldu.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Özellikleri (X) ve Hedefi (y) Belirleme
# RATING'i tahmin etmeye çalışacağız.
# Kullanacağımız özellikler: REVIEWS (Popülerlik), PACKAGE (Maaş), YEAR OLD (Yaş)
X = df_clean[['REVIEWS', 'PACKAGE', 'YEAR OLD']]
y = df_clean['RATING']

# 2. Veriyi Eğitim ve Test Olarak Bölme (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeli Seçme ve Eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Tahmin Yapma
y_pred = model.predict(X_test)

# 5. Modeli Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performansı:")
print(f"Ortalama Kare Hata (MSE): {mse:.4f}")
print(f"R-kare (R2 Score): {r2:.4f}")
# Katsayıları İnceleme (Hangi özellik ne kadar etkili ?)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Katsayı (Etki)'])
print("\nÖzelliklerin Etkisi:")
print(coefficients)

# Örnek bir tahmin yapalım:
# 1000 yorumu olan, 500.000 maaş veren ve 10 yıllık bir şirket
new_company = pd.DataFrame([[1000, 500000, 10]], columns=['REVIEWS', 'PACKAGE', 'YEAR OLD'])
predicted_rating = model.predict(new_company)
print(f"\nÖrnek Şirket Tahmini Puanı: {predicted_rating[0]:.2f}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

print("GELİŞMİŞ MODELLEME AŞAMASI")
# Özellikler ve Hedef
X = df_clean[['REVIEWS', 'PACKAGE', 'YEAR OLD']]
y = df_clean['RATING']

# Eğitim ve Test Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ADIM 1: Farklı bir regresyon modeli (Random Forest)
print("\n1. Model: Random Forest Regressor (Sayısal Tahmin)")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"Random Forest R2 Score: {r2_rf:.4f}")
print("Yorum: R2 skoru hala düşükse, veriler arasında sayısal bir ilişki kurmak zordur.")

# Puanı tam tahmin etmek yerine, şirketin "İyi" mi "Orta/Kötü" mü olduğunu tahmin edelim.
# Eşik değerimiz 3.8  olsun 

print("\n 2. Model: Random Forest Classifier (Sınıflandırma)")
print("Strateji: Puanı 3.8 ve üzeri olanlara '1' (İyi), altı olanlara '0' (Diğer) diyelim.")

# Hedef değişkeni kategoriye cevirme
y_class = (df_clean['RATING'] >= 3.8).astype(int)

# Yeni eğitim/test setleri (X aynı kalıyor, y değişti)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Sınıflandırma Modeli
rf_class = RandomForestClassifier(n_estimators=100 , random_state=42)
rf_class.fit(X_train_c, y_train_c)

y_pred_class = rf_class.predict(X_test_c)

# Basarı Metrikleri
acc = accuracy_score(y_test_c, y_pred_class)
print(f"\nSınıflandırma Doğruluğu (Accuracy): %{acc * 100:.2f}")

# Confusion Matrix (Hata Matrisi) Görselleştirmesi
# Bu grafik, modelin ne kadar dogru ne kadar yanlış oldugunu gösterir
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test_c, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Diğer', 'İyi Şirket'], yticklabels=['Diğer', 'İyi Şirket'])
plt.title('Sınıflandırma Modeli Performansı (Confusion Matrix)')
plt.xlabel('Tahmin Edilen ')
plt.ylabel('Gercek Durum ')
plt.savefig('model_basarisi.png')

# Özelliklerin Önem Düzeyi
# Hangi özellik şirketin iyi olup olmadığını belirlemede daha etkili diye bakıyoruz
importances = pd.Series(rf_class.feature_importances_, index=X.columns)
print("Şirket Puanını Etkileyen En Önemli Faktörler: ")
print(importances.sort_values(ascending=False))

plt.figure(figsize=(8, 4))
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Hangi Özellik Daha Önemli?')
plt.xlabel('Önem Derecesi')
plt.savefig('feature_importance.png')