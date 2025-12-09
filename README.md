# 🇮🇳 Hint Şirketleri Veri Analizi ve ML Projesi

Bu projede 10.000 satırlık Hint şirketleri veri setini inceledim. Amacım şirketlerin yaşı, verdiği maaş ve yorum sayılarına bakarak o şirketlerin çalışanlar tarafından ne kadar
sevildiğini tahmin eden bir model geliştirmekti.

# Veri Ön İşleme 
Veri seti oldukça karmaşıktı. Özellikle maaş ve yorum sayılarında '1.1L' veya '67.9k' gibi metinler vardı.
Önce pandas ile veri temizliği yaptım.  
Aşağıdaki fonksiyon da veri temizliğinden önemli bir kesit.

// 'REVIEWS'sütununu temizleme (k = 1000, L = 100000 dönüşümü)
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

Temizlenen veriyi Seaborn ve Matplotlib kullanarak görselleştirdim. Maaş ve puan arasında linear bir ilişki var mı diye baktım. Sonuçlar biraz şaşırtıcıydı çünkü çok maaş veren 
şirketlerin hep çok yüksek puanı yoktu. Hangi feature daha çok etkiledi merak ettim ve sonuç aşağıda:
blob:https://colab.research.google.com/83c8cdf8-a7aa-4c64-b012-57abf7f310b6<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/03bce96b-46a3-4490-8042-5a44a3543a7e" />



# Model Performansı
Projenin ilk aşamasında basit bir `Linear Regression` modeli kurdum ancak R² skoru (başarı oranı) çok düşüktü. 'Acaba veriler arasında doğrusal olmayan (non-linear) karmaşık ilişkiler mi var?' sorusundan yola çıkarak daha gelişmiş ağaç tabanlı modelleri denemeye karar verdim. Bunun üzerine Random Forest Classifier yöntemini kullandım:

// Hedef değişkeni kategoriye cevirme
y_class = (df_clean['RATING'] >= 3.8).astype(int)

// Yeni eğitim/test setleri (X aynı kalıyor, y değişti)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

// Sınıflandırma Modeli
rf_class = RandomForestClassifier(n_estimators=100 , random_state=42)
rf_class.fit(X_train_c, y_train_c)

Bu yöntemle iyi şirketlere (puanı 3.8 ve üzeri) -> 1
diğer şirketlere -> 0 dedim. 

Aşağıdaki kod bloğunda Linear Regression,Decision Tree ve Random Forest modellerini aynı veri seti üzerinde yarıştırdım:

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10)
}

Modelleri döngü ile eğitip test ettim
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} R2 Score: {score}")

blob:https://colab.research.google.com/2eb082a4-4e35-41b5-bc89-37b9a39f136a<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/f826b87f-2538-447f-bc76-cf37ae810a07" />

<img width="1416" height="172" alt="image" src="https://github.com/user-attachments/assets/780e5b80-fc20-4b92-b67c-beb7873dd2b7" />

# Sonuç
Random Forest modelinin performansı digerlerine göre 10 kat daha iyi olsa da genel skor hala düşük seviyede.
Bu durum veri bilimi açısından çok önemli bir gerçeği ortaya koyuyor:

Bir şirketin çalışan memnuniyeti (Rating); sadece Maaş, Şirket Yaşı veya Tanınırlık gibi sayısal verilerle tam olarak tahmin edilemez.
Bu proje bana özellikle gerçek hayat verilerinin her zaman linear olmadığını doğru soruyu sormanın sormanın model başarısını nasıl değiştirdiğini de öğretti. 
blob:https://colab.research.google.com/6f001733-21ec-4491-acff-1175fb5a7bff<img width="2000" height="1500" alt="image" src="https://github.com/user-attachments/assets/8c361a99-99a3-43c4-9649-a9da7f883040" />


İncelediğiniz için teşekkür ederim :)
