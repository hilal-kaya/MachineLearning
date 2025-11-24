# 🇮🇳 Hint Şirketleri Veri Analizi ve ML Projesi

Bu projede 10.000 satırlık Hint şirketleri veri setini inceledim. Amacım şirketlerin yaşı, verdiği maaş ve yorum sayılarına bakarak o şirketlerin çalışanlar tarafından ne kadar
sevildiğini tahmin eden bir model geliştirmekti.

Neler yaptım:

Veri seti oldukça karmaşıktı. Özellikle maaş ve yorum sayılarında '1.1L' veya '67.9k' gibi metinler vardı.
Önce pandas ile veri temizliği yaptım.  
Aşağıdaki fonksiyon da veri temizliğinden önemli bir kesit.

def clean_reviews(x):
    if 'L' in x: return float(x.replace('L', '')) * 100000  # Lakh dönüşümü
    elif 'k' in x: return float(x.replace('k', '')) * 1000  # Bin dönüşümü
    return float(x)

Temizlenen veriyi Seaborn ve Matplotlib kullanarak görselleştirdim. Maaş ve puan arasında linear bir ilişki var mı diye baktım. Sonuçlar biraz şaşırtıcıydı çünkü çok maaş veren 
şirketlerin hep çok yüksek puanı yoktu. 

İki farklı stratejiyle makine öğrenmesi denemesi yaptım ama ilki olan linear regression ile hiçbir sonuç alamadım. Bunun üzerine Random Forest Classifier yöntemini kullandım:

y_class = (df_clean['RATING'] >= 3.8).astype(int) 
rf_class = RandomForestClassifier(n_estimators=100)
rf_class.fit(X_train, y_train)

Bu yöntemle iyi şirketlere (puanı 3.8 ve üzeri) -> 1
diğer şirketlere -> 0 dedim. 

Bu proje bana özellikle gerçek hayat verilerinin her zaman linear olmadığını doğru soruyu sormanın sormanın model başarısını nasıl değiştirdiğini öğretti. 

İncelediğiniz için teşekkür ederim :)
