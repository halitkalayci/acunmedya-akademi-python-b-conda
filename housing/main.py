# Indexing => Proje dosyalarınızın Agent tarafından okunması.

# `` -> Back Tick

# Prompt Engineering

# Türkiye Ev Fiyat Tahmini - Random Forest Modeli
# Bu proje Türkiye'deki ev fiyatlarını tahmin etmek için Random Forest algoritmasını kullanır

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']

print("🏠 Türkiye Ev Fiyat Tahmini - Random Forest Modeli")
print("=" * 60)

# 1. VERİ YÜKLEMESİ VE KEŞF ANALİZİ
print("\n📊 1. Veri Yükleniyor...")
df = pd.read_csv('turkiye_ev_fiyatlari.csv')

print(f"✅ Veri seti başarıyla yüklendi!")
print(f"📋 Veri şekli: {df.shape}")
print(f"📈 Özellik sayısı: {df.shape[1] - 1}")
print(f"🎯 Hedef değişken: fiyat_tl")

print(f"\n📊 Veri Seti Genel Bilgileri:")
print(df.info())

print(f"\n🔍 Eksik Veri Kontrolü:")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✅ Eksik veri bulunmamaktadır!")
else:
    print(missing_data[missing_data > 0])

print(f"\n📈 Hedef Değişken (Fiyat) İstatistikleri:")
print(df['fiyat_tl'].describe())

# 2. VERİ ÖN İŞLEME
print(f"\n🔧 2. Veri Ön İşleme...")

# Kategorik değişkenleri belirleme
categorical_columns = ['sehir', 'ilce', 'ev_tipi', 'isinma_turu']
numerical_columns = ['metrekare', 'oda_sayisi', 'salon_sayisi', 'banyo_sayisi', 
                    'bina_yasi', 'bina_kat_sayisi', 'bulundugu_kat']
boolean_columns = ['balkon', 'asansor', 'park_yeri', 'site_icinde', 'esyali']

print(f"📊 Kategorik özellikler: {len(categorical_columns)}")
print(f"🔢 Sayısal özellikler: {len(numerical_columns)}")
print(f"✅ Boolean özellikler: {len(boolean_columns)}")

# Kategorik verileri encode etme
label_encoders = {}
df_processed = df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"✅ {col} encode edildi ({len(le.classes_)} sınıf)")

# Boolean verileri 0/1'e çevirme
for col in boolean_columns:
    df_processed[col] = df_processed[col].astype(int)

# 3. ÖZELLİK VE HEDEF AYIRMA
print(f"\n🎯 3. Özellik ve Hedef Değişken Ayrımı...")

X = df_processed.drop('fiyat_tl', axis=1)
y = df_processed['fiyat_tl']

print(f"📊 Özellik matrisi şekli: {X.shape}")
print(f"🎯 Hedef değişken şekli: {y.shape}")

# 4. VERİ BÖLME
print(f"\n✂️ 4. Veri Eğitim/Test Bölümü...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_processed['sehir']
)

print(f"🏋️ Eğitim seti: {X_train.shape}")
print(f"🧪 Test seti: {X_test.shape}")

# 5. RANDOM FOREST MODELİ EĞİTİMİ
print(f"\n🌲 5. Random Forest Modeli Eğitiliyor...")

# Temel Random Forest modeli
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Modeli eğitme
print("⏳ Model eğitimi başlıyor...")
rf_model.fit(X_train, y_train)
print("✅ Model eğitimi tamamlandı!")

# 6. TAHMİN VE DEĞERLENDİRME
print(f"\n📊 6. Model Performans Değerlendirmesi...")

# Eğitim ve test tahminleri
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Metrikler hesaplama
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n📈 MODEL PERFORMANS METRİKLERİ:")
print("=" * 50)
print(f"📊 EĞİTİM SETİ:")
print(f"   MAE (Ortalama Mutlak Hata): {train_mae:,.0f} TL")
print(f"   MSE (Ortalama Kare Hata): {train_mse:,.0f}")
print(f"   RMSE (Kök Ortalama Kare Hata): {train_rmse:,.0f} TL")
print(f"   R² Skoru: {train_r2:.4f}")

print(f"\n🧪 TEST SETİ:")
print(f"   MAE (Ortalama Mutlak Hata): {test_mae:,.0f} TL")
print(f"   MSE (Ortalama Kare Hata): {test_mse:,.0f}")
print(f"   RMSE (Kök Ortalama Kare Hata): {test_rmse:,.0f} TL")
print(f"   R² Skoru: {test_r2:.4f}")

# Aşırı öğrenme kontrolü
overfitting_check = train_r2 - test_r2
print(f"\n🔍 AŞIRI ÖĞRENME KONTROLÜ:")
print(f"   Eğitim R² - Test R²: {overfitting_check:.4f}")
if overfitting_check < 0.1:
    print("   ✅ Model iyi genelleme yapıyor")
elif overfitting_check < 0.2:
    print("   ⚠️ Hafif aşırı öğrenme var")
else:
    print("   ❌ Aşırı öğrenme problemi var")

# 7. ÖZELLİK ÖNEMİ ANALİZİ
print(f"\n🎯 7. Özellik Önem Analizi...")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 EN ÖNEMLİ 10 ÖZELLİK:")
print("-" * 40)
for i, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:20s}: {row['importance']:.4f}")

# 8. CROSS VALİDATİON
print(f"\n🔄 8. Cross Validation Analizi...")

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"📊 5-Fold CV R² Skorları: {cv_scores}")
print(f"📈 Ortalama CV R² Skoru: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# 9. MODEL KAYDETME
print(f"\n💾 9. Model Kaydediliyor...")

# Model ve encoder'ları kaydetme
model_data = {
    'model': rf_model,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist(),
    'feature_importance': feature_importance,
    'metrics': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }
}

with open('ev_fiyat_tahmin_modeli.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Model 'ev_fiyat_tahmin_modeli.pkl' olarak kaydedildi!")

# 10. ÖRNEKLEMDEĞERLENDİRME
print(f"\n🏠 10. Örnek Tahmin Değerlendirmesi...")

# Test setinden rastgele 5 örnek
sample_indices = np.random.choice(X_test.index, 5, replace=False)

print(f"\n📋 RASTGELE 5 EVİN TAHMİN SONUÇLARI:")
print("-" * 60)
for idx in sample_indices:
    actual = y_test.loc[idx]
    predicted = rf_model.predict(X_test.loc[[idx]])[0]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    
    # Orijinal veri bilgileri
    original_data = df.loc[idx]
    
    print(f"🏡 Ev {idx}:")
    print(f"   📍 {original_data['sehir']} - {original_data['ilce']}")
    print(f"   🏠 {original_data['ev_tipi']}, {original_data['metrekare']}m²")
    print(f"   💰 Gerçek Fiyat: {actual:,.0f} TL")
    print(f"   🎯 Tahmin Fiyat: {predicted:,.0f} TL")
    print(f"   📊 Hata: {error:,.0f} TL (%{error_pct:.1f})")
    print()

print(f"\n🎉 MODEL GELİŞTİRME TAMAMLANDI!")
print("=" * 60)
print(f"📊 Model Özeti:")
print(f"   • Algoritma: Random Forest Regressor")
print(f"   • Test R² Skoru: {test_r2:.4f}")
print(f"   • Test RMSE: {test_rmse:,.0f} TL")
print(f"   • Test MAE: {test_mae:,.0f} TL")
print(f"   • Model Dosyası: ev_fiyat_tahmin_modeli.pkl")
print(f"✅ Model başarıyla geliştirildi ve kaydedildi!")