#!/usr/bin/env python3
"""
Türkiye Ev Fiyat Tahmin API Demo
Bu script, API'yi başlatır ve farklı endpoint'leri test eder
"""

import pickle
import pandas as pd
import json

def load_and_test_model():
    """Model dosyasını yükleyip test et"""
    print("🏠 Türkiye Ev Fiyat Tahmin API Demo")
    print("=" * 60)
    
    # Model yükleme
    print("\n📊 Model Yükleniyor...")
    try:
        with open('ev_fiyat_tahmin_modeli.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model başarıyla yüklendi!")
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return
    
    # 1. Model metrikleri göster
    print("\n📈 MODEL PERFORMANS METRİKLERİ:")
    print("-" * 50)
    metrics = model_data['metrics']
    print(f"🎯 Algoritma: Random Forest Regressor")
    print(f"📊 Test R² Skoru: {metrics['test_r2']:.4f} (%{metrics['test_r2']*100:.1f})")
    print(f"💰 Test MAE: {metrics['test_mae']:,.0f} TL")
    print(f"📐 Test RMSE: {metrics['test_rmse']:,.0f} TL")
    print(f"🏋️ Eğitim R² Skoru: {metrics['train_r2']:.4f}")
    
    # Aşırı öğrenme kontrolü
    overfitting_diff = metrics['train_r2'] - metrics['test_r2']
    if overfitting_diff < 0.1:
        status = "✅ Model iyi genelleme yapıyor"
    elif overfitting_diff < 0.2:
        status = "⚠️ Hafif aşırı öğrenme var"
    else:
        status = "❌ Aşırı öğrenme problemi var"
    print(f"🔍 Durum: {status}")
    
    # 2. Özellik önem analizi
    print(f"\n🎯 EN ÖNEMLİ 10 ÖZELLİK:")
    print("-" * 40)
    feature_importance = model_data['feature_importance']
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")
    
    # 3. Örnek tahmin testleri
    print(f"\n🏡 ÖRNEK TAHMİN TESTLERİ:")
    print("-" * 50)
    
    # Test verisi 1: Lüks İstanbul evi
    test_1 = {
        "sehir": "İstanbul",
        "ilce": "Beşiktaş",
        "ev_tipi": "4+1",
        "metrekare": 150,
        "oda_sayisi": 4,
        "salon_sayisi": 1,
        "banyo_sayisi": 3,
        "bina_yasi": 2,
        "bina_kat_sayisi": 12,
        "bulundugu_kat": 8,
        "balkon": True,
        "asansor": True,
        "park_yeri": True,
        "site_icinde": True,
        "esyali": False,
        "isinma_turu": "Doğalgaz"
    }
    
    prediction_1 = predict_with_model(model_data, test_1)
    print(f"\n🏠 Test 1 - Lüks İstanbul Evi:")
    print(f"   📍 {test_1['sehir']} - {test_1['ilce']}")
    print(f"   🏡 {test_1['ev_tipi']}, {test_1['metrekare']}m², {test_1['bina_yasi']} yaş")
    print(f"   💰 Tahmin Fiyat: {prediction_1:,.0f} TL")
    
    # Test verisi 2: Orta segment Ankara evi
    test_2 = {
        "sehir": "Ankara",
        "ilce": "Keçiören",
        "ev_tipi": "3+1",
        "metrekare": 110,
        "oda_sayisi": 3,
        "salon_sayisi": 1,
        "banyo_sayisi": 2,
        "bina_yasi": 8,
        "bina_kat_sayisi": 6,
        "bulundugu_kat": 3,
        "balkon": True,
        "asansor": True,
        "park_yeri": False,
        "site_icinde": False,
        "esyali": True,
        "isinma_turu": "Kombi"
    }
    
    prediction_2 = predict_with_model(model_data, test_2)
    print(f"\n🏠 Test 2 - Orta Segment Ankara Evi:")
    print(f"   📍 {test_2['sehir']} - {test_2['ilce']}")
    print(f"   🏡 {test_2['ev_tipi']}, {test_2['metrekare']}m², {test_2['bina_yasi']} yaş")
    print(f"   💰 Tahmin Fiyat: {prediction_2:,.0f} TL")
    
    # Test verisi 3: Ekonomik İzmir evi
    test_3 = {
        "sehir": "İzmir",
        "ilce": "Buca",
        "ev_tipi": "2+1",
        "metrekare": 85,
        "oda_sayisi": 2,
        "salon_sayisi": 1,
        "banyo_sayisi": 1,
        "bina_yasi": 15,
        "bina_kat_sayisi": 4,
        "bulundugu_kat": 1,
        "balkon": False,
        "asansor": False,
        "park_yeri": False,
        "site_icinde": False,
        "esyali": False,
        "isinma_turu": "Soba"
    }
    
    prediction_3 = predict_with_model(model_data, test_3)
    print(f"\n🏠 Test 3 - Ekonomik İzmir Evi:")
    print(f"   📍 {test_3['sehir']} - {test_3['ilce']}")
    print(f"   🏡 {test_3['ev_tipi']}, {test_3['metrekare']}m², {test_3['bina_yasi']} yaş")
    print(f"   💰 Tahmin Fiyat: {prediction_3:,.0f} TL")
    
    # 4. API endpoint bilgileri
    print(f"\n🌐 API ENDPOINT'LERİ:")
    print("-" * 40)
    print(f"🏠 API Base URL: http://127.0.0.1:8000")
    print(f"📋 Ana Sayfa: GET /")
    print(f"🎯 Tahmin: POST /predict")
    print(f"📊 Metrikler: GET /metrics")
    print(f"❤️ Sağlık: GET /health")
    print(f"📚 Dokümantasyon: GET /docs")
    
    # 5. Örnek API kullanımı
    print(f"\n💻 ÖRNEK API KULLANIMI:")
    print("-" * 40)
    
    api_example = {
        "method": "POST",
        "url": "http://127.0.0.1:8000/predict",
        "headers": {"Content-Type": "application/json"},
        "body": test_1
    }
    
    print(f"curl -X POST http://127.0.0.1:8000/predict \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(test_1, ensure_ascii=False)}'")
    
    print(f"\n🎉 Demo Tamamlandı!")
    print(f"✅ Model hazır ve çalışıyor!")

def predict_with_model(model_data, input_data):
    """Model ile tahmin yap"""
    try:
        # DataFrame'e çevirme
        df = pd.DataFrame([input_data])
        
        # Kategorik verileri encode etme
        for col, encoder in model_data['label_encoders'].items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Boolean verileri 0/1'e çevirme
        boolean_columns = ['balkon', 'asansor', 'park_yeri', 'site_icinde', 'esyali']
        for col in boolean_columns:
            df[col] = df[col].astype(int)
        
        # Sütun sırasını ayarlama
        df = df[model_data['feature_names']]
        
        # Tahmin yapma
        prediction = model_data['model'].predict(df)[0]
        return max(300000, int(prediction))
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return 0

if __name__ == "__main__":
    load_and_test_model() 