import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("🧪 Türkiye Ev Fiyat Tahmin API Testi")
    print("=" * 50)
    
    # 1. Ana endpoint testi
    print("\n1️⃣ Ana Endpoint Testi...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        return
    
    # 2. Health check testi
    print("\n2️⃣ Sağlık Kontrolü Testi...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # 3. Metrikler endpoint testi
    print("\n3️⃣ Model Metrikleri Testi...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # 4. Tahmin endpoint testi
    print("\n4️⃣ Ev Fiyat Tahmini Testi...")
    
    # Test verisi
    test_data = {
        "sehir": "İstanbul",
        "ilce": "Kadıköy",
        "ev_tipi": "3+1",
        "metrekare": 100,
        "oda_sayisi": 3,
        "salon_sayisi": 1,
        "banyo_sayisi": 2,
        "bina_yasi": 5,
        "bina_kat_sayisi": 8,
        "bulundugu_kat": 3,
        "balkon": True,
        "asansor": True,
        "park_yeri": True,
        "site_icinde": True,
        "esyali": False,
        "isinma_turu": "Doğalgaz"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # 5. Farklı bir şehir ile test
    print("\n5️⃣ Ankara Evi Tahmin Testi...")
    
    ankara_data = {
        "sehir": "Ankara",
        "ilce": "Çankaya",
        "ev_tipi": "2+1",
        "metrekare": 85,
        "oda_sayisi": 2,
        "salon_sayisi": 1,
        "banyo_sayisi": 1,
        "bina_yasi": 10,
        "bina_kat_sayisi": 5,
        "bulundugu_kat": 2,
        "balkon": False,
        "asansor": False,
        "park_yeri": False,
        "site_icinde": False,
        "esyali": True,
        "isinma_turu": "Kombi"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=ankara_data)
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    print(f"\n🎉 API Test Tamamlandı!")
    print(f"🌐 API Dokümantasyonu: {BASE_URL}/docs")

if __name__ == "__main__":
    # API'nin başlaması için biraz bekle
    print("⏳ API'nin başlaması bekleniyor...")
    time.sleep(2)
    test_api() 