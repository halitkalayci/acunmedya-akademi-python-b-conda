# 🏠 Türkiye Ev Fiyat Tahmin API

Random Forest algoritması kullanarak Türkiye'deki ev fiyatlarını tahmin eden RESTful API.

## 📊 Proje Özeti

Bu proje, 12,000 adet gerçekçi ev verisi kullanarak Türkiye'deki ev fiyatlarını tahmin eden bir makine öğrenmesi modelini FastAPI ile web servis olarak sunar.

### 🎯 Model Performansı
- **Algoritma**: Random Forest Regressor
- **Test R² Skoru**: 0.8469 (%84.7 doğruluk)
- **Test MAE**: 101,407 TL
- **Test RMSE**: 167,148 TL

### 🔍 En Önemli Özellikler
1. **Şehir** (42.97%) - En kritik faktör
2. **Metrekare** (35.13%) - İkinci en önemli
3. **Bina Yaşı** (10.70%)
4. **Eşyalı** (2.28%)
5. **İlçe** (1.86%)

## 📁 Dosya Yapısı

```
housing/
├── api.py                          # FastAPI web servisi
├── main.py                         # Model eğitim scripti
├── demo_api.py                     # API demo scripti
├── test_api.py                     # API test scripti
├── start_api.bat                   # Windows API başlatma dosyası
├── requirements.txt                # Python paket listesi
├── turkiye_ev_fiyatlari.csv       # Eğitim veri seti (12K kayıt)
├── ev_fiyat_tahmin_modeli.pkl     # Eğitimli model dosyası
└── README.md                       # Bu dosya
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install fastapi uvicorn pandas numpy scikit-learn requests
```

### 2. API'yi Başlatma

**Yöntem 1: Doğrudan**
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

**Yöntem 2: Windows Batch**
```bash
start_api.bat
```

**Yöntem 3: Python**
```bash
python api.py
```

### 3. API Erişimi
- **Base URL**: http://127.0.0.1:8000
- **Dokümantasyon**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 🌐 API Endpoint'leri

### 1. Ana Sayfa
```
GET /
```
API hakkında genel bilgi döner.

### 2. Ev Fiyat Tahmini
```
POST /predict
```

**Request Body Örneği:**
```json
{
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
  "balkon": true,
  "asansor": true,
  "park_yeri": true,
  "site_icinde": true,
  "esyali": false,
  "isinma_turu": "Doğalgaz"
}
```

**Response Örneği:**
```json
{
  "tahmin_fiyat_tl": 1500000,
  "tahmin_fiyat_formatted": "1.500.000 TL",
  "girdi_verileri": { ... },
  "model_bilgileri": {
    "algoritma": "Random Forest Regressor",
    "test_r2_skoru": 0.8469,
    "guvenilirlik": "Yüksek"
  }
}
```

### 3. Model Metrikleri
```
GET /metrics
```
Model performans metriklerini döner.

### 4. Sağlık Kontrolü
```
GET /health
```
API'nin durumunu kontrol eder.

## 📝 Kullanım Örnekleri

### cURL ile Kullanım
```bash
# Tahmin yapma
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "balkon": true,
    "asansor": true,
    "park_yeri": true,
    "site_icinde": true,
    "esyali": false,
    "isinma_turu": "Doğalgaz"
  }'

# Metrikler alma
curl http://127.0.0.1:8000/metrics
```

### Python ile Kullanım
```python
import requests

# Tahmin yapma
data = {
    "sehir": "Ankara",
    "ilce": "Çankaya",
    "ev_tipi": "3+1",
    "metrekare": 110,
    # ... diğer alanlar
}

response = requests.post(
    "http://127.0.0.1:8000/predict", 
    json=data
)
print(response.json())
```

## 🎯 Test ve Demo

### Demo Çalıştırma
```bash
python demo_api.py
```
Model ve API özelliklerini gösterir.

### API Testi
```bash
python test_api.py
```
Tüm endpoint'leri test eder.

## 📊 Veri Seti Özellikleri

### Lokasyon
- **Şehirler**: İstanbul, Ankara, İzmir, Bursa, Antalya, Adana, Gaziantep, Konya, Mersin, Kayseri
- **İlçeler**: Her şehir için gerçek ilçe isimleri

### Ev Özellikleri
- **Ev Tipi**: 1+0'dan 5+2'ye kadar
- **Metrekare**: 35-500 m²
- **Oda/Salon/Banyo Sayısı**: Gerçekçi oranlar
- **Bina Bilgileri**: Yaş, kat sayısı, bulunduğu kat

### Özellikler
- **Boolean**: Balkon, asansör, park yeri, site içinde, eşyalı
- **Isınma**: Doğalgaz, kombi, soba, merkezi, klima, elektrik

### Fiyat Aralığı
- **Minimum**: 300,000 TL
- **Maksimum**: 5,682,775 TL
- **Ortalama**: 670,511 TL

## 🔧 Teknik Detaylar

### Model Özellikleri
- **Algoritma**: Random Forest Regressor
- **Hiperparametreler**: 
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 2

### Veri İşleme
- **Kategorik Encoding**: Label Encoder
- **Boolean Dönüşüm**: True/False → 1/0
- **Validation**: 5-Fold Cross Validation

### API Teknolojileri
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Validation**: Pydantic
- **Format**: JSON REST API

## 🚨 Kısıtlamalar

1. **Şehir Sınırlaması**: Sadece 10 büyük şehir desteklenir
2. **İlçe Kısıtlaması**: Veri setindeki ilçelerle sınırlı
3. **Tahmin Aralığı**: Minimum 300,000 TL
4. **Model Tarihi**: 2025 verilerine dayanır

## 📈 Gelecek Geliştirmeler

- [ ] Daha fazla şehir desteği
- [ ] Gerçek zamanlı veri entegrasyonu
- [ ] Model versiyonlama
- [ ] Docker konteynerizasyon
- [ ] Veritabanı entegrasyonu
- [ ] Kullanıcı kimlik doğrulaması

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

**Proje**: Türkiye Ev Fiyat Tahmin API  
**Teknoloji**: Python, FastAPI, Random Forest, scikit-learn  
**Tarih**: 2025  

---

💡 **İpucu**: API dokümantasyonu için http://127.0.0.1:8000/docs adresini ziyaret edin! 