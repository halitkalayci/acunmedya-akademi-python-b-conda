from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Global değişkenler
model_data = None

# Model yükleme fonksiyonu
def load_model():
    global model_data
    try:
        with open('ev_fiyat_tahmin_modeli.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model başarıyla yüklendi!")
        return True
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return False

# Uygulama yaşam döngüsü yönetimi
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not load_model():
        raise Exception("Model yüklenemedi!")
    yield
    # Shutdown (gerekirse cleanup kodu buraya)

# FastAPI uygulaması oluşturma
app = FastAPI(
    title="Türkiye Ev Fiyat Tahmin API",
    description="Random Forest algoritması ile Türkiye'deki ev fiyatlarını tahmin eden API",
    version="1.0.0",
    lifespan=lifespan
)

# Tahmin isteği için veri modeli
class EvTahminRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )
    
    sehir: str
    ilce: str
    ev_tipi: str
    metrekare: int
    oda_sayisi: int
    salon_sayisi: int
    banyo_sayisi: int
    bina_yasi: int
    bina_kat_sayisi: int
    bulundugu_kat: int
    balkon: bool
    asansor: bool
    park_yeri: bool
    site_icinde: bool
    esyali: bool
    isinma_turu: str

# Tahmin sonucu için veri modeli
class EvTahminResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    tahmin_fiyat_tl: int
    tahmin_fiyat_formatted: str
    girdi_verileri: dict
    model_bilgileri: dict

# Model metrikleri için veri modeli
class ModelMetrikleri(BaseModel):
    algoritma: str
    test_r2_skoru: float
    test_mae_tl: int
    test_rmse_tl: int
    train_r2_skoru: float
    train_mae_tl: int
    train_rmse_tl: int
    asiri_ogrenme_kontrol: str
    ozelliklerin_onemi: dict

@app.get("/", summary="Ana Sayfa")
async def root():
    """API ana sayfası - API hakkında genel bilgi"""
    return {
        "mesaj": "🏠 Türkiye Ev Fiyat Tahmin API'sine Hoş Geldiniz!",
        "aciklama": "Bu API Random Forest algoritması ile Türkiye'deki ev fiyatlarını tahmin eder",
        "endpoints": {
            "/predict": "POST - Ev fiyat tahmini yapın",
            "/metrics": "GET - Model performans metriklerini görün",
            "/docs": "GET - API dokümantasyonu"
        },
        "model_durumu": "Aktif" if model_data else "Pasif"
    }

@app.post("/predict", response_model=EvTahminResponse, summary="Ev Fiyat Tahmini")
async def predict_price(ev_data: EvTahminRequest):
    """
    Verilen ev özelliklerine göre fiyat tahmini yapar
    
    - **sehir**: Şehir adı (İstanbul, Ankara, İzmir, vs.)
    - **ilce**: İlçe adı
    - **ev_tipi**: Ev tipi (1+1, 2+1, 3+1, vs.)
    - **metrekare**: Ev büyüklüğü (m²)
    - **oda_sayisi**: Yatak odası sayısı
    - **salon_sayisi**: Salon sayısı
    - **banyo_sayisi**: Banyo sayısı
    - **bina_yasi**: Binanın yaşı
    - **bina_kat_sayisi**: Binanın toplam kat sayısı
    - **bulundugu_kat**: Dairenin bulunduğu kat
    - **balkon**: Balkon var mı? (true/false)
    - **asansor**: Asansör var mı? (true/false)
    - **park_yeri**: Park yeri var mı? (true/false)
    - **site_icinde**: Site içinde mi? (true/false)
    - **esyali**: Eşyalı mı? (true/false)
    - **isinma_turu**: Isınma türü (Doğalgaz, Kombi, vs.)
    """
    
    if not model_data:
        raise HTTPException(status_code=500, detail="Model yüklenmemiş!")
    
    try:
        # Giriş verilerini DataFrame'e çevirme
        input_data = pd.DataFrame([ev_data.model_dump()])
        
        # Kategorik verileri encode etme
        for col, encoder in model_data['label_encoders'].items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col])
                except ValueError as e:
                    # Bilinmeyen kategori durumu
                    available_classes = list(encoder.classes_)
                    raise HTTPException(
                        status_code=400, 
                        detail=f"'{col}' için geçersiz değer: '{ev_data.model_dump()[col]}'. Geçerli değerler: {available_classes}"
                    )
        
        # Boolean verileri 0/1'e çevirme
        boolean_columns = ['balkon', 'asansor', 'park_yeri', 'site_icinde', 'esyali']
        for col in boolean_columns:
            input_data[col] = input_data[col].astype(int)
        
        # Sütun sırasını model ile uyumlu hale getirme
        input_data = input_data[model_data['feature_names']]
        
        # Tahmin yapma
        prediction = model_data['model'].predict(input_data)[0]
        prediction = max(300000, int(prediction))  # Minimum 300k TL
        
        # Sonuç hazırlama
        return EvTahminResponse(
            tahmin_fiyat_tl=prediction,
            tahmin_fiyat_formatted=f"{prediction:,} TL".replace(",", "."),
            girdi_verileri=ev_data.model_dump(),
            model_bilgileri={
                "algoritma": "Random Forest Regressor",
                "test_r2_skoru": round(model_data['metrics']['test_r2'], 4),
                "guvenilirlik": "Yüksek" if model_data['metrics']['test_r2'] > 0.8 else "Orta"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata: {str(e)}")

@app.get("/metrics", response_model=ModelMetrikleri, summary="Model Performans Metrikleri")
async def get_metrics():
    """
    Random Forest modelinin performans metriklerini döner
    
    Bu endpoint model hakkında aşağıdaki bilgileri sağlar:
    - Test ve eğitim R² skorları
    - Ortalama mutlak hata (MAE) değerleri
    - Kök ortalama kare hata (RMSE) değerleri
    - Aşırı öğrenme kontrol durumu
    - En önemli özelliklerin listesi
    """
    
    if not model_data:
        raise HTTPException(status_code=500, detail="Model yüklenmemiş!")
    
    try:
        metrics = model_data['metrics']
        feature_importance = model_data['feature_importance']
        
        # Aşırı öğrenme kontrolü
        overfitting_diff = metrics['train_r2'] - metrics['test_r2']
        if overfitting_diff < 0.1:
            overfitting_status = "✅ Model iyi genelleme yapıyor"
        elif overfitting_diff < 0.2:
            overfitting_status = "⚠️ Hafif aşırı öğrenme var"
        else:
            overfitting_status = "❌ Aşırı öğrenme problemi var"
        
        # En önemli 10 özellik
        top_features = {}
        for _, row in feature_importance.head(10).iterrows():
            top_features[row['feature']] = round(row['importance'], 4)
        
        return ModelMetrikleri(
            algoritma="Random Forest Regressor",
            test_r2_skoru=round(metrics['test_r2'], 4),
            test_mae_tl=int(metrics['test_mae']),
            test_rmse_tl=int(metrics['test_rmse']),
            train_r2_skoru=round(metrics['train_r2'], 4),
            train_mae_tl=int(metrics['train_mae']),
            train_rmse_tl=int(metrics['train_rmse']),
            asiri_ogrenme_kontrol=overfitting_status,
            ozelliklerin_onemi=top_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrikler alınırken hata: {str(e)}")

@app.get("/health", summary="Sağlık Kontrolü")
async def health_check():
    """API'nin sağlığını kontrol eder"""
    return {
        "durum": "Sağlıklı",
        "model_durumu": "Yüklü" if model_data else "Yüklenmemiş",
        "api_versiyonu": "1.0.0"
    }

if __name__ == "__main__":
    print("🚀 Türkiye Ev Fiyat Tahmin API başlatılıyor...")
    # reload için import string kullanımı gerekiyor
    import subprocess
    subprocess.run(["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]) 