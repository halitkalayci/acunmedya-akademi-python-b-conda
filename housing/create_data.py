import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Rastgele seed belirleme
np.random.seed(42)
random.seed(42)

# Türkiye'deki şehirler ve ilçeler
sehir_ilce_data = {
    'İstanbul': ['Beşiktaş', 'Şişli', 'Kadıköy', 'Üsküdar', 'Beyoğlu', 'Fatih', 'Bakırköy', 'Zeytinburnu', 
                 'Maltepe', 'Pendik', 'Kartal', 'Ataşehir', 'Çekmeköy', 'Sancaktepe', 'Sultanbeyli',
                 'Bahçelievler', 'Güngören', 'Esenler', 'Gaziosmanpaşa', 'Eyüpsultan', 'Arnavutköy'],
    'Ankara': ['Çankaya', 'Keçiören', 'Yenimahalle', 'Mamak', 'Sincan', 'Etimesgut', 'Pursaklar',
               'Gölbaşı', 'Polatlı', 'Beypazarı', 'Altındağ', 'Elmadağ'],
    'İzmir': ['Konak', 'Karşıyaka', 'Bornova', 'Buca', 'Çiğli', 'Gaziemir', 'Narlıdere', 'Balçova',
              'Bayraklı', 'Güzelbahçe', 'Foça', 'Menderes', 'Torbalı', 'Seferihisar'],
    'Bursa': ['Osmangazi', 'Nilüfer', 'Yıldırım', 'Mudanya', 'Gemlik', 'İnegöl', 'Orhangazi', 'Kestel'],
    'Antalya': ['Muratpaşa', 'Kepez', 'Konyaaltı', 'Döşemealtı', 'Aksu', 'Alanya', 'Manavgat', 'Side', 'Kaş', 'Kalkan'],
    'Adana': ['Seyhan', 'Yüreğir', 'Çukurova', 'Sarıçam', 'Aladağ', 'Ceyhan', 'Kozan'],
    'Gaziantep': ['Şahinbey', 'Şehitkamil', 'Oğuzeli', 'Nizip', 'İslahiye', 'Nurdağı'],
    'Konya': ['Meram', 'Karatay', 'Selçuklu', 'Ereğli', 'Akşehir', 'Beyşehir'],
    'Mersin': ['Yenişehir', 'Mezitli', 'Toroslar', 'Akdeniz', 'Tarsus', 'Erdemli'],
    'Kayseri': ['Melikgazi', 'Kocasinan', 'Talas', 'İncesu', 'Develi']
}

# Ev tipleri ve oda sayıları
ev_tipleri = ['1+0', '1+1', '2+1', '3+1', '4+1', '5+1', '2+2', '3+2', '4+2', '5+2']

# Isınma türleri
isinma_turleri = ['Doğalgaz', 'Kombi', 'Soba', 'Merkezi', 'Klima', 'Elektrik']

# Bina yaşı kategorileri
bina_yasi_kategorileri = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31+']

def generate_housing_data(num_records=12000):
    data = []
    
    for i in range(num_records):
        # Şehir ve ilçe seçimi
        sehir = random.choice(list(sehir_ilce_data.keys()))
        ilce = random.choice(sehir_ilce_data[sehir])
        
        # Ev tipi
        ev_tipi = random.choice(ev_tipleri)
        
        # Oda sayısından metrekareyi tahmin etme
        oda_sayisi = int(ev_tipi.split('+')[0])
        salon_sayisi = int(ev_tipi.split('+')[1])
        
        # Metrekare hesaplama (oda sayısına göre)
        base_m2 = (oda_sayisi * 12) + (salon_sayisi * 20) + 15  # Banyo, mutfak, koridor
        metrekare = int(np.random.normal(base_m2, base_m2 * 0.2))
        metrekare = max(35, min(metrekare, 500))  # 35-500 m2 arası sınırlama
        
        # Kat bilgileri
        bina_kat_sayisi = random.choice([3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30])
        bulundugu_kat = random.randint(0, bina_kat_sayisi)  # 0 = zemin kat
        
        # Bina yaşı
        bina_yasi_kategori = random.choice(bina_yasi_kategorileri)
        if bina_yasi_kategori == '0-5':
            bina_yasi = random.randint(0, 5)
        elif bina_yasi_kategori == '6-10':
            bina_yasi = random.randint(6, 10)
        elif bina_yasi_kategori == '11-15':
            bina_yasi = random.randint(11, 15)
        elif bina_yasi_kategori == '16-20':
            bina_yasi = random.randint(16, 20)
        elif bina_yasi_kategori == '21-25':
            bina_yasi = random.randint(21, 25)
        elif bina_yasi_kategori == '26-30':
            bina_yasi = random.randint(26, 30)
        else:  # 31+
            bina_yasi = random.randint(31, 50)
        
        # Boolean özellikler
        balkon = random.choice([True, False])
        asansor = random.choice([True, False]) if bina_kat_sayisi > 4 else random.choice([True, False])
        park_yeri = random.choice([True, False])
        site_icinde = random.choice([True, False])
        esyali = random.choice([True, False])
        
        # Isınma türü
        isinma = random.choice(isinma_turleri)
        
        # Banyo sayısı
        banyo_sayisi = max(1, min(oda_sayisi // 2 + 1, 4))
        
        # Fiyat hesaplama (şehir, ilçe, özellikler bazında)
        # Şehir katsayıları
        sehir_multiplier = {
            'İstanbul': 1.8, 'Ankara': 1.3, 'İzmir': 1.2, 'Bursa': 1.0, 'Antalya': 1.4,
            'Adana': 0.7, 'Gaziantep': 0.6, 'Konya': 0.65, 'Mersin': 0.75, 'Kayseri': 0.7
        }
        
        # İlçe katsayıları (lüks ilçeler daha pahalı)
        lux_ilceler = ['Beşiktaş', 'Şişli', 'Kadıköy', 'Üsküdar', 'Beyoğlu', 'Çankaya', 'Konak', 
                       'Karşıyaka', 'Bornova', 'Nilüfer', 'Muratpaşa', 'Konyaaltı']
        ilce_multiplier = 1.3 if ilce in lux_ilceler else 1.0
        
        # Temel fiyat (m2 başına)
        base_price_per_m2 = 8000  # TL
        
        # Fiyat hesaplama
        price = metrekare * base_price_per_m2 * sehir_multiplier[sehir] * ilce_multiplier
        
        # Özellik bonusları
        if balkon:
            price *= 1.05
        if asansor:
            price *= 1.08
        if park_yeri:
            price *= 1.12
        if site_icinde:
            price *= 1.15
        if esyali:
            price *= 1.2
        
        # Yaş penaltisi
        age_penalty = max(0.6, 1 - (bina_yasi * 0.015))
        price *= age_penalty
        
        # Kat bonusu/penaltisi
        if bulundugu_kat == 0:  # Zemin kat
            price *= 0.95
        elif bulundugu_kat <= 3:  # Alt katlar
            price *= 1.02
        elif bulundugu_kat > bina_kat_sayisi * 0.8:  # Üst katlar
            price *= 1.08
        
        # Rastgele varyasyon ekleme
        price *= np.random.normal(1.0, 0.15)
        price = max(300000, int(price))  # Minimum 300k TL
        
        # Veriyi kaydetme
        data.append({
            'sehir': sehir,
            'ilce': ilce,
            'ev_tipi': ev_tipi,
            'metrekare': metrekare,
            'oda_sayisi': oda_sayisi,
            'salon_sayisi': salon_sayisi,
            'banyo_sayisi': banyo_sayisi,
            'bina_yasi': bina_yasi,
            'bina_kat_sayisi': bina_kat_sayisi,
            'bulundugu_kat': bulundugu_kat,
            'balkon': balkon,
            'asansor': asansor,
            'park_yeri': park_yeri,
            'site_icinde': site_icinde,
            'esyali': esyali,
            'isinma_turu': isinma,
            'fiyat_tl': price
        })
        
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} kayıt oluşturuldu...")
    
    return pd.DataFrame(data)

# Veri setini oluştur
print("Türkiye ev fiyatları veri seti oluşturuluyor...")
df = generate_housing_data(12000)

# Veri setini CSV olarak kaydet
df.to_csv('turkiye_ev_fiyatlari.csv', index=False, encoding='utf-8')

print(f"\nVeri seti başarıyla oluşturuldu!")
print(f"Toplam kayıt sayısı: {len(df)}")
print(f"Dosya adı: turkiye_ev_fiyatlari.csv")
print(f"\nVeri seti önizlemesi:")
print(df.head())
print(f"\nFiyat istatistikleri:")
print(df['fiyat_tl'].describe())
print(f"\nŞehir dağılımı:")
print(df['sehir'].value_counts()) 