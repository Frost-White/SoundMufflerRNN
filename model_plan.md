# Model Training Plan (MVP)

Bu dokuman, `Gurultu_Engelleme_PRD_v1.docx` icindeki model gelistirme gereksinimlerinden (ozellikle Blok 1 + Sprint 1) turetilmistir.  
Mevcut durum: **veri on isleme tamamlandi**. Bundan sonraki odak egitim ve degerlendirme dongulerinin uygulanmasidir.

## 1) Hedefler ve Basari Kriterleri

- Mimari odagi:
  - Birincil aday: `DTLN` (dusuk gecikme, streaming uyumu)
  - Alternatif/ablasyon: kucuk `Conv-TasNet`
- Cikti kalitesi hedefleri:
  - `PESQ >= 3.5`
  - `STOI >= 0.85`
  - `SNR improvement >= 10 dB`
- Performans hedefi:
  - `Inference latency < 20 ms/chunk` (20 ms chunk yapisinda)

## 2) Pipeline Varsayimlari (On Isleme Sonrasi)

- Dataset split hazir: `train / val / test` (speaker/noise leakage olmayacak sekilde)
- Giris-cikis ciftleri hazir:
  - Giris: noisy chunk/sinyal
  - Hedef: clean chunk/sinyal
- Chunk stratejisi:
  - `20 ms` chunk (WebRTC uyumlu)
  - Hem egitimde hem inferenceda ayni chunk semantiği korunacak

## 3) Egitim Dongusu (Train Loop) - Yapilacaklar

1. **Model ve optimizer kurulumu**
   - `DTLN` model init
   - Optimizer: `Adam/AdamW`
   - LR scheduler: `ReduceLROnPlateau` veya cosine
2. **Loss tanimi**
   - Baslangic: zaman domeni `L1/L2` + opsiyonel SI-SDR tabanli kayip
   - Gerekirse multi-objective loss (kalite + anlasilabilirlik dengesi)
3. **Batch akisi**
   - Noisy input -> model -> denoised output
   - Loss hesapla -> backprop -> optimizer step
   - Gradient clipping (egitim stabilitesi icin)
4. **Egitim stabilitesi**
   - Mixed precision (varsa) ile hiz/memory kazanimi
   - NaN/inf guard kontrolu
5. **Logging**
   - Her N step: train loss, lr, step time
   - Her epoch: ortalama train metrikleri

## 4) Degerlendirme Dongusu (Eval/Validation Loop) - Yapilacaklar

1. **Validation adimi (her epoch sonu)**
   - `model.eval()` + `torch.no_grad()`
   - Val set uzerinde loss + metrik hesapla
2. **Metrikler**
   - `PESQ`, `STOI`, `SNR improvement`
3. **Checkpoint stratejisi**
   - `best_pesq.ckpt` (ana secim)
   - `best_stoi.ckpt` (ikincil)
   - `last.ckpt` (resume icin)
4. **Early stopping**
   - Ana metrikte (PESQ/STOI) iyilesme durursa durdur
5. **Model secimi**
   - Final model: val metrik + latency birlikte optimize edilerek secilecek

## 5) Test Dongusu ve Raporlama

- Test setinde tek seferlik final rapor:
  - PESQ, STOI, SNR
  - 20 ms chunk inferenceda ortalama ve p95 latency
- Use-case bazli alt raporlar:
  - Video konferans gurultuleri
  - Oyun ici (klavye/fan/ani impuls) gurultuleri
- A/B ciktilari:
  - Noisy vs Clean vs Predicted ses ornekleri (dinleme testi icin)

## 6) Deney Plani (Ablation / Iterasyon)

1. **Baseline DTLN**
   - Hedef: calisan ve stabil baseline
2. **Hyperparameter taramasi**
   - LR, batch size, loss agirliklari, scheduler
3. **Augmentation etkisi**
   - Oyun-gurultu agirlikli sentetik karisimlarin etkisi
4. **Alternatif mimari**
   - Kucuk `Conv-TasNet` ile kalite/gecikme karsilastirmasi
5. **Secim karari**
   - Metrik + latency + model boyutu uzerinden MVP model freeze

## 7) Cikis Kriterleri (Sprint 1 Done)

- [ ] Train loop stabil sekilde calisiyor (resume destekli)
- [ ] Eval loop her epoch otomatik calisiyor
- [ ] Checkpoint + early stopping aktif
- [ ] Metrik raporu otomatik uretiliyor (PESQ/STOI/SNR)
- [ ] 20 ms chunk inferenceda latency olcum scripti hazir
- [ ] MVP hedefleri (veya hedefe uzaklik analizi) dokumante edildi

## 8) Hemen Siradaki Isler (Senin Durumuna Gore)

Veri on isleme tamam oldugu icin siradaki uygulanacak adimlar:

1. `train.py` icinde tam train loop'u devreye al
2. `evaluate.py` veya train icinde epoch-sonu eval loop ekle
3. PESQ/STOI/SNR hesap modullerini bagla
4. Best checkpoint secimini `PESQ` odakli yap
5. Ayrica bir `latency_benchmark.py` ile `<20ms/chunk` dogrulamasi yap
