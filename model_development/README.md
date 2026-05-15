# SoundMufflerRNN — model_development

Bu klasör, gürültülü/temiz eşleşmiş WAV çiftlerinden **STFT magnitüd** özellikleriyle **GRU tabanlı** bir maskeleme modeli eğitmek ve tek dosya demo çıktısı üretmek için kullanılır.

## Bağımlılıklar

Proje kökündeki `requirements.txt` (torch, soundfile, librosa, numpy, matplotlib, …). Önerilen çalıştırma:

- Sanal ortam: repo kökünde `.venv`
- Eğitim: `model_development` içinden `python train.py` (import yolları buna göre)

## Dosya yapısı (özet)

| Dosya | Rol |
|--------|-----|
| `train.py` | Hiperparametreler, veri hazırlığı, DataLoader, eğitim başlatma |
| `model.py` | `GRUChunkDenoiser`, `FREQ_BINS`, `model_info` |
| `audio_pipeline.py` | WAV okuma, overlap’lı zaman chunk’ları, chunk başına STFT |
| `training_data.py` | Çift tarama, RAM’e magnitüd preload, `Dataset`, pad collate |
| `training_loop.py` | Maskeli MSE, val SNR, epoch döngüsü, checkpoint |
| `run_artifacts.py` | Run klasörü, plotlar, CSV, JSON özet, `model_info.txt` |
| `io_progress.py` | Preload sırasında konsol ilerlemesi |
| `demo_single_pair_forward.py` | Tek noisy WAV → maske → overlap-add WAV |
| `runs/` | Her eğitim koşusunun çıktıları |

## Model: `GRUChunkDenoiser` (`model.py`)

- **Girdi:** `x` şekli `(batch, time, FREQ_BINS)` — her zaman adımı bir **zaman chunk’ının** log-magnitüdü: `log(|STFT| + eps)`.
- **Ek girdi:** `lengths` — `(batch,)` uzunluk vektörü; batch içinde farklı ses uzunlukları için **padding** kullanılır, GRU tarafında `pack_padded_sequence` ile pad zamanları işlenmez.
- **Çıktı:** `(batch, time, FREQ_BINS)` aralığında **Sigmoid maske** \((0, 1)\).
- **Tahmin magnitüd:** `pred_mag = mask * mag_noisy` (noisy magnitüd ile çarpım).
- **Mimari:** `nn.GRU(FREQ_BINS → hidden_dim, num_layers, batch_first=True)` + `Linear(hidden_dim → FREQ_BINS)` + `Sigmoid`.
- **Frekans boyutu:** `N_FFT = 960` → `FREQ_BINS = N_FFT // 2 + 1` (**481**).

`train.py` içindeki varsayılan model hiperparametreleri: `hidden_dim`, `gru_num_layers`, `gru_dropout` (dropout yalnızca `num_layers > 1` iken anlamlı).

## Ses ön-işleme (`audio_pipeline.py`)

1. **Örnekleme:** 48 kHz mono (stereo ise kanal ortalaması).
2. **Zaman chunk’ları:** `CHUNK_SAMPLES = 960` (20 ms), hop `CHUNK_HOP = 720` → **240 örnek overlap** (5 ms).
3. **Chunk başına STFT:** `librosa.stft`, `N_FFT = 960`, `STFT_HOP = 240`, `center=False`. Şu anki ayarla çoğu chunk için tek frekans çerçevesi → tensör `(num_chunks, FREQ_BINS)` karmaşık; eğitimde **magnitüd** kullanılır.

## Veri nasıl yükleniyor ve modele nasıl gidiyor?

### 1) Çift listesi

`training_data.collect_pairs(noisy_root, clean_root)`:

- Her iki kökte `.wav` taranır, **dosya adı (basename)** ile noisy ↔ clean eşlenir.
- Clean’i olmayan noisy dosyalar uyarıyla atlanır.

### 2) RAM preload (önemli)

`preload_stft_mag_pairs`:

- Her çift için diskten iki dalga okunur, uzunluk `min(noisy, clean)` ile hizalanır.
- `chunk_waveform` → `stft_chunks` → `|.|` ile **`mag_noisy[T, F]`**, **`mag_clean[T, F]`** (float32) üretilir.
- Sonuçlar bir **sözlükte** `(noisy_path, clean_path) → (mag_n, mag_c)` olarak **RAM’de** tutulur.

Bu tasarım **epoch boyunca diske tekrar gitmez**; hızlıdır ama **RAM kullanımı yüksektir**. RAM kısıtlıysa ileride “diskten veya cache’ten `__getitem__` içinde okuma” şeklinde değiştirilebilir (şu anki kod preload bekler).

### 3) Train / val bölmesi

`prepare_train_val_datasets`:

- Çiftler karıştırılır, `val_fraction` kadarı validation anahtarlarına ayrılır.
- Train ve val için ayrı küçük `feats` sözlükleri oluşturulup büyük sözlük silinir (bir miktar RAM rahatlatma).

### 4) Dataset

`UtteranceMagDataset`:

- **Bir örnek = bir wav çiftinin tamamı:** tensörler `(T, FREQ_BINS)`.
- `__getitem__` dönüşü: `x` (log-mag), `mag_noisy`, `mag_clean`.

### 5) Batch ve padding

`collate_padded_utterances`:

- Aynı batch’teki farklı `T` değerleri `pad_sequence` ile `(B, T_max, F)` olur.
- `lengths`: her örneğin gerçek zaman uzunluğu.

### 6) Eğitim adımı (`training_loop.py`)

- `mask = model(x, lengths)` → `pred_mag = mask * mag_n`.
- **Kayıp:** magnitüd alanında maskeli MSE — pad edilen zaman indeksleri hem kayıpta hem val metriklerinde **sayılmaz**.
- **Val SNR kazancı (dB):** aynı geçerli hücreler üzerinden noisy→clean vs pred→clean MSE oranının \(10 \log_{10}\) değeri.

### 7) DataLoader notları

`train.py`: `shuffle=True` yalnızca **dosya (utterance) sırasını** karıştırır; tek dosya içindeki chunk sırası korunur. `workers` genelde 0 (düşük RAM, Windows’ta basit debug). `drop_last=False`.

## Eğitimi çalıştırma

```text
cd model_development
..\.venv\Scripts\python.exe train.py
```

(Hedef: `HYPERPARAMS` içindeki `noisy_root` / `clean_root` yollarında eşleşmiş `.wav` dosyaları.)

Çıktı klasörü: `runs/<timestamp>_<run_tag>/` — içinde `run_config.json`, `model_info.txt`, `metrics_train.csv`, `loss_curve.png`, `snr_curve.png`, `best_weights.pt`, `last.pt`, vb.

## Demo (tek dosya WAV)

```text
cd model_development
python demo_single_pair_forward.py --help
```

İsteğe bağlı `--weights` ile eğitilmiş `state_dict` yüklenebilir; model boyutu `--hidden-dim` / `--gru-layers` ile eşleşmeli.

## Hiperparametre özeti (`train.py` — `HYPERPARAMS`)

- Veri: `noisy_root`, `clean_root`, `val_fraction`, `log_eps`
- Model: `hidden_dim`, `gru_num_layers`, `gru_dropout`
- Eğitim: `epochs`, `batch_size`, `lr`, `workers`, `seed`, `device`
- Çıktı: `out_dir` (None ise otomatik run klasörü), `run_tag` (otomatik doldurulur)

Donanım ipucu: **VRAM** çoğunlukla `batch_size`, `hidden_dim`, `T_max` (batch içi en uzun dosya) ile büyür; **RAM** çoğunlukla preload edilen toplam `(çift sayısı × T × F)` ile büyür.
