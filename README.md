# SoundMufflerRNN

SoundMufflerRNN, gürültülü konuşma kaydını iyileştirmek için eğitilmiş bir GRU tabanlı maskeleme modeline sahip bir ses geliştirme uygulamasıdır. Proje aşağıdaki dört ana bölümden oluşur:

- **Model Development**: eğitim, değerlendirme ve single-file demo pipeline
- **Backend**: FastAPI tabanlı API, kullanıcı/anahtar yönetimi, rate limit ve inference
- **Frontend**: React + Vite tabanlı arayüz, ses yükleme ve dinleme
- **Deployment**: Docker Compose ile lokal servis orkestrasyonu

---

## 1. Model Development

Bu proje, `model_development/` klasöründe yer alan bir eğitim ve değerlendirme hattına sahiptir.

### Özet
- Model: `GRUChunkDenoiser`
- Mimari: STFT tabanlı zaman-chunk maskeleme
- Girdi: `log(|STFT| + eps)` magnitüd özellikleri
- Çıktı: sigmoid maske ile filtrelenmiş magnitüd
- Inference: faz koruma + STFT tutarlılığı blend

### Güncel Model Performansı
- Ortalama `SI-SDR`: **14.40 dB**
- Ortalama SNR kazancı: **6.12 dB**
- Ortalama `PESQ`: **2.10**
- Ortalama `STOI`: **0.915**
- Değerlendirme seti: **824 dosya**

Bu skorlar proje içindeki `model_development/eval_outputs/20260525_163456_20260525_015037_all1_resume_resume/eval_summary.json` dosyasından alınmıştır.

### Nerede
- `model_development/train.py`: eğitim döngüsü, hiperparametreler, veri hazırlığı
- `model_development/eval.py`: eşleşmiş noisy/clean dosyalar için toplu değerlendirme
- `model_development/eval_one.py`: tek dosya değerlendirme
- `model_development/demo_single_pair_forward.py`: tek noisy WAV girişinden iyileştirilmiş çıktı üretme
- `model_development/model.py`: GRU model tanımı
- `model_development/audio_pipeline.py`: STFT/ISTFT, chunk'lama, overlap-add

### Çalıştırma
```bash
cd model_development
python train.py
```

### Değerlendirme
```bash
cd model_development
python eval.py
```

### Tek dosya demo
```bash
cd model_development
python demo_single_pair_forward.py --help
```

---

## 2. Backend

Backend, `backend/` klasöründe yer alan FastAPI uygulamasıdır. API, kullanıcı kimlik doğrulaması, anahtar yönetimi ve ses iyileştirme uç noktalarını içerir.

### Öne çıkan özellikler
- `FastAPI` tabanlı servis
- PostgreSQL veritabanı (`docker-compose.yml` ile `postgres:16-alpine`)
- JWT tabanlı kimlik doğrulama
- API anahtarı ve web kullanıcıları için ayrı rate limitler
- Maksimum yükleme boyutu: **2 MB**
- Model ağırlıkları varsayılan olarak `backend/app/inference/assets/best_weights.pt` içinden yüklenir
- İsteğe bağlı olarak `SOUNDMUFFLER_WEIGHTS_PATH` ile özel ağırlık yolu belirlenebilir

### Önemli rotalar
- `GET /api/health`: sağlık kontrolü
- `POST /enhance`: API anahtarı ile iyileştirme
- `POST /enhance/web`: web kullanıcısı için iyileştirme
- `GET /`: servis temel bilgisi

### Docker / çalışma
Backend container içinde başlatılırken:
- `backend/Dockerfile` PyTorch CPU paketi ile derlenir
- `backend/docker-entrypoint.sh` veritabanını hazırlar ve `alembic upgrade head` çalıştırır

### Gerekli ortam değişkenleri
`docker-compose.yml` içinde tanımlı varsayılanlar:
- `POSTGRES_PASSWORD`
- `JWT_SECRET`
- `CORS_ORIGINS`
- `ENHANCE_WEB_RATE_LIMIT`
- `ENHANCE_WEB_RATE_WINDOW_SECONDS`
- `ENHANCE_API_FREE_RATE_LIMIT`
- `ENHANCE_API_FREE_RATE_WINDOW_SECONDS`
- `ENHANCE_API_PRO_RATE_LIMIT`
- `ENHANCE_API_PRO_RATE_WINDOW_SECONDS`
- `ENHANCE_MAX_UPLOAD_BYTES`

---

## 3. Frontend

Frontend, `frontend/app` içinde bir React + Vite uygulamasıdır.

### Özellikler
- Ses dosyası yükleme ve oynatma
- Kullanıcı kayıt/giriş akışları
- Hesap bakiyesi / anahtar yönetimi
- Backend `POST /enhance/web` ve `POST /enhance` uç noktalarına bağlanma

### Hızlı başlangıç
```bash
cd frontend/app
npm install
npm run dev
```

Yerel geliştirme için tarayıcınızda genellikle `http://localhost:5173` açılmalıdır.

### Üretim yapısı
```bash
cd frontend/app
npm run build
```

### Test
```bash
cd frontend/app
npm run test
```

---

## 4. Deployment

Proje, `docker-compose.yml` ile üç servis halinde çalışır:

- `db`: PostgreSQL
- `backend`: FastAPI uygulaması
- `frontend`: Nginx üzerinden servis edilen statik React uygulaması

### Başlatma
```bash
docker-compose up --build
```

### Erişim
- Frontend: `http://localhost`
- Backend sağlık: `http://127.0.0.1:8000/api/health`

### Notlar
- Backend yalnızca localhost üzerinden `127.0.0.1:8000` dinler. Frontend dış trafiği `80` üzerinden sunar.
- Eğer model ağırlıklarını proje dışında tutmak istiyorsanız, `SOUNDMUFFLER_WEIGHTS_PATH` environment variable ile `backend` container içinde yükleme yolunu gösterebilirsiniz.
- Veritabanı verisi `postgres_data` adında bir Docker hacminde saklanır.

---

## Ek Bilgiler

- `run-tests-report.ps1`: test raporu oluşturma
- `backend/pyproject.toml`: backend paket tanımı
- `backend/requirements.txt`: backend bağımlılıkları
- `frontend/app/package.json`: frontend bağımlılıkları
- `model_development/eval_outputs/`: model değerlendirme çıktıları

Bu README, proje yapısını ve çalıştırma adımlarını daha kapsamlı şekilde açıklar. Model geliştirme bölümündeki metrikler, mevcut değerlendirme özetine göre güncellenmiştir.
