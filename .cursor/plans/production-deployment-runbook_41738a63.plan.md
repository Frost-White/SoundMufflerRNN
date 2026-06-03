---
name: production-deployment-runbook
overview: Kiralık VPS üzerinde SoundMufflerRNN uygulamasını mevcut Docker Compose yapısıyla canlıya almak için adım adım operasyon planı; yalnızca `.cursor` altına runbook dokümanı eklenecek, repoda kod/config değişikliği yapılmayacak.
todos:
  - id: write-runbook
    content: .cursor/production-deployment.plan.md runbook dosyasını yaz (sunucu kurulumu, .env, Caddy HTTPS, güvenlik, backup, checklist, troubleshooting)
    status: pending
isProject: false
---

# Production Deployment Runbook (.cursor)

## Hedef

Mevcut [docker-compose.yml](docker-compose.yml) stack'ini (PostgreSQL + FastAPI backend + nginx frontend) kiralık bir Linux VPS'e taşıyıp HTTPS ile dış dünyaya açmak. Repoda **yeni dosya eklenmeyecek**; tek deliverable: `.cursor/production-deployment.plan.md` runbook'u.

## Mevcut mimari (referans)

```mermaid
flowchart LR
  User["Browser"] -->|"HTTPS 443"| Caddy["Host Caddy"]
  Caddy -->|"localhost:8080"| Frontend["Docker frontend nginx"]
  Frontend -->|"/api /enhance"| Backend["Docker backend :8000"]
  Backend --> DB["Docker PostgreSQL"]
```

Bugünkü compose yapısı:
- [docker-compose.yml](docker-compose.yml): `db`, `backend` (`8000:8000`), `frontend` (`8080:80`)
- Backend startup: [backend/docker-entrypoint.sh](backend/docker-entrypoint.sh) — DB oluşturma + Alembic + uvicorn
- Frontend proxy: [frontend/app/nginx.conf](frontend/app/nginx.conf) — `/api/` ve `/enhance/` backend'e gider
- Env şablonu: [.env.example](.env.example)

## Runbook içeriği (yazılacak bölümler)

### 1. Sunucu gereksinimleri
- Minimum: **2 vCPU, 2–4 GB RAM, 20 GB disk**
- OS: Ubuntu 22.04/24.04 LTS
- Gerekçe: PyTorch CPU inference (~225 MB backend RAM idle+model), PostgreSQL volume, Docker overhead

### 2. Ön hazırlık checklist
- Domain satın al ve DNS **A kaydı** → sunucu public IP
- SSH erişimi (key-based auth önerisi)
- Git repo URL'si ve deploy branch'i

### 3. Sunucu ilk kurulum komutları
Runbook'ta kopyala-yapıştır blokları:
```bash
apt update && apt upgrade -y
apt install -y git curl ufw
curl -fsSL https://get.docker.com | sh
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw enable
```

### 4. Uygulama deploy adımları
```bash
git clone <repo-url> && cd SoundMufflerRNN
cp .env.example .env
# .env düzenle (aşağıdaki production değerleri)
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8080/api/health
```

**Production `.env` değerleri** (runbook'ta tablo):
| Değişken | Production değeri |
|----------|-------------------|
| `POSTGRES_PASSWORD` | `openssl rand -hex 32` ile üret |
| `JWT_SECRET` | `openssl rand -hex 32` ile üret |
| `CORS_ORIGINS` | `https://yourdomain.com,https://www.yourdomain.com` |

Not: `ENHANCE_*` rate limit env'leri şu an compose'da expose edilmemiş; runbook'ta sunucuda `docker-compose.yml` environment bloğuna manuel ekleme seçeneği belgelenecek (opsiyonel sıkılaştırma).

### 5. HTTPS — host üzerinde Caddy
Repoda dosya olmayacağı için runbook, sunucuda doğrudan oluşturulacak `/etc/caddy/Caddyfile` örneğini içerecek:

```
yourdomain.com {
    reverse_proxy 127.0.0.1:8080
}
```

Adımlar: Caddy kurulumu, servis başlatma, DNS propagation kontrolü, otomatik Let's Encrypt.

### 6. Port güvenliği (sunucuda manuel)
Runbook, production'da şu değişiklikleri **sunucudaki** `docker-compose.yml` üzerinde yapmayı tarif edecek (repo'ya commit gerekmez):
- `8000:8000` → kaldır veya `127.0.0.1:8000:8000` (backend dışarı kapalı)
- `8080:80` → `127.0.0.1:8080:80` (sadece Caddy erişsin)
- Her servise `restart: unless-stopped` ekle

### 7. Bilinen güvenlik notları
Runbook'ta mevcut koddan türetilmiş uyarılar:
- `/enhance` ve `/enhance/web` kimlik doğrulamasız ([backend/app/main.py](backend/app/main.py)) — CPU abuse riski
- `/enhance` rate limit (120/dk) > `/enhance/web` (30/dk) — doğrudan backend erişimi bypass edebilir; bu yüzden backend portu dışarı kapatılmalı
- `/api/auth`, `/api/billing`, `/api/keys` JWT korumalı — enhance ile ilgisi yok

### 8. Yedekleme ve bakım
- Günlük PostgreSQL dump cron örneği:
  `docker compose exec -T db pg_dump -U postgres soundmuffler > backup.sql`
- Volume: `postgres_data` — `docker compose down -v` **silme** uyarısı
- Güncelleme akışı: `git pull && docker compose up --build -d`
- Log izleme: `docker compose logs -f backend`

### 9. Doğrulama checklist (canlıya geçiş)
Runbook sonunda işaretlenebilir checklist:
- [ ] DNS A kaydı çözülüyor
- [ ] `.env` güçlü secret'lar
- [ ] `docker compose ps` — 3 servis Up
- [ ] `https://domain/api/health` → `{"status":"ok"}`
- [ ] Frontend demo sayfası açılıyor
- [ ] Backend 8000 dışarıdan erişilemiyor
- [ ] UFW sadece 22/80/443
- [ ] DB backup cron kurulu

### 10. Sorun giderme
Runbook'ta kısa bölüm:
- Backend `exec docker-entrypoint.sh: no such file or directory` → CRLF (Windows); rebuild
- CORS hatası → `CORS_ORIGINS` domain uyumsuz
- 502 Bad Gateway → frontend/backend container down; `docker compose logs`
- Enhance timeout → nginx `proxy_read_timeout 300s` zaten var; büyük dosyalar için `client_max_body_size 50M`

## Dosya konumu

Oluşturulacak tek dosya:

**[.cursor/production-deployment.plan.md](.cursor/production-deployment.plan.md)**

Format, mevcut [.cursor/stft-consistency-loss-upgrade_ba975d33.plan.md](.cursor/stft-consistency-loss-upgrade_ba975d33.plan.md) ile uyumlu YAML frontmatter + markdown gövde olacak (`name`, `overview`, `todos`, `isProject: false`).

## Kapsam dışı (bilinçli)

- Repoya `docker-compose.prod.yml`, Caddyfile veya deploy script eklenmeyecek (kullanıcı tercihi)
- CI/CD pipeline
- Managed DB / CDN / WAF
- Enhance endpoint auth implementasyonu
