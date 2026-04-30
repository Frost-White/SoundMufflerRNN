# Frontend Plan (PRD 5.1 - Web Demo Sitesi)

## Hedef
- Kullaniciya gürültülü ses dosyasini yukleyip temizlenmis sonuc ile karsilastirma deneyimi sunan bir web arayuzu hazirlamak.
- Sadece frontend kapsami; backend entegrasyonu minimum seviyede ve API sozlesmesi varsayimi ile ele alinacak.

## Teknoloji Karari
- Build araci: Vite
- UI: React (Vite + React kurulumu)
- Ses gorsellestirme: WaveSurfer.js
- Tema: Koyu tema (dark mode) varsayilan olacak

## Yapilacaklar Listesi

### 1) Proje Kurulumu
- Vite + React projesi baslat.
- Temel klasor yapisini olustur (`components`, `pages`, `services`, `assets`, `styles`).
- Ortam degiskeni ile API base URL tanimi yap (`.env` uzerinden).

### 2) Ana Ekran Akisi
- Tek sayfada demo akisina uygun yerlesim tasarla:
  - Dosya yukleme alani
  - Islem durumu/progress alani
  - Orijinal vs temizlenmis ses karsilastirma alani
  - Indirme butonu

### 3) Ses Dosyasi Yukleme
- Surukle-birak ve dosya secici destekli upload bileseni gelistir.
- Desteklenen formatlar ve dosya boyutu icin istemci tarafi dogrulama ekle.
- Gecersiz dosya durumlari icin anlasilir hata mesajlari goster.

### 4) API Cagrisi ve Progress
- Yuklenen dosyayi backend API'ye gonderen bir servis katmani yaz.
- Islem surecinde loading/progress durumunu UI'da goster.
- Basarili/hatali cevaplarda kullanici geri bildirimi sagla.

### 5) Ses Karsilastirma Deneyimi
- Orijinal ve temizlenmis ses icin iki ayri player sun.
- Ayni ekranda A/B karsilastirmayi kolaylastiran kontrol yapisi ekle.
- Oynatma kontrollerini (play/pause, seek) net ve sade tut.

### 6) Waveform Gorsellestirmesi
- WaveSurfer.js ile her iki ses icin waveform goster.
- Waveform yukleme ve render hatalarinda fallback mesaji ver.
- Performans icin gereksiz yeniden renderlari azalt.

### 7) Temizlenmis Ses Indirme
- Temizlenmis ses URL'inden dosya indirme aksiyonunu ekle.
- Indirme durumunda buton state'lerini (aktif/pasif) dogru yonet.

### 8) Kayit Formu (Backend'den Bagimsiz)
- Ayrı bir kayit sayfasi olustur (isim, e-posta, sifre, sifre tekrar gibi alanlar).
- Form dogrulamalarini frontend tarafinda yap (zorunlu alan, e-posta formati, sifre kurali, sifre eslesmesi).
- Gercek kayit istegi atmadan, submit akisini mock/placeholder davranis ile tamamla.

### 9) Hesap Ayarlari Sayfasi (Backend'den Bagimsiz)
- Profil bilgileri icin temel ayarlar formu olustur (ad, e-posta, sifre degistirme alanlari).
- Kaydet/iptal butonlari ve form durumlarini (dirty, loading, disabled) UI seviyesinde yonet.
- Backend entegrasyonu olmadan calisacak sekilde mock basari/hata geri bildirimi goster.

### 10) API Key Yonetim Alani (Hesap Ayarlari Icinde)
- Hesap ayarlari altinda API key yonetimi bolumu ekle.
- API key olustur, listele, kopyala, gizle/goster, iptal et (revoke) aksiyonlari icin UI akisini tasarla.
- Guvenlik odakli gorunum detaylari ekle (kismi maskeleme, son kullanma tarihi/olusturma tarihi alani, onay dialogu).
- Bu adimda yalnizca arayuz ve state yonetimi; gercek key islemleri backend baglantisi olmadan mocklanacak.

### 11) UI/UX ve Erisilebilirlik
- Basit, hizli, mobil uyumlu ve koyu tema odakli bir arayuz olustur.
- Klavye ile erisilebilir temel kontrolleri sagla.
- Bos durum, hata durumu, yukleniyor durumu tasarimlarini tamamla.

### 12) Test ve Teslim Hazirligi
- Kritik akislar icin temel component/integration testleri yaz.
- Manuel test checklisti hazirla:
  - Dosya yukleme
  - API basarili/basarisiz donusleri
  - Karsilastirmali oynatma
  - Waveform goruntuleme
  - Indirme
  - Kayit formu dogrulama
  - Hesap ayarlari form durumlari
  - API key yonetim UI akislari
- Vite production build alinip demo dagitimina hazir hale getir.

## Backend Notu (Minimum)
- Bu dokumanda backend detayina girilmeyecek.
- Frontend tarafinda sadece su sozlesme varsayilacak:
  - `POST /enhance` benzeri bir endpoint dosyayi alip isler.
  - Donuste temizlenmis ses kaynagi (URL veya blob) saglanir.
