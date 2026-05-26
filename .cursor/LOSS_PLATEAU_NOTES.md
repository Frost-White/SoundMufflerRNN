# Loss Plateau Notlari

## Ana Sorun
- Kombine loss ile egitim erken plato oluyor.

## Muhtemel Nedenler
- MSS tarafi `torch.stft(center=True)` kullanirken ana ozellik hatti librosa `center=False` kullaniyor; temsil uyumsuzlugu gradient catismasi uretebilir.
- Agirliklar MSS'i baskin yapiyor (`mse_weight=0.5`, `mss_weight=1.0`).
- `Sigmoid` maske cikisi 1.0 ustu gain veremiyor; modelin ulasabilecegi cozum uzayi kisitli.

## Uygulanacak Siralama
1. Hemen ablation yap:
   - Kosu A: `mss_weight=0.0`
   - Kosu B: `mss_weight=0.1`
   - Ilk birkac epoch'ta `train_mse` ve `val_mse` egimini karsilastir.
2. MSS-STFT hizala:
   - MSS hesaplamasinda `center=False` dene.
   - Ayni seed ile karsilastir.
3. Loss agirliklarini dengele:
   - Baslangic: `mse_weight=1.0`, `mss_weight=0.05~0.2`
4. Gerekirse cikis basligini esnet:
   - Sigmoid mask yerine pozitif gain (ornegin softplus tabanli) A/B dene.
5. Opsiyonel stabilite:
   - Log-mag girisine basit normalizasyon (per-utt mean/std) dene.

## Basari Kriteri
- Ilk 3-5 epoch'ta net dusus trendi.
- `val_mse` kotulesmeden iyilesme.
- Sonraki degerlendirmede PESQ/STOI/SI-SDR en az baseline kadar iyi.

## Referans Dosyalar
- `model_development/train.py`
- `model_development/training_loop.py`
- `model_development/audio_pipeline.py`
- `model_development/model.py`
- `model_development/training_data.py`
