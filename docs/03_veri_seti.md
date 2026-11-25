# 3 · Veri Seti ve Test Düzeni

## 3.1 Amaç
Önceden eğitilmiş modeller kullandığımız için veri setinin amacı **eğitim** değil, **test ve doğrulama**dır:
1. Yöntemleri aynı koşullarda, **ground-truth'a (GT) karşı** nicel olarak (PSNR, SSIM, LPIPS) karşılaştırmak.
2. Her bozulma türünün etkisini ayrı gruplarda incelemek.
3. Gerçek eski fotoğraflarda niteliksel (görsel) değerlendirme yapmak.

## 3.2 Klasör düzeni

```
dataset/
├── manifest.json              # tüm eşleşmeler + bozulma zinciri + seed (izlenebilirlik)
├── ground_truth/              # 6 temiz HR görüntü (≤512 px, kenarlar 4'ün katı)
│   ├── astronaut.png  camera.png  chelsea.png  coffee.png  rocket.png  test_pattern.png
├── degraded/
│   ├── bicubic_x4/            # 6 × (GT / 4) — yalnızca bicubic
│   ├── realworld_x4/          # 6 × (GT / 4) — 2. derece gerçekçi bozulma
│   ├── realworld_x2/          # 6 × (GT / 2)
│   ├── old_photo/             # 6 × (GT / 2) — sepya, solma, vinyet, gren, çizik, toz, JPEG
│   └── grayscale/             # 4 × (renkli kaynakların S/B hali) — renklendirme testi
├── videos/
│   ├── gt/clip_clean.mp4      # 640×360, 24 fps, 6 sn, sesli
│   └── degraded/clip_old_film.mp4  # 320×180, S/B-sepya, gren, titreme, film çiziği, 180 kbit/s
├── real_old/                  # GT'siz gerçek eski fotoğraflar (kullanıcı ekler)
└── results/                   # evaluate.py çıktıları: metrics.csv/json, summary.md, grid_*.jpg
```

Toplam: **6 GT görüntü, 28 bozulmuş–GT çifti, 1 video çifti** (yaklaşık 4 MB, depoya dahil).

## 3.3 Kaynak görüntüler
scikit-image'ın örnek veri setinden alınmıştır. Lisansları serbesttir:
`astronaut`, `rocket` (NASA, kamu malı), `coffee`, `chelsea`, `camera` (CC0) ve prosedürel olarak üretilen
`test_pattern` (zone plate, ince çizgiler, metin, renk rampası; aliasing ve keskinlik testi için).
Daha büyük bir değerlendirme için **DIV2K validation** (100 görüntü) veya **RealSR** kullanılabilir:

```bash
python scripts/make_dataset.py --source /yol/DIV2K_valid_HR --max-side 768 --limit 30
```

## 3.4 Bozulma zincirleri
Her örnekteki zincirin tamamı `manifest.json` dosyasında ve web arayüzünün **Veri Seti** sekmesinde görünür.
Örnek:

```json
{
  "group": "realworld_x4", "name": "astronaut",
  "gt": "ground_truth/astronaut.png", "degraded": "degraded/realworld_x4/astronaut.png",
  "size_gt": "512x512", "size_lr": "128x128", "scale": 4.0,
  "degradations": ["aniso_blur(σx=1.83,σy=0.71,θ=122°)", "resize(bicubic,410x410)",
                   "gauss_noise(σ=11.2)", "jpeg(q=57)", "gauss_blur(σ=0.93)", "resize(area,...)",
                   "poisson_noise(ölçek=0.84)", "resize(bilinear,128x128)", "jpeg(q=44)"]
}
```

| Grup | Neyi test eder? |
|---|---|
| `bicubic_x4` | Klasik SR kıyas ortamı. SRCNN/EDSR'ın "en rahat" olduğu koşul |
| `realworld_x4/x2` | Kör SR ve Real-ESRGAN'ın asıl hedeflediği koşul |
| `old_photo` | Eski fotoğraf boru hattının tamamı (çizik giderme, renk, SR) |
| `grayscale` | Renklendirme (GT renkli orijinaldir; renk zenginliği ve görsel değerlendirme) |
| `videos` | Kare çıkarma, işleme, birleştirme ve **ses–görüntü senkronu** |

**Senkron testi:** Test videosunda her saniyenin ilk 2 karesinde beyaz çerçeve flaşı ve aynı anda 80 ms süren
1 kHz bip vardır. İşlenmiş videoda flaş ile bip hâlâ çakışıyorsa senkron korunmuştur. Otomatik test
(`tests/test_pipeline.py::test_api_video_job_keeps_audio_and_fps`) fps, süre ve ses akışının korunduğunu doğrular.

## 3.5 Tekrar üretilebilirlik
Tüm rastgele bozulmalar `--seed 2026` ile belirlenir. Aynı komut her zaman aynı veri setini üretir.

## 3.6 Değerlendirme protokolü
- PSNR ve SSIM, **Y (luma) kanalında**, kenarlardan 4 piksel kırpılarak hesaplanır (SR literatüründeki yaygın protokol).
- Yöntem çıktısı GT boyutuna eşitlenir (gerekirse bicubic ile).
- LPIPS (AlexNet) için `pip install lpips` gerekir; kurulu değilse sütun boş kalır.
- GT olmayan örnekler için referanssız ölçüler kullanılır: Laplacian varyansı (keskinlik), Immerkær gürültü σ,
  Hasler–Süsstrunk renk zenginliği.

```bash
python scripts/evaluate.py                    # tüm gruplar ve uygun tüm yöntemler
python scripts/evaluate.py --groups realworld_x4 --methods bicubic realesr-general-x4v3 RealESRGAN_x4plus
```
