# RetroRestore — Eski Fotoğraf ve Video İyileştirme

**Görüntü İşleme dersi dönem projesi** · Real-ESRGAN tabanlı süper çözünürlük, gürültü giderme, renk restorasyonu ve renklendirme, kare tabanlı video iyileştirme ve tüm adımları şeffaf biçimde gösteren bir web arayüzü.

**Depo:** https://github.com/aysecetin0/retro-restore

### ▶ Kurulum yapmadan hemen dene (Google Colab, ücretsiz GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aysecetin0/retro-restore/blob/main/RetroRestore_Colab.ipynb)

Linke tıklayın → **Çalışma zamanı → Tümünü çalıştır** → 1. bölümde çıkan linkten web sitesini açın. Not defterinin geri kalanı veri setini, arka plan akışını, yöntem karşılaştırmasını ve DIV2K üzerinde büyük ölçekli testi gösterir.

| | |
|---|---|
| **Görüntü** | Gürültü ve JPEG artefaktı giderme, detay netleştirme, 1–4× süper çözünürlük, çizik/toz giderme |
| **Renk** | Soluk renkli fotoğrafta renk canlandırma; siyah-beyaz fotoğrafta otomatik renklendirme |
| **Video** | Karelere ayırma, kare kare iyileştirme, H.264 ile yeniden birleştirme, **ses senkronu korunur** |
| **Arayüz** | Son kullanıcı için sade web sitesi: yükle → ayarla → önce/sonra karşılaştır → indir |
| **Sunum** | Colab/Jupyter not defteri: veri seti, arka plan akışı, yöntem karşılaştırma, DIV2K üzerinde büyük ölçekli test |
| **Donanım** | CUDA (FP16), Apple MPS veya CPU otomatik seçilir; PyTorch yoksa bile klasik modda çalışır |

---

## İçindekiler
1. [Hızlı başlangıç](#1-hızlı-başlangıç)
2. [Kullanım](#2-kullanım)
3. [Boru hattı (pipeline)](#3-boru-hattı-pipeline)
4. [Yöntem karşılaştırması ve neden Real-ESRGAN?](#4-yöntem-karşılaştırması-ve-neden-real-esrgan)
5. [Veri seti ve değerlendirme](#5-veri-seti-ve-değerlendirme)
6. [Proje yapısı](#6-proje-yapısı)
7. [Donanım ve performans](#7-donanım-ve-performans)
8. [API](#8-api)
9. [Sorun giderme](#9-sorun-giderme)
10. [Kaynaklar ve lisans](#10-kaynaklar-ve-lisans)

---

## 1. Hızlı başlangıç

### Gereksinimler
- Python **3.10+**
- **FFmpeg** (video için). Windows: `winget install ffmpeg` · macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg`
- İsteğe bağlı: NVIDIA GPU + CUDA

### Kurulum
```bash
git clone https://github.com/aysecetin0/retro-restore.git
cd retro-restore

python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate

# GPU kullanacaksanız ÖNCE CUDA'lı PyTorch'u kurun (https://pytorch.org), örnek:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/download_models.py        # Real-ESRGAN + renklendirme ağırlıkları (~270 MB)
python run.py                            # → http://127.0.0.1:8000
```

> Veri seti (`dataset/`) depoya dahildir. Yeniden üretmek için `python scripts/make_dataset.py` komutunu çalıştırın.
> Testler için: `python -m pytest -q` (16 test; model ağırlığı gerektirmez).

---

## 2. Kullanım

### Web sitesi (son kullanıcı)
1. Fotoğraf veya video sürükleyin.
2. Ayarları seçin: **büyütme oranı** (1–4×), **gürültü temizleme**, **renk** (otomatik / canlandır / renklendir), **çizik ve toz temizleme**.
   *Gelişmiş* bölümünde yöntem (Real-ESRGAN, klasik, Lanczos, Bicubic), model, keskinleştirme ve çıktı biçimi seçilebilir.
3. **İyileştir** düğmesine basın, önce/sonra kaydırıcısıyla (video için yan yana oynatıcıyla) karşılaştırın, **Sonucu indir** ile kaydedin.

### Sunum / teknik inceleme not defteri (`RetroRestore_Colab.ipynb`)
Veri seti, arka plan akışı, yöntem karşılaştırması ve büyük ölçekli test, projenin kendi Python kodu çalıştırılarak bu not defterinde gösterilir:

| Bölüm | Gösterilen |
|---|---|
| 1 | Web sitesini başlatır ve linkini verir |
| 2 | Veri seti: gruplar, sayılar, GT ↔ bozulmuş örnekler ve bozulma zincirleri |
| 3 | Arka plan akışı: bir fotoğrafın her adımı (süre, parametre, ara görüntü) |
| 4 | Real-ESRGAN içeride: mimari, parametre sayıları, DNI ile gürültü ayarı |
| 5 | Yöntem karşılaştırma: Bicubic, Lanczos, Klasik, FSRCNN, EDSR, Real-ESRGAN (PSNR/SSIM/LPIPS + görsel ızgara) |
| 6 | Video boru hattı: kare çıkarma → işleme → ses senkronlu birleştirme |
| 7 | **DIV2K üzerinde büyük ölçekli test:** 100 görüntü × 4 bozulma grubu = 400 test çifti, tablo ve grafikler |
| 8 | Sonuçları (CSV, Markdown tablo, ızgaralar) zip olarak indirme |

### Komut satırı
```bash
python enhance_cli.py dataset/degraded/old_photo/chelsea.png -o out.png --scale 2 --scratch --trace
python enhance_cli.py dataset/videos/degraded/clip_old_film.mp4 -o film.mp4 --scale 2 --preview-seconds 3
python enhance_cli.py klasorum/ -o cikti/ --model RealESRGAN_x4plus --scale 4
```

---

## 3. Boru hattı (pipeline)

```
GÖRÜNTÜ
 girdi ─► [1] Ön işleme ─► [2] Çizik/toz* ─► [3] Ön denoise* ─► [4] Süper çözünürlük ─► [5] Renk ─► [6] Son işleme ─► çıktı
          doğrulama, alfa,   top-hat +        NLM              Real-ESRGAN (döşemeli,   restorasyon   unsharp*,
          piksel limiti,     Telea                             FP16, OOM korumalı)      veya          alfa, kayıt
          S/B ve gürültü     inpainting                        | Klasik | Lanczos        renklendirme
          analizi                                                                                              (* isteğe bağlı)
VİDEO
 video ─► ffprobe ─► ffmpeg CFR kare çıkarma (PNG) ─► her kare = görüntü boru hattı ─► ffmpeg H.264 + orijinal ses ─► mp4
                                                     (renk kararı ve renk geçmişi
                                                      kareler arasında taşınır)
```

Her adım `trace.json` dosyasına ve arayüze kaydedilir. Ayrıntılar: [`docs/04_arka_plan_akisi.md`](docs/04_arka_plan_akisi.md) ·
Mimari: [`docs/01_mimari.md`](docs/01_mimari.md)

---

## 4. Yöntem karşılaştırması ve neden Real-ESRGAN?

| Yöntem | Gerçek bozulmaya dayanıklılık | Maliyet | Artı | Eksi |
|---|---|---|---|---|
| Bicubic / Lanczos | ✗ | ≈0 | Hızlı, standart | Bulanık; gürültü ve JPEG büyür |
| Klasik (NLM + USM) | △ (yalnızca gürültü) | Düşük | Açıklanabilir | Detay ile gürültüyü birlikte siler |
| SRCNN / FSRCNN | ✗ | Düşük | İlk derin SR | Bicubic bozulmayla eğitilmiş, bulanık |
| EDSR | ✗ | Yüksek | Yüksek PSNR | Gerçek gürültüyü keskinleştirir |
| ESRGAN | △ | Yüksek | Keskin dokular | Basit bozulma modeli, artefakt |
| **Real-ESRGAN** | **✓✓** | Orta (compact: düşük) | Kör restorasyon, tek geçiş, DNI ile ayarlanabilir | Ağır hasarda doku uydurabilir |

**Neden Real-ESRGAN?** Eski fotoğraf ve videolardaki bozulma *bilinmez ve karışıktır*. Real-ESRGAN, eğitimde iki tur
`bulanıklık → yeniden boyutlandırma → gürültü → JPEG` (ve sinc) uygulayan **yüksek dereceli bozulma sentezi** ile
eğitildiği için gürültüyü, sıkıştırma artefaktlarını ve bulanıklığı **tek bir modelle** giderirken çözünürlüğü artırır.
U-Net + spectral norm ayırt edici yerel doku kalitesini yükseltir. Hafif `general-x4v3` modeli CPU ve video için
pratiktir. Ayrıca DNI sayesinde gürültü giderme gücü kullanıcı tarafından ayarlanabilir.

Tam tablo, renklendirme yöntemleri, sunum notları ve "PSNR neden düşük çıkabilir?" sorusunun yanıtı:
[`docs/02_yontem_karsilastirma.md`](docs/02_yontem_karsilastirma.md)

---

## 5. Veri seti ve değerlendirme

```
dataset/
├── manifest.json        eşleşmeler + bozulma zinciri + seed
├── ground_truth/        6 temiz HR görüntü
├── degraded/
│   ├── bicubic_x4/      yalnızca bicubic küçültme (klasik SR kıyası)
│   ├── realworld_x4/    Real-ESRGAN tarzı 2. derece bozulma
│   ├── realworld_x2/
│   ├── old_photo/       sepya, solma, vinyet, gren, çizik, toz, JPEG
│   └── grayscale/       renklendirme testi
├── videos/{gt,degraded} 6 sn sesli klip (saniyede bir flaş + bip senkron işaretçisi)
├── real_old/            gerçek eski fotoğraflarınız (GT yok)
└── results/             evaluate.py çıktıları
```

```bash
python scripts/make_dataset.py                       # dahili örneklerle (seed=2026)
python scripts/make_dataset.py --source DIV2K_valid_HR --limit 30
python scripts/evaluate.py                           # → dataset/results/{metrics.csv, metrics.json, summary.md, grid_*.jpg}

# Büyük ölçekli test (DIV2K doğrulama seti, 100 görüntü → 400 test çifti)
python scripts/download_benchmarks.py
python scripts/make_dataset.py --source benchmarks/DIV2K_valid_HR --prefix div2k_ --append --no-video --max-side 768 --limit 100
python scripts/evaluate.py --prefix div2k_
```

- PSNR ve SSIM: Y kanalında, kenarlardan 4 piksel kırpılarak hesaplanır · LPIPS: `pip install lpips` ile
- GT olmayan örnekler: keskinlik (Laplacian varyansı), gürültü σ (Immerkær), renk zenginliği

> **Not:** `dataset/results/` klasörü boş gelir. Sonuçlar, modeller indirildikten sonra **kendi donanımınızda**
> `scripts/evaluate.py` çalıştırılarak üretilir. Böylece rapordaki sayılar ile ölçülen değerler birebir aynı olur.

Ayrıntılar: [`docs/03_veri_seti.md`](docs/03_veri_seti.md)

---

## 6. Proje yapısı

```
retro-restore/
├── README.md                     ← bu dosya
├── LICENSE
├── requirements.txt
├── run.py                        ← web sunucusunu başlatır
├── RetroRestore_Colab.ipynb      ← sunum + canlı demo + büyük ölçekli test (Colab/Jupyter)
├── enhance_cli.py                ← komut satırı arayüzü
├── retrorestore/
│   ├── config.py                 ← tüm limitler ve ayarlar (RR_* ortam değişkenleri)
│   ├── models_registry.py        ← model adları, dosyalar, indirme adresleri
│   ├── core/
│   │   ├── device.py             ← CUDA/MPS/CPU seçimi, FP16, otomatik döşeme boyutu
│   │   ├── trace.py              ← şeffaflık: adım kaydı + ara görüntüler
│   │   └── metrics.py            ← PSNR, SSIM, LPIPS, referanssız ölçüler
│   ├── methods/
│   │   ├── classical.py          ← interpolasyon, NLM, bilateral, unsharp, renk restorasyonu
│   │   ├── old_photo.py          ← çizik/toz tespiti + inpainting
│   │   ├── realesrgan_arch.py    ← RRDBNet + SRVGGNetCompact (saf PyTorch)
│   │   ├── realesrgan.py         ← döşemeli çıkarım, FP16, OOM kurtarma, DNI, model önbelleği
│   │   ├── colorize.py           ← Zhang 2016 renklendirme (OpenCV DNN) + zamansal EMA
│   │   └── opencv_sr.py          ← EDSR/FSRCNN karşılaştırma taban çizgileri
│   ├── pipeline/
│   │   ├── image_pipeline.py     ← görüntü boru hattı + EnhanceOptions
│   │   ├── video_pipeline.py     ← ffprobe → kareler → işleme → H.264 + ses
│   │   └── benchmark.py          ← çok yöntemli karşılaştırma + ızgara
│   ├── utils/io.py               ← Unicode-güvenli okuma/yazma, EXIF
│   └── web/
│       ├── app.py                ← Flask REST API
│       ├── jobs.py               ← iş kuyruğu (tek işçi), iptal, TTL temizlik
│       └── static/               ← kullanıcı sitesi: index.html, css/style.css, js/app.js
├── scripts/
│   ├── download_models.py        ← ağırlık indirici (yedek adresli)
│   ├── download_benchmarks.py    ← DIV2K doğrulama seti indirici
│   ├── make_dataset.py           ← seed'li test veri seti üretici
│   └── evaluate.py               ← toplu nicel değerlendirme
├── dataset/                      ← test/doğrulama veri seti (bkz. bölüm 5)
├── models/                       ← indirilen ağırlıklar (git'e eklenmez)
├── outputs/jobs/<id>/            ← iş klasörleri: input, result, steps/, trace.json, job.json
├── docs/
│   ├── 01_mimari.md
│   ├── 02_yontem_karsilastirma.md
│   ├── 03_veri_seti.md
│   ├── 04_arka_plan_akisi.md
│   └── rapor_kapagi.md           ← rapor kapağı ve proje künyesi
└── tests/test_pipeline.py        ← birim, API, güvenlik ve video senkronu testleri
```

---

## 7. Donanım ve performans

| Önlem | Açıklama |
|---|---|
| Otomatik cihaz | CUDA → MPS → CPU. `RR_DEVICE=cpu` ile zorlanabilir |
| FP16 | Yalnızca CUDA'da etkindir (yaklaşık 2 kat hız, yarı VRAM). `RR_FP16=0` ile kapatılabilir |
| Döşemeli çıkarım | Döşeme boyutu VRAM'e göre seçilir (≥12 GB: döşemesiz, 6 GB: 512, 3 GB: 384, CPU: 256). `RR_TILE=...` ile değiştirilebilir |
| OOM kurtarma | Bellek yetmezse döşeme boyutu yarıya indirilir ve iş yeniden denenir |
| Tek işçi kuyruğu | Aynı anda tek ağır iş çalışır, diğerleri sırada bekler |
| Limitler | 6 MP girdi, 40 MP çıktı, 120 sn / 3600 kare video, 1920 px video kenarı (`RR_MAX_*`) |
| Video önizleme | "İlk N saniye" ile uzun videoda önce hızlı deneme yapılabilir |
| Fallback | PyTorch veya ağırlık yoksa klasik yöntem kullanılır ve arayüzde turuncu uyarı gösterilir |

**İpucu:** CPU'da video için `realesr-general-x4v3`, ölçek 2× ve önizleme 3–5 sn ile başlayın.

---

## 8. API

| Metot | Yol | Açıklama |
|---|---|---|
| GET | `/api/system` | Cihaz, modeller, limitler |
| GET | `/api/methods` | Karşılaştırılabilir yöntemler |
| GET | `/api/dataset` | Veri seti manifestosu |
| GET | `/api/report` | `evaluate.py` sonuçları |
| POST | `/api/jobs` | `file` **veya** `sample` + parametreler (`kind=enhance|benchmark`) |
| GET | `/api/jobs/<id>` | Durum, ilerleme, trace |
| POST | `/api/jobs/<id>/cancel` | İptal |
| GET | `/files/jobs/<id>/<dosya>?download=1` | Sonuç ve ara dosyalar |

```bash
curl -F file=@eski.jpg -F scale=2 -F color_mode=auto http://127.0.0.1:8000/api/jobs
```

---

## 9. Sorun giderme

| Belirti | Çözüm |
|---|---|
| "PyTorch kurulu değil → klasik yönteme düşüldü" | `pip install torch` (GPU için pytorch.org'daki CUDA komutu) |
| "Ağırlık bulunamadı" | `python scripts/download_models.py`. Olmazsa dosyayı `models/README.md` tablosundan elle indirin |
| "Renklendirme modeli bulunamadı" | `python scripts/download_models.py --colorization` |
| "FFmpeg bulunamadı" | FFmpeg'i kurun ve `ffmpeg -version` komutunun çalıştığını doğrulayın |
| CUDA out of memory | Otomatik kurtarma devreye girer. Sürerse `RR_TILE=128` kullanın veya ölçeği düşürün |
| Video çok yavaş | Önizleme süresi, 2× ölçek ve `realesr-general-x4v3` kullanın |

---

## 10. Kaynaklar ve lisans
- X. Wang et al., *Real-ESRGAN*, ICCVW 2021 — https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)
- X. Wang et al., *ESRGAN*, ECCVW 2018 · B. Lim et al., *EDSR*, CVPRW 2017 · C. Dong et al., *SRCNN*, ECCV 2014
- R. Zhang et al., *Colorful Image Colorization*, ECCV 2016 — https://github.com/richzhang/colorization
- Tam kaynak listesi: [`docs/02_yontem_karsilastirma.md`](docs/02_yontem_karsilastirma.md#24-kaynaklar)

Proje kodu **MIT** lisanslıdır. Model ağırlıkları kendi lisanslarına tabidir.
Rapor kapağı taslağı: [`docs/rapor_kapagi.md`](docs/rapor_kapagi.md)
