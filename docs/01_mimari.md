# 1 · Mimari

## 1.1 Genel görünüm

```
┌──────────────────────────── Tarayıcı (frontend) ────────────────────────────┐
│  İyileştir · Yöntem Karşılaştırma · Veri Seti · Arka Plan & Mimari · İşler  │
│  HTML + CSS + saf JavaScript (derleme yok) — /retrorestore/web/static        │
└───────────────▲───────────────────────────────────────────────┬─────────────┘
                │  JSON (durum, ilerleme, trace)                 │ multipart (dosya + parametreler)
┌───────────────┴───────────────── Flask REST API ───────────────▼─────────────┐
│  retrorestore/web/app.py   — uç noktalar, doğrulama, güvenli dosya sunumu    │
│  retrorestore/web/jobs.py  — iş kuyruğu (TEK işçi thread), iptal, TTL temizlik│
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼────────────────── Boru hattı (pipeline) ─────────────────────┐
│ image_pipeline.py   ön işleme → çizik → ön denoise → SR → renk → son işleme  │
│ video_pipeline.py   ffprobe → kare çıkarma → (kare kare image_pipeline) → mux │
│ benchmark.py        aynı girdide çok yöntem + PSNR/SSIM/LPIPS + ızgara       │
└───────┬───────────────────┬───────────────────┬──────────────────────────────┘
        │                   │                   │
┌───────▼──────┐  ┌─────────▼────────┐  ┌───────▼─────────┐  ┌──────────────────┐
│ classical.py │  │ realesrgan.py    │  │ colorize.py     │  │ core/            │
│ interpolasyon│  │ + _arch.py       │  │ Zhang 2016      │  │ device.py  CPU/GPU│
│ NLM, CLAHE,  │  │ döşemeli çıkarım │  │ (OpenCV DNN)    │  │ trace.py  şeffaflık│
│ beyaz dengesi│  │ FP16, OOM, DNI   │  │ zamansal EMA    │  │ metrics.py PSNR.. │
│ old_photo.py │  │ opencv_sr.py     │  │                 │  │                  │
└──────────────┘  └──────────────────┘  └─────────────────┘  └──────────────────┘
```

## 1.2 Teknoloji seçimleri

| Katman | Seçim | Gerekçe |
|---|---|---|
| Dil | Python 3.10+ | Görüntü işleme ve derin öğrenme ekosistemi |
| Derin öğrenme | PyTorch (isteğe bağlı) | Real-ESRGAN'ın resmi çerçevesi. **Kurulu değilse uygulama klasik moda düşer** |
| Görüntü işleme | OpenCV (contrib) | Hızlı C++ çekirdekleri, NLM, inpainting, DNN (renklendirme, EDSR/FSRCNN) |
| Video | FFmpeg / ffprobe | Endüstri standardı. Kare çıkarma, H.264 kodlama, ses kopyalama |
| Web | Flask | Az bağımlılıklı ve anlaşılır. Ağır işler zaten arka plan thread'inde |
| Frontend | Saf HTML/CSS/JS | Node veya derleme adımı gerekmez, hoca doğrudan çalıştırabilir |
| Test | pytest | Boru hattı, API, güvenlik ve video senkronu testleri |

## 1.3 Donanım kısıtlarına karşı önlemler

| Risk | Önlem | Kod |
|---|---|---|
| GPU yok | CPU'da hafif `realesr-general-x4v3` otomatik seçilir; PyTorch yoksa klasik yöntem | `realesrgan.pick_model`, `image_pipeline` |
| VRAM taşması (OOM) | Döşemeli çıkarım. Döşeme boyutu boş VRAM'e göre seçilir; OOM olursa yarıya indirilip yeniden denenir (en fazla 4 kez) | `device.auto_tile_size`, `RealESRGANer.enhance` |
| Büyük çıktının VRAM'e sığmaması | Çıktı tensörü döşeme döşeme **CPU'da** birleştirilir | `_tile_process` |
| GPU yavaşlığı | CUDA'da FP16 (yarı hassasiyet): yaklaşık 2 kat hız, yarı bellek | `device.get_device_info` |
| Aynı anda birden çok ağır iş | **Tek işçi thread**, istekler sıraya girer | `web/jobs.py` |
| Aşırı büyük girdi | Girdi piksel limiti (6 MP) ve çıktı limiti (40 MP); aşılırsa küçültme veya ölçek düşürme yapılır ve kullanıcı uyarılır | `config.py`, `image_pipeline` |
| Uzun video | Süre (120 sn), kare (3600) ve çıktı kenarı (1920 px) limitleri; "ilk N saniye" önizleme modu | `video_pipeline` |
| Disk dolması | Kare klasörleri iş bitince silinir, iş klasörleri 24 saat sonra otomatik temizlenir | `jobs._janitor` |
| Model tekrar tekrar yüklenmesi | Model önbelleği (aynı anda tek model, bellek dostu) | `get_upsampler` |
| Güvenlik | Uzantı beyaz listesi, `secure_filename`, yol geçişi (path traversal) koruması, yükleme boyutu limiti | `web/app.py` |

Tüm limitler ortam değişkeniyle değiştirilebilir, örneğin `RR_MAX_VIDEO_SECONDS=300 RR_TILE=256 python run.py`
(bkz. `retrorestore/config.py`).

### Yaklaşık işlem süreleri (512×512 → 2048×2048, 4×)

| Donanım | general-x4v3 | x4plus |
|---|---|---|
| Orta seviye NVIDIA GPU (8 GB, FP16) | < 0.5 sn | 1–2 sn |
| Apple M serisi (MPS) | 1–3 sn | 5–10 sn |
| 4–8 çekirdekli CPU | 5–15 sn | 60–120 sn |

> Bu değerler yaklaşık tahminlerdir. Kendi donanımınızdaki gerçek süreler arayüzün "Arka plan akışı" panelinde ve
> `scripts/evaluate.py` çıktısında görünür. Video için kare başına süre × kare sayısı formülü kullanılır.
> Örneğin 10 sn, 24 fps, 2× bir klip CPU'da 240 kare × ~2 sn ≈ 8 dakika sürer. Bu yüzden önizleme modu vardır.

## 1.4 Dosya planı

Ayrıntılı ağaç için `README.md` dosyasındaki **Proje yapısı** bölümüne bakın.
