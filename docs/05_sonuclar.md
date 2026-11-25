# 5 · Deney Sonuçları — DIV2K Üzerinde Büyük Ölçekli Değerlendirme

| | |
|---|---|
| **Veri seti** | DIV2K doğrulama seti (NTIRE 2017), **100 farklı görüntü** (uzun kenar 768 px'e indirgenmiş) |
| **Test çifti** | 100 görüntü × 4 bozulma grubu = **400 bozulmuş–GT çifti** |
| **Yöntem** | 7 yöntem → **2800 çıkarım** |
| **Donanım** | Google Colab, NVIDIA Tesla T4 (FP16) |
| **Tarih** | 25.11.2026 |
| **Metrikler** | PSNR / SSIM (Y kanalı, 4 px kenar kırpma), LPIPS-AlexNet (algısal; düşük = daha iyi) |
| **Tekrar üretme** | `RetroRestore_Colab.ipynb` bölüm 7 veya `python scripts/evaluate.py --prefix div2k_` (seed = 2026) |

Her grupta satırlar LPIPS'e göre sıralanmıştır. **Kalın** değerler o sütunun en iyisidir.

### bicubic_x4 — yalnızca bicubic 4× küçültme

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x4plus | 22.09 | 0.6476 | **0.2285** | 0.543 |
| Real-ESRGAN General v3 | 22.32 | 0.6712 | 0.2454 | 0.084 |
| FSRCNN (CNN) | **24.40** | **0.7189** | 0.2741 | 0.037 |
| Real-ESRGAN x2plus | 22.13 | 0.6219 | 0.3545 | 0.412 |
| Bicubic | 23.78 | 0.7075 | 0.4391 | 0.001 |
| Lanczos-4 | 23.53 | 0.6945 | 0.4528 | 0.009 |
| Klasik (NLM+Lanczos+USM) | 22.33 | 0.6678 | 0.4927 | 0.104 |

### realworld_x4 — gerçekçi 2. derece bozulma, 4×

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x4plus | 21.44 | 0.5666 | **0.2969** | 0.580 |
| Real-ESRGAN General v3 | 22.39 | **0.5916** | 0.3513 | 0.082 |
| Real-ESRGAN x2plus | 22.09 | 0.5823 | 0.3835 | 0.422 |
| FSRCNN (CNN) | 22.94 | 0.5842 | 0.5480 | 0.030 |
| Klasik (NLM+Lanczos+USM) | 22.80 | 0.5828 | 0.6016 | 0.104 |
| Bicubic | **23.00** | 0.5862 | 0.6776 | 0.001 |
| Lanczos-4 | **23.00** | 0.5860 | 0.6802 | 0.008 |

### realworld_x2 — gerçekçi 2. derece bozulma, 2×

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x2plus | 23.57 | 0.6692 | **0.2082** | 0.560 |
| Real-ESRGAN x4plus | 23.07 | 0.6583 | 0.2489 | 1.127 |
| Real-ESRGAN General v3 | **24.63** | **0.6854** | 0.3359 | 0.159 |
| Klasik (NLM+Lanczos+USM) | 23.84 | 0.6344 | 0.5084 | 0.329 |
| FSRCNN (CNN) | 24.08 | 0.6494 | 0.5267 | 0.156 |
| Lanczos-4 | 24.11 | 0.6457 | 0.5403 | 0.009 |
| Bicubic | 24.15 | 0.6478 | 0.5423 | 0.001 |

### old_photo — eski fotoğraf simülasyonu, 2×

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN General v3 | 14.49 | 0.4886 | **0.6801** | 0.142 |
| FSRCNN (CNN) | **14.72** | **0.4898** | 0.6942 | 0.136 |
| Real-ESRGAN x4plus | 14.53 | 0.4779 | 0.6988 | 1.099 |
| Real-ESRGAN x2plus | 14.59 | 0.4648 | 0.7075 | 0.546 |
| Bicubic | 14.66 | 0.4829 | 0.7081 | 0.001 |
| Lanczos-4 | 14.65 | 0.4782 | 0.7146 | 0.010 |
| Klasik (NLM+Lanczos+USM) | 14.31 | 0.4476 | 0.7319 | 0.334 |

## Yorum

**1. Algısal kalitede (LPIPS) Real-ESRGAN, 4 grubun hepsinde en iyi sonucu veriyor.**
- `realworld_x4` (gerçekçi bozulma, 4×): Real-ESRGAN x4plus LPIPS **0.297**, Bicubic **0.678** → algısal hata yaklaşık **%56 daha düşük**.
- `realworld_x2`: Real-ESRGAN x2plus LPIPS **0.208**, Bicubic **0.542** → **%62 daha düşük**.
- `bicubic_x4`: x4plus **0.229**, FSRCNN 0.274, Bicubic 0.439.

**2. PSNR'da GAN tabanlı modeller interpolasyonun gerisinde kalabiliyor. Bu beklenen bir sonuç.**
`bicubic_x4` ve `realworld_x4` gruplarında en yüksek PSNR FSRCNN ve Bicubic'te. PSNR piksel bazlı ortalama hatayı ölçer
ve bulanık "ortalama" tahmini ödüllendirir. Real-ESRGAN ise keskin, gerçekçi doku üretir. Bu dokular GT ile piksel piksel
çakışmadığı için PSNR düşer, ama görsel kalite ve LPIPS belirgin şekilde iyileşir. Bu durum literatürde
**algı–bozulma ödünleşimi** (perception–distortion trade-off; Blau & Michaeli, CVPR 2018) olarak bilinir.

**3. Gerçekçi 2× bozulmada Real-ESRGAN General v3 hem PSNR hem SSIM'de de birinci** (24.63 dB / 0.685).
Model gürültü ve JPEG'i temizlediği için, bozulmayı olduğu gibi büyüten interpolasyon yöntemlerini piksel
doğruluğunda da geçiyor.

**4. FSRCNN yalnızca kendi eğitim koşulunda (`bicubic_x4`) iyi.** Gerçekçi bozulmada LPIPS'i Bicubic'e yakın kalıyor
(0.548 ve 0.678 vs. Real-ESRGAN 0.297). Yalnızca bicubic bozulmayla eğitilen CNN'ler gerçek dünya girdisine
genellenemiyor. Projede Real-ESRGAN'ı seçmemizin temel gerekçesi bu.

**5. `old_photo` grubunda tüm yöntemler düşük ve birbirine yakın skor veriyor (PSNR ≈ 14.5 dB).**
Bu tabloda yalnızca süper çözünürlük ölçülüyor. GT renkli, girdi ise sepya tonlu. Renk bilgisi kaybolduğu için hiçbir
SR yöntemi GT'ye piksel düzeyinde yaklaşamaz. Bu grup için asıl iyileştirmeyi tam boru hattı (çizik giderme +
renklendirme/renk restorasyonu) sağlar ve sunum not defterinin 3. bölümünde görsel olarak gösterilir.
LPIPS'te yine Real-ESRGAN General v3 en iyi (0.680).

**6. Hız ve kalite dengesi.** T4 GPU'da General v3 görüntü başına ~0.08–0.16 sn, x4plus ~0.5–1.1 sn sürüyor.
General v3 yaklaşık 14 kat daha az parametreyle x4plus'a yakın LPIPS veriyor. Bu yüzden CPU ve video için
varsayılan model General v3.

## Görsel karşılaştırma
Etiketli karşılaştırma ızgaraları (`grid_<grup>_div2k_080x.jpg`) not defterinin 7. bölümünde gösterilir ve
8. bölümdeki zip ile indirilir.
