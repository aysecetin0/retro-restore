# Değerlendirme sonuçları (DIV2K, 400 test çifti, Tesla T4)

Ayrıntı ve yorum: docs/05_sonuclar.md

## bicubic_x4

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x4plus | 22.09 | 0.6476 | **0.2285** | 0.543 |
| Real-ESRGAN General v3 | 22.32 | 0.6712 | 0.2454 | 0.084 |
| FSRCNN (CNN) | **24.40** | **0.7189** | 0.2741 | 0.037 |
| Real-ESRGAN x2plus | 22.13 | 0.6219 | 0.3545 | 0.412 |
| Bicubic | 23.78 | 0.7075 | 0.4391 | 0.001 |
| Lanczos-4 | 23.53 | 0.6945 | 0.4528 | 0.009 |
| Klasik (NLM+Lanczos+USM) | 22.33 | 0.6678 | 0.4927 | 0.104 |

## realworld_x4

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x4plus | 21.44 | 0.5666 | **0.2969** | 0.580 |
| Real-ESRGAN General v3 | 22.39 | **0.5916** | 0.3513 | 0.082 |
| Real-ESRGAN x2plus | 22.09 | 0.5823 | 0.3835 | 0.422 |
| FSRCNN (CNN) | 22.94 | 0.5842 | 0.5480 | 0.030 |
| Klasik (NLM+Lanczos+USM) | 22.80 | 0.5828 | 0.6016 | 0.104 |
| Bicubic | **23.00** | 0.5862 | 0.6776 | 0.001 |
| Lanczos-4 | **23.00** | 0.5860 | 0.6802 | 0.008 |

## realworld_x2

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN x2plus | 23.57 | 0.6692 | **0.2082** | 0.560 |
| Real-ESRGAN x4plus | 23.07 | 0.6583 | 0.2489 | 1.127 |
| Real-ESRGAN General v3 | **24.63** | **0.6854** | 0.3359 | 0.159 |
| Klasik (NLM+Lanczos+USM) | 23.84 | 0.6344 | 0.5084 | 0.329 |
| FSRCNN (CNN) | 24.08 | 0.6494 | 0.5267 | 0.156 |
| Lanczos-4 | 24.11 | 0.6457 | 0.5403 | 0.009 |
| Bicubic | 24.15 | 0.6478 | 0.5423 | 0.001 |

## old_photo

| Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn/görüntü) |
|---|---:|---:|---:|---:|
| Real-ESRGAN General v3 | 14.49 | 0.4886 | **0.6801** | 0.142 |
| FSRCNN (CNN) | **14.72** | **0.4898** | 0.6942 | 0.136 |
| Real-ESRGAN x4plus | 14.53 | 0.4779 | 0.6988 | 1.099 |
| Real-ESRGAN x2plus | 14.59 | 0.4648 | 0.7075 | 0.546 |
| Bicubic | 14.66 | 0.4829 | 0.7081 | 0.001 |
| Lanczos-4 | 14.65 | 0.4782 | 0.7146 | 0.010 |
| Klasik (NLM+Lanczos+USM) | 14.31 | 0.4476 | 0.7319 | 0.334 |
