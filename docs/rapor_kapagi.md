# Rapor Kapak Sayfası Taslağı

> Köşeli parantez içindeki alanları doldurun. Kapak sayfasında bağlantı hem metin olarak hem de (isteğe bağlı)
> QR kod olarak verilebilir.

---

<div align="center">

**[ÜNİVERSİTE ADI]**
**[FAKÜLTE ADI] — [BÖLÜM ADI]**

&nbsp;

**[DERS KODU] GÖRÜNTÜ İŞLEME**
**Dönem Projesi Raporu**

&nbsp;

# RetroRestore
### Derin Öğrenme Tabanlı Eski Fotoğraf ve Video İyileştirme Sistemi
*Real-ESRGAN ile Süper Çözünürlük, Gürültü Giderme ve Renk Restorasyonu*

&nbsp;

| | |
|---|---|
| **Hazırlayan(lar)** | [Ad Soyad] — [Öğrenci No] |
| | [Ad Soyad] — [Öğrenci No] |
| **Ders Sorumlusu** | [Unvan Ad Soyad] |
| **Teslim Tarihi** | [GG.AA.2026] |
| **GitHub Deposu** | **https://github.com/aysecetin0/retro-restore** |

&nbsp;

[İl], [Ay] 2026

</div>

---

## Proje künyesi (kapak arkası veya özet sayfası için)

| Alan | Bilgi |
|---|---|
| Proje adı | RetroRestore — Eski Fotoğraf ve Video İyileştirme |
| Kaynak kod | https://github.com/aysecetin0/retro-restore |
| Problem | Düşük çözünürlüklü, gürültülü, sıkıştırılmış, solmuş veya renksiz eski fotoğraf ve videoların kör (blind) restorasyonu |
| Ana yöntem | Real-ESRGAN (Wang et al., ICCVW 2021): RRDBNet ve SRVGGNetCompact, döşemeli çıkarım, DNI ile ayarlanabilir gürültü giderme |
| Yardımcı yöntemler | Zhang et al. (ECCV 2016) renklendirme; gri-dünya beyaz dengesi, otomatik seviye ve CLAHE ile renk restorasyonu; morfolojik çizik tespiti ve Telea inpainting |
| Karşılaştırılan yöntemler | Bicubic, Lanczos-4, Klasik (NLM + Lanczos + Unsharp), FSRCNN, EDSR, Real-ESRGAN General v3, x4plus, x2plus |
| Video | FFmpeg ile CFR kare çıkarma, kare kare iyileştirme, aynı fps ile H.264 kodlama ve orijinal ses (senkron korunur) |
| Veri seti | 6 GT görüntü, 28 bozulmuş–GT çifti (bicubic_x4, realworld_x4, realworld_x2, old_photo, grayscale), senkron işaretçili test videosu; seed = 2026 |
| Metrikler | PSNR, SSIM (Y kanalı), LPIPS; referanssız: keskinlik, gürültü σ, renk zenginliği |
| Büyük ölçekli test | DIV2K doğrulama seti: 100 görüntü × 4 bozulma grubu = 400 test çifti, 7 yöntem, Tesla T4 |
| Ana bulgu | Real-ESRGAN gerçekçi bozulmada algısal hatayı (LPIPS) bicubic'e göre %56–62 azaltıyor |
| Yazılım | Python 3.10+, PyTorch, OpenCV, FFmpeg, Flask, saf HTML/CSS/JS |
| Donanım | CPU, NVIDIA CUDA (FP16) veya Apple MPS — otomatik seçim |
| Arayüz | Web: yükleme, parametre seçimi, önce/sonra kaydırıcı, adım adım arka plan akışı, yöntem karşılaştırma, veri seti görüntüleyici, indirme |

## Rapor için önerilen bölüm sırası
1. Giriş ve problem tanımı
2. Literatür ve yöntem karşılaştırması → `docs/02_yontem_karsilastirma.md`
3. Sistem mimarisi → `docs/01_mimari.md`
4. Veri seti ve deney düzeni → `docs/03_veri_seti.md`
5. Uygulama ayrıntıları ve arka plan akışı → `docs/04_arka_plan_akisi.md`
6. Deneysel sonuçlar → `docs/05_sonuclar.md` (DIV2K, 400 test çifti) + not defterindeki grafik ve ızgaralar
7. Web arayüzü ekran görüntüleri
8. Sınırlılıklar ve gelecek çalışmalar
9. Kaynaklar
