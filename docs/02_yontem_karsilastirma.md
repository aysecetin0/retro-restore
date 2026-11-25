# 2 · Yöntem Karşılaştırması ve Seçim Gerekçesi

Bu belge sunumda "Hangi yöntemleri inceledik, neden Real-ESRGAN'ı seçtik?" sorusunu yanıtlamak için hazırlanmıştır.
Nicel sonuçlar kendi donanımınızda `python scripts/evaluate.py` ile üretilir ve `dataset/results/summary.md`
dosyasına yazılır. Görsel kanıtlar `dataset/results/grid_*.jpg` dosyalarında ve web arayüzündeki **Yöntem Karşılaştırma**
sekmesinde bulunur.

## 2.1 Problem: eski fotoğraf ve videolarda "kör" (blind) bozulma

Eski bir fotoğrafta veya videoda tek bir bozulma yoktur, birkaç bozulma üst üste biner:

| Bozulma | Kaynağı | Görünümü |
|---|---|---|
| Düşük çözünürlük | Eski kamera, küçük baskı, düşük çözünürlüklü tarama | Detay kaybı, pikselleşme |
| Bulanıklık | Odak/hareket, lens, tarayıcı | Yumuşak kenarlar |
| Gürültü / gren | Film greni, sensör, tarama | Rastgele noktalar |
| Sıkıştırma artefaktı | JPEG, MPEG/H.264 düşük bit hızı | 8×8 bloklar, halkalanma (ringing) |
| Solma, renk kayması | Kimyasal yaşlanma | Düşük kontrast, sararma / sepya |
| Çizik, toz | Fiziksel hasar | İnce çizgiler, noktalar |
| Renk yokluğu | Siyah-beyaz film | Tek kanal |

Bozulmanın türü ve şiddeti **bilinmez**. Bu nedenle bozulmanın yalnızca bicubic küçültme olduğunu varsayan yöntemler
gerçek fotoğraflarda başarısız olur.

## 2.2 Karşılaştırma tablosu

| Yöntem | Yıl / Kaynak | Yaklaşım | Kayıp / Eğitim | Gerçek bozulmaya dayanıklılık | Hesaplama maliyeti | Artıları | Eksileri |
|---|---|---|---|---|---|---|---|
| **En yakın komşu / Bilinear** | – | Sabit interpolasyon | – | ✗ | ≈0 | Anlık | Pikselli veya bulanık sonuç |
| **Bicubic** | Keys 1981 | 4×4 kübik çekirdek | – | ✗ (gürültü ve JPEG'i de büyütür) | ≈0 | Standart taban çizgisi, PSNR'da şaşırtıcı derecede iyi | Yeni detay üretmez, bulanık |
| **Lanczos-4** | – | Pencereli sinc (8×8) | – | ✗ | ≈0 | Bicubic'ten keskin | Halkalanma (ringing) |
| **Klasik filtreleme** (NLM, bilateral, median + unsharp) | Buades 2005, Tomasi 1998 | Elle tasarlanmış ön bilgi (prior) | – | △ Yalnızca gürültüye karşı | Düşük (NLM orta) | Açıklanabilir, eğitim gerektirmez | Detayı gürültüyle birlikte siler, keskinleştirme gürültüyü artırır |
| **SRCNN / FSRCNN** | Dong 2014 / 2016 | 3 katmanlı sığ CNN | L2 (MSE) | ✗ Bicubic bozulmayla eğitilmiş | Düşük | İlk derin SR, hızlı | Pürüzsüz ve bulanık çıktı, sınırlı kapasite |
| **EDSR** | Lim 2017 | Derin ResNet, BatchNorm yok | L1 | ✗ Sentetik bozulmayla eğitilmiş | Yüksek (43M param.) | Bicubic veri setlerinde yüksek PSNR | Gerçek gürültüyü "keskinleştirir", dokular yağlı boya gibi görünür |
| **SRGAN / ESRGAN** | Ledig 2017 / Wang 2018 | RRDB üreteci + GAN + algısal (VGG) kayıp | L1 + algısal + GAN | △ | Yüksek (16.7M) | Keskin, gerçekçi dokular | Bozulma modeli basit, gerçek girdide artefakt üretebilir |
| **BSRGAN** | Zhang 2021 | ESRGAN + rastgele karıştırılmış bozulma | aynı | ✓ | Yüksek | Gerçek dünyaya dayanıklı | Ağırlıkları ve kodu daha az yaygın |
| **SwinIR** | Liang 2021 | Swin Transformer | L1 / GAN | ✓ (real-world sürümü) | Çok yüksek | En güncel kalite | Yavaş, VRAM'i fazla kullanır, video için pahalı |
| **Real-ESRGAN** ✔ | **Wang 2021** | RRDBNet veya SRVGGNetCompact + **U-Net ayırt edici (spectral norm)** | L1 + algısal + GAN | **✓✓ Yüksek dereceli bozulma sentezi** | Orta (compact: düşük) | Bulanıklık, gürültü, JPEG ve yeniden boyutlandırmayı tek geçişte giderir; hafif model ve DNI ile ayarlanabilir; olgun, açık ağırlıklar | Çok ağır hasarda doku "uydurabilir" (halüsinasyon), yüzlerde sınırlı |

### Renk restorasyonu ve renklendirme

| Yöntem | Yaklaşım | Artıları | Eksileri | Projede |
|---|---|---|---|---|
| Gri-dünya beyaz dengesi + otomatik seviye + CLAHE | Klasik istatistik | Hızlı, öngörülebilir, solmuş renkli fotoğrafta etkili | Renk bilgisi hiç yoksa renk üretemez | ✔ "Renk restorasyonu" modu |
| **Zhang et al. 2016** (Colorful Image Colorization) | CNN; 313 ab sınıfı üzerinde sınıflandırma + yeniden dengeleme | OpenCV DNN ile çalışır (PyTorch gerektirmez), hafif (129 MB), canlı renkler | Bazen soluk veya kahverengimsi tonlar verir, nesneyi anlama kapasitesi sınırlı | ✔ "Renklendirme" modu |
| DeOldify | GAN (NoGAN eğitimi), U-Net | Doğal renkler, video modu var | fastai bağımlılığı, ağır | Alternatif olarak anlatılır |
| DDColor | Transformer, çift dekoder | Güncel en iyi sonuçlar | Ağır, VRAM ister | Gelecek çalışma |
| Bringing Old Photos Back to Life | VAE + çizik tespiti + yüz modülü | Eski fotoğrafa özel uçtan uca çözüm | Karmaşık kurulum | Çizik giderme fikrine ilham verdi |

## 2.3 Neden Real-ESRGAN? (sunum notları)

> **Tek cümle:** Eski fotoğraf ve videolardaki bozulma bilinmez ve karışıktır. Real-ESRGAN tam olarak bu kör senaryo
> için, gerçek dünya bozulmalarını taklit eden yüksek dereceli bir bozulma sentezi ile eğitildiğinden gürültü giderme,
> sıkıştırma artefaktı temizleme, keskinleştirme ve süper çözünürlüğü **tek bir modelle, tek geçişte** yapar.

**1. Eğitim verisi gerçeğe yakın.** SRCNN, EDSR ve ESRGAN eğitimde düşük çözünürlüklü görüntüyü yalnızca bicubic
küçültmeyle üretir. Real-ESRGAN ise iki tur `bulanıklık → yeniden boyutlandırma → gürültü → JPEG` uygular ve araya
sinc filtresi (halkalanma artefaktı) ekler. Bizim `make_dataset.py` dosyamızdaki `realworld_x4` grubu bu fikrin
sadeleştirilmiş halidir. `bicubic_x4` ile `realworld_x4` sonuçları karşılaştırıldığında, sentetik bozulmada iyi olan
yöntemlerin gerçekçi bozulmada nasıl çöktüğü görülebilir.

**2. Mimari kararları.**
- *RRDB (Residual-in-Residual Dense Block):* BatchNorm kullanılmaz (BN, renk kaymasına ve artefakta yol açar), artık
  ölçekleme (×0.2) ile derin ağ kararlı eğitilir.
- *U-Net + spectral normalization ayırt edici:* Yalnızca tüm görüntü için değil, **piksel düzeyinde** gerçek/sahte
  geri bildirimi verir. Yerel doku kalitesi artar, eğitim kararlı olur.
- *SRVGGNetCompact (general-x4v3):* RRDBNet'in yaklaşık 1/14'ü kadar parametreye (1.2M) sahiptir ve CPU'da bile
  makul sürede çalışır. **Video** gibi binlerce kare işlenen durumlarda belirleyici avantaj sağlar.

**3. Ayarlanabilirlik (DNI).** `realesr-general-x4v3` ile gürültü gidermeyen eşi `wdn` arasında ağırlık uzayında
doğrusal interpolasyon yapılır (Deep Network Interpolation):
`θ = w·θ_denoise + (1−w)·θ_wdn`. Arayüzdeki **"Gürültü giderme gücü"** kaydırıcısı tam olarak bu `w` değeridir. Film
grenini korumak isteyen kullanıcı ile temiz görüntü isteyen kullanıcı aynı modeli kullanabilir.

**4. Mühendislik olgunluğu.** Açık ağırlıklar, geniş topluluk, döşemeli çıkarım ve FP16 desteği mevcuttur. Biz
mimariyi saf PyTorch ile yeniden yazdık (`retrorestore/methods/realesrgan_arch.py`). Böylece `basicsr` kütüphanesinin
yeni torchvision sürümleriyle yaşadığı uyumsuzluklardan bağımsız kalıyoruz ve resmi ağırlıklar doğrudan yükleniyor.

**5. Maliyet ve kalite dengesi.** SwinIR ve DDColor gibi güncel transformer modelleri bazı ölçütlerde daha iyidir,
ancak 5–10 kat daha yavaştır ve çok daha fazla VRAM ister. Ders projesinde hem görüntü **hem video** hedeflendiği için
Real-ESRGAN, kalite, hız ve kurulum kolaylığı açısından en dengeli seçenektir.

### Beklenen soru: "PSNR tablosunda Real-ESRGAN neden bicubic'ten düşük çıkabiliyor?"

Bu beklenen ve açıklanması gereken bir sonuçtur:
- PSNR ve SSIM piksel bazında ortalama hatayı ölçer. Belirsizlik olduğunda en iyi PSNR'ı **bulanık ortalama** tahmin verir.
- GAN tabanlı modeller *keskin ve gerçekçi* dokular üretir. Bu dokular GT ile piksel piksel çakışmaz, bu yüzden PSNR
  düşebilir, ancak insan gözü için kalite belirgin şekilde artar.
- Bu durum literatürde **algı–bozulma ödünleşimi** (perception–distortion trade-off, Blau & Michaeli, CVPR 2018) olarak
  bilinir. Bu nedenle değerlendirmede PSNR ve SSIM'in yanında **LPIPS** (algısal, düşük değer daha iyi) ve görsel
  karşılaştırma ızgaraları da sunulur.
- `realworld_x4` ve `old_photo` gruplarında, gürültü ve JPEG'i büyüten interpolasyon yöntemleri hem görsel olarak hem
  LPIPS'te Real-ESRGAN'ın gerisinde kalır.

### Sınırlılıklar (dürüst değerlendirme)
- Çok küçük yüzlerde Real-ESRGAN yetersiz kalabilir. **GFPGAN / CodeFormer** gibi yüz önselli modeller eklenebilir.
- Karelerin bağımsız işlenmesi videoda hafif titreşime (temporal flicker) yol açabilir. Renklendirmede bunu EMA
  yumuşatmayla azaltıyoruz. SR için zamansal modeller (BasicVSR++, RealBasicVSR) gelecek çalışmadır.
- Renklendirme "doğru" rengi bilemez, yalnızca *makul* bir renk önerir. Tarihî belge için uyarı notu eklenmelidir.

## 2.4 Kaynaklar
1. C. Dong et al., "Image Super-Resolution Using Deep Convolutional Networks," ECCV 2014 / TPAMI 2016.
2. B. Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)," CVPRW 2017.
3. C. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)," CVPR 2017.
4. X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," ECCVW 2018.
5. X. Wang, L. Xie, C. Dong, Y. Shan, "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCVW 2021.
6. K. Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (BSRGAN)," ICCV 2021.
7. J. Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCVW 2021.
8. R. Zhang, P. Isola, A. Efros, "Colorful Image Colorization," ECCV 2016.
9. Z. Wan et al., "Bringing Old Photos Back to Life," CVPR 2020.
10. A. Buades, B. Coll, J.-M. Morel, "A Non-Local Algorithm for Image Denoising," CVPR 2005.
11. A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method," J. Graphics Tools, 2004.
12. Y. Blau, T. Michaeli, "The Perception-Distortion Tradeoff," CVPR 2018.
13. R. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)," CVPR 2018.
14. X. Wang et al., "Towards Real-World Blind Face Restoration with Generative Facial Prior (GFPGAN)," CVPR 2021.
