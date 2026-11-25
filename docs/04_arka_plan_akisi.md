# 4 · Arka Plan Akışı (şeffaflık)

Projenin önemli bir hedefi, "kara kutu" görünümünden kaçınmak ve **her adımda ne olduğunu göstermek**tir.

## 4.1 Üç yerden izlenebilir

1. **Sunum not defteri (`RetroRestore_Colab.ipynb`, bölüm 3 ve 6):** Bir fotoğraf ve bir video boru hattından
   geçirilir; her adımın aşaması (ön işleme, çıkarım, son işleme, video), parametreleri, süresi ve **ara görüntüsü**
   tablo ve görsel olarak gösterilir. Model yoksa adım "fallback" olarak işaretlenir.
2. **İş klasörü:** `outputs/jobs/<iş-id>/`
   ```
   input.png           yüklenen dosya
   gt.png              (veri setinden seçildiyse) ground truth
   result.png|mp4      çıktı
   steps/01_preprocess.jpg, 02_scratch.jpg, 03_sr.jpg ...   ara görüntüler
   trace.json          adımların tam kaydı
   job.json            iş üst verisi (parametreler, süreler, metrikler)
   frame_samples/      (video) ilk, orta ve son karenin önce/sonra hali
   input_preview.mp4   (video) tarayıcıda yan yana karşılaştırma için
   error.log           (hata olursa) tam yığın izi
   ```
3. **Komut satırı:** `python enhance_cli.py girdi.png -o cikti.png --trace` komutu `cikti_trace/` klasörüne aynı
   kaydı yazar.

## 4.2 Görüntü boru hattı adım adım

| # | Adım (`key`) | Ne yapar? | Neden? |
|---|---|---|---|
| 1 | `preprocess` | Gri → BGR, alfa kanalını ayırma, piksel limiti, S/B veya sepya tespiti (Lab a/b std), gürültü σ tahmini | Bellek güvenliği ve otomatik karar verme |
| 2 | `scratch` *(ops.)* | Top-hat ve black-hat ile ince parlak/koyu yapıları bulur, şekil filtresi uygular, Telea inpainting ile doldurur | Çizik ve tozlar SR'dan **önce** giderilmezse büyütülür |
| 3 | `desepia` *(S/B ise)* | Sepya tonunu nötr griye çevirir, hafif CLAHE uygular | Renklendirme modeli L kanalı bekler |
| 4 | `pre_denoise` *(ops.)* | NLM ön gürültü giderme | Aşırı gürültülü girdide x4plus'a yardım eder |
| 5 | `sr` | Real-ESRGAN döşemeli çıkarım (veya klasik, Lanczos, Bicubic) | Çözünürlük artırma, gürültü/JPEG/bulanıklık giderme |
| 6 | `colorize` / `color` | Zhang 2016 renklendirme **veya** beyaz dengesi → otomatik seviye → CLAHE → doygunluk | Renk restorasyonu |
| 7 | `sharpen` *(ops.)* | Eşikli unsharp mask | İnce ayar |
| 8 | `save` | PNG/JPG/WEBP kaydı, alfa birleştirme | – |

**Neden renk işlemi SR'dan sonra?** Renklendirme ağı 224×224 girdide çalışır ve ab kanallarını üretir. Bu kanallar
**SR çıktısının tam çözünürlüklü L kanalıyla** birleştirildiğinde keskinlik kaybı olmaz. SR'dan önce renklendirme
yapılsaydı renk kanalları da büyütülecek ve renk taşmaları belirginleşecekti.

## 4.3 Real-ESRGAN çıkarımı içeride nasıl çalışır?

```
BGR uint8 → RGB float [0,1] → tensör (1,3,H,W) [FP16 GPU'da]
  → ön dolgu (reflect, 10 px)            kenar artefaktlarını önler
  → mod dolgusu (x2 modeli: çift boyut)   pixel_unshuffle gereksinimi
  → döşemelere böl (ör. 256 px + 10 px örtüşme)
       her döşeme: GPU'ya → ağ → CPU'ya → çıktı tuvaline örtüşme kırpılarak yaz
  → dolguları kırp → [0,1] sınırla → uint8 BGR
  → hedef ölçek ≠ model ölçeği ise Lanczos/Area ile yeniden boyutlandır (ör. 4× model → 2× çıktı)
```

## 4.4 Video boru hattı adım adım

| # | Adım | Komut / işlem |
|---|---|---|
| 1 | `probe` | `ffprobe -show_streams -show_format -of json` ile fps (`avg_frame_rate`), süre, ses varlığı ve döndürme bilgisi okunur |
| 2 | `extract` | `ffmpeg -i in.mp4 -t <sn> -vf fps=<fps> -pix_fmt rgb24 frames_in/%08d.png` → **sabit kare hızı (CFR)**, kayıpsız PNG |
| 3 | `frames` | Her kare `enhance_image` fonksiyonundan geçer. İlk karede verilen renk kararı tüm videoda sabit kalır, renklendirmede önceki karenin ab kanalı EMA ile harmanlanır (titreme azaltma). İlerleme ve kalan süre (ETA) canlı gösterilir |
| 4 | `assemble` | `ffmpeg -framerate <fps> -i frames_out/%08d.png -t <sn> -i in.mp4 -map 0:v -map 1:a? -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest out.mp4` |
| 5 | önizleme | Girdinin H.264 kopyası (tarayıcıda yan yana senkron oynatma için) |

**Senkron garantisi:** Çıkarmada ve birleştirmede aynı fps ile aynı kare sayısı kullanıldığı için görüntü süresi
kaynakla aynı olur. Ses ise kaynaktan aynı `-t` aralığıyla, yeniden zamanlanmadan alınır.
