# models/

Önceden eğitilmiş ağırlıklar bu klasöre indirilir (boyutları nedeniyle Git'e eklenmez).

```bash
python scripts/download_models.py          # Real-ESRGAN (4 dosya) + renklendirme (3 dosya)
python scripts/download_models.py --all    # + EDSR / FSRCNN karşılaştırma modelleri
python scripts/download_models.py --list   # dosya adları ve kaynakları
```

| Dosya | Model | Boyut (yakl.) | Kaynak |
|---|---|---|---|
| realesr-general-x4v3.pth | SRVGGNetCompact, 4× (hafif) | 5 MB | xinntao/Real-ESRGAN v0.2.5.0 |
| realesr-general-wdn-x4v3.pth | Aynı ağ, gürültü giderme kapalı (DNI eşi) | 5 MB | xinntao/Real-ESRGAN v0.2.5.0 |
| RealESRGAN_x4plus.pth | RRDBNet, 4× | 64 MB | xinntao/Real-ESRGAN v0.1.0 |
| RealESRGAN_x2plus.pth | RRDBNet, 2× | 64 MB | xinntao/Real-ESRGAN v0.2.1 |
| colorization_release_v2.caffemodel + .prototxt + pts_in_hull.npy | Zhang et al. 2016 renklendirme | 129 MB | richzhang/colorization |
| EDSR_x4.pb, FSRCNN_x4.pb | Karşılaştırma taban çizgileri | 38 MB / 40 KB | Saafke/EDSR_Tensorflow, FSRCNN_Tensorflow |

Otomatik indirme başarısız olursa dosyayı tarayıcıdan indirip **aynı adla** bu klasöre koyun.
