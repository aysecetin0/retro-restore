"""Önceden eğitilmiş model kaydı: dosya adları, indirme adresleri, açıklamalar.

`scripts/download_models.py` bu kaydı kullanır. Bir adres çalışmazsa sıradaki
yedek (mirror) denenir; hepsi başarısız olursa dosyayı elle `models/` klasörüne
koyabilirsiniz (adlar aşağıdaki `filename` alanlarıdır).
"""
from __future__ import annotations

GH = "https://github.com/xinntao/Real-ESRGAN/releases/download"

SR_MODELS: dict[str, dict] = {
    "realesr-general-x4v3": {
        "label": "Real-ESRGAN General v3 (hafif, hızlı)",
        "arch": "srvgg", "scale": 4, "params_m": 1.21,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
        "filename": "realesr-general-x4v3.pth",
        "urls": [f"{GH}/v0.2.5.0/realesr-general-x4v3.pth"],
        # Gürültü giderme gücü ayarı için 'without denoise' eş modeli (DNI)
        "dni_partner": "realesr-general-wdn-x4v3",
        "notes": "CPU ve video için varsayılan. Gürültü gücü (DNI) ayarlanabilir.",
    },
    "realesr-general-wdn-x4v3": {
        "label": "Real-ESRGAN General v3 (gürültü giderme kapalı eş model)",
        "arch": "srvgg", "scale": 4, "params_m": 1.21, "hidden": True,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
        "filename": "realesr-general-wdn-x4v3.pth",
        "urls": [f"{GH}/v0.2.5.0/realesr-general-wdn-x4v3.pth"],
    },
    "RealESRGAN_x4plus": {
        "label": "Real-ESRGAN x4plus (RRDBNet, en yüksek kalite)",
        "arch": "rrdb", "scale": 4, "params_m": 16.7,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "filename": "RealESRGAN_x4plus.pth",
        "urls": [f"{GH}/v0.1.0/RealESRGAN_x4plus.pth"],
        "notes": "GPU önerilir. Fotoğraflar için en iyi doku üretimi.",
    },
    "RealESRGAN_x2plus": {
        "label": "Real-ESRGAN x2plus (RRDBNet, 2x)",
        "arch": "rrdb", "scale": 2, "params_m": 16.7,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        "filename": "RealESRGAN_x2plus.pth",
        "urls": [f"{GH}/v0.2.1/RealESRGAN_x2plus.pth"],
        "notes": "2x büyütme için doğal model (pixel-unshuffle girişli).",
    },
}

COLORIZATION_FILES: dict[str, dict] = {
    "prototxt": {
        "filename": "colorization_deploy_v2.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt",
            "https://huggingface.co/spaces/Th3BossC/OpenCvColourize/resolve/main/imageModel/utils/models/colorization_deploy_v2.prototxt",
        ],
    },
    "points": {
        "filename": "pts_in_hull.npy",
        "urls": [
            "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy",
            "https://huggingface.co/spaces/Th3BossC/OpenCvColourize/resolve/main/imageModel/utils/models/pts_in_hull.npy",
        ],
    },
    "caffemodel": {
        "filename": "colorization_release_v2.caffemodel",
        "urls": [
            "https://huggingface.co/spaces/Th3BossC/OpenCvColourize/resolve/main/imageModel/utils/models/colorization_release_v2.caffemodel",
            "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
        ],
    },
}

# OpenCV dnn_superres ile çalışan klasik derin öğrenme taban çizgileri
# (yalnızca Yöntem Karşılaştırması içindir; opencv-contrib gerekir)
OPENCV_SR_MODELS: dict[str, dict] = {
    "edsr": {
        "label": "EDSR x4 (Lim et al. 2017)", "algo": "edsr", "scale": 4,
        "filename": "EDSR_x4.pb",
        "urls": ["https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"],
    },
    "fsrcnn": {
        "label": "FSRCNN x4 (Dong et al. 2016, SRCNN'in hızlı sürümü)", "algo": "fsrcnn", "scale": 4,
        "filename": "FSRCNN_x4.pb",
        "urls": ["https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"],
    },
}
