"""OpenCV `dnn_superres` ile EDSR / FSRCNN taban çizgileri.

Yalnızca Yöntem Karşılaştırması için kullanılır. `opencv-contrib-python`
ve `models/` altında ilgili .pb dosyası gerekir; yoksa yöntem atlanır.
Bu modeller *yalnızca bicubic bozulma* ile eğitildiği için gerçek dünya
(gürültülü/JPEG'li) girdilerde Real-ESRGAN'a göre zayıf kalır — sunumda bu
farkı görsel olarak göstermek için eklenmiştir.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import settings
from ..models_registry import OPENCV_SR_MODELS

_cache: dict[str, object] = {}


def opencv_sr_available(name: str) -> bool:
    return hasattr(cv2, "dnn_superres") and (settings.models_dir / OPENCV_SR_MODELS[name]["filename"]).exists()


def upscale(img: np.ndarray, name: str) -> np.ndarray:
    spec = OPENCV_SR_MODELS[name]
    if name not in _cache:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(settings.models_dir / spec["filename"]))
        sr.setModel(spec["algo"], spec["scale"])
        _cache[name] = sr
    return _cache[name].upsample(img)
