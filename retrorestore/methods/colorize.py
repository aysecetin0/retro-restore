"""Siyah-beyaz fotoğraf renklendirme.

Model: Zhang, Isola, Efros — "Colorful Image Colorization", ECCV 2016.
* Girdi: Lab uzayının L (açıklık) kanalı, 224x224.
* Çıktı: 313 nicemlenmiş ab rengi üzerinde olasılık dağılımı → "annealed mean"
  ile ab kanallarına dönüştürülür (OpenCV DNN, PyTorch gerektirmez).
* Tam çözünürlüklü L kanalı korunur, yalnızca düşük çözünürlüklü ab kanalları
  büyütülür → keskinlik kaybı olmaz.

Video için: karelerde renk titremesini azaltmak amacıyla ab kanallarına
üstel hareketli ortalama (EMA) ile zamansal yumuşatma uygulanır.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..config import settings
from ..models_registry import COLORIZATION_FILES


class ColorizerUnavailable(RuntimeError):
    pass


def colorizer_ready() -> bool:
    return all((settings.models_dir / f["filename"]).exists() for f in COLORIZATION_FILES.values())


class Colorizer:
    def __init__(self):
        if not colorizer_ready():
            raise ColorizerUnavailable(
                "Renklendirme modeli eksik. `python scripts/download_models.py --colorization` çalıştırın.")
        md = settings.models_dir
        f = {k: str(md / v["filename"]) for k, v in COLORIZATION_FILES.items()}
        self.net = cv2.dnn.readNetFromCaffe(f["prototxt"], f["caffemodel"])
        pts = np.load(f["points"]).transpose().reshape(2, 313, 1, 1).astype(np.float32)
        self.net.getLayer(self.net.getLayerId("class8_ab")).blobs = [pts]
        self.net.getLayer(self.net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, np.float32)]
        self._lock = threading.Lock()

    def predict_ab(self, img_bgr: np.ndarray) -> np.ndarray:
        """Düşük çözünürlüklü (56x56) değil, girdi boyutunda ab kanalları döndürür."""
        h, w = img_bgr.shape[:2]
        rgb_small = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        l_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2LAB)[:, :, 0] - 50.0
        with self._lock:  # cv2.dnn.Net iş parçacığı güvenli değildir
            self.net.setInput(cv2.dnn.blobFromImage(l_small))
            ab = self.net.forward()[0].transpose((1, 2, 0))
        return cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    def colorize(self, img_bgr: np.ndarray, strength: float = 1.0,
                 prev_ab: np.ndarray | None = None, temporal: float = 0.0):
        """(renkli BGR, kullanılan ab) döndürür.

        strength: ab kanallarının ölçeği (0 = gri, 1 = model çıktısı, >1 daha canlı)
        temporal: 0..0.95, önceki karenin ab'si ile harmanlama oranı (video)
        """
        ab = self.predict_ab(img_bgr)
        if prev_ab is not None and temporal > 0 and prev_ab.shape == ab.shape:
            ab = temporal * prev_ab + (1 - temporal) * ab
        lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        out = np.concatenate([lab[:, :, :1], ab * float(strength)], axis=2)
        bgr = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0, 1)
        return (bgr * 255).round().astype(np.uint8), ab


_inst: Colorizer | None = None
_inst_lock = threading.Lock()


def get_colorizer() -> Colorizer:
    global _inst
    with _inst_lock:
        if _inst is None:
            _inst = Colorizer()
        return _inst
