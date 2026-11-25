"""Görüntü okuma/yazma yardımcıları (Unicode yol güvenli, EXIF yönlendirmeli)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray:
    """Görüntüyü BGR (veya BGRA) uint8 olarak okur.

    * Türkçe karakterli yollarda çalışması için np.fromfile + imdecode kullanılır.
    * EXIF yönlendirmesi uygulanır (telefon fotoğrafları yan dönmesin).
    * 16-bit görüntüler 8-bit'e indirgenir, gri görüntüler 3 kanala çıkarılır.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Görüntü okunamadı ya da desteklenmeyen biçim: {path}")
    # EXIF yönü: IMREAD_UNCHANGED yönü uygulamaz, IMREAD_COLOR uygular.
    # Alfa kanalı yoksa IMREAD_COLOR ile tekrar çözerek yönü düzeltiriz.
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3):
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def imwrite(path: str | Path, img: np.ndarray, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Görüntü yazılamadı: {path}")
    buf.tofile(str(path))
    return path


def make_thumbnail(img: np.ndarray, max_side: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def split_alpha(img: np.ndarray):
    """BGRA -> (BGR, alpha | None)."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3].copy(), img[:, :, 3].copy()
    return img, None


def merge_alpha(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return bgr
    h, w = bgr.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.dstack([bgr, alpha])
