"""Klasik (öğrenme tabanlı olmayan) görüntü işleme yöntemleri.

Bu modül iki amaca hizmet eder:
1. Derin öğrenme modeli yokken *fallback* boru hattı.
2. Yöntem karşılaştırmasında *taban çizgisi* (baseline): Bicubic, Lanczos,
   Klasik (NLM denoise + Lanczos + Unsharp).
"""
from __future__ import annotations

import cv2
import numpy as np

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


# ---------------------------------------------------------------------------
# Yeniden örnekleme (interpolasyon)
# ---------------------------------------------------------------------------
def resize_by(img: np.ndarray, scale: float, method: str = "lanczos") -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = INTERP[method]
    if scale < 1 and method in ("bicubic", "lanczos"):
        interp = cv2.INTER_AREA  # küçültmede aliasing'i önler
    return cv2.resize(img, (nw, nh), interpolation=interp)


def resize_to(img: np.ndarray, size: tuple[int, int], method: str = "lanczos") -> np.ndarray:
    w, h = size
    if (img.shape[1], img.shape[0]) == (w, h):
        return img
    interp = INTERP[method]
    if w < img.shape[1] and method in ("bicubic", "lanczos"):
        interp = cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)


# ---------------------------------------------------------------------------
# Gürültü giderme
# ---------------------------------------------------------------------------
def denoise_nlm(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Non-Local Means (Buades et al. 2005). strength ∈ [0,1] -> h ∈ [0, 15]."""
    strength = float(np.clip(strength, 0, 1))
    if strength <= 0:
        return img
    h = 2 + 13 * strength
    return cv2.fastNlMeansDenoisingColored(img, None, h, h * 0.8, 7, 21)


def denoise_bilateral(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    s = float(np.clip(strength, 0, 1))
    if s <= 0:
        return img
    return cv2.bilateralFilter(img, d=7, sigmaColor=20 + 60 * s, sigmaSpace=5 + 5 * s)


def denoise_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Tuz-biber (salt & pepper) türü toz noktaları için."""
    return cv2.medianBlur(img, ksize)


def deblock_jpeg(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Basit JPEG blok artefaktı yumuşatma (kenar korumalı)."""
    return denoise_bilateral(img, 0.3 * strength) if strength > 0 else img


# ---------------------------------------------------------------------------
# Keskinleştirme
# ---------------------------------------------------------------------------
def unsharp_mask(img: np.ndarray, amount: float = 0.6, radius: float = 1.2,
                 threshold: int = 2) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    if threshold > 0:
        low_contrast = np.abs(img.astype(np.int16) - blur.astype(np.int16)) < threshold
        sharp = np.where(low_contrast, img, sharp)
    return sharp.astype(np.uint8)


# ---------------------------------------------------------------------------
# Klasik süper çözünürlük taban çizgisi
# ---------------------------------------------------------------------------
def classical_sr(img: np.ndarray, scale: float, denoise: float = 0.4,
                 sharpen: float = 0.6) -> np.ndarray:
    """NLM denoise -> Lanczos büyütme -> Unsharp mask."""
    out = denoise_nlm(img, denoise)
    out = resize_by(out, scale, "lanczos")
    out = unsharp_mask(out, amount=sharpen, radius=1.0 + 0.3 * scale)
    return out


# ---------------------------------------------------------------------------
# Renk restorasyonu (soluk / sararmış / renk kaymalı fotoğraflar)
# ---------------------------------------------------------------------------
def is_grayscale(img: np.ndarray, tol: float = 6.0) -> bool:
    """Kanallar arası ortalama fark küçükse görüntü gri/sepya kabul edilir.

    Sepya tonlu fotoğraflar da (tek tonlu) gri kabul edilir; bunun için
    Lab uzayında a/b kanallarının *varyansına* bakılır (ortalamaya değil).
    """
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lab[:, :, 1].std() + lab[:, :, 2].std()) < tol


def gray_world_wb(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gri-dünya varsayımıyla beyaz dengesi (renk kayması / sararma düzeltme)."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    gain = means.mean() / means
    gain = 1 + (gain - 1) * float(np.clip(strength, 0, 1))
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def auto_levels(img: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Kanal bazlı yüzdelik germe: soluk fotoğraflarda kontrastı geri getirir."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, (low_pct, high_pct))
        if hi - lo < 10:
            out[:, :, c] = ch
            continue
        out[:, :, c] = np.clip((ch.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return out


def clahe_luma(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Lab uzayında yalnızca L kanalına CLAHE (renkleri bozmadan yerel kontrast)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def boost_saturation(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
    if abs(factor - 1) < 1e-3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def restore_faded_colors(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Soluk renkli fotoğraf için klasik renk canlandırma zinciri.

    Beyaz dengesi -> otomatik seviye -> CLAHE (L) -> doygunluk artırımı.
    `strength` ile orijinal görüntüyle harmanlanır.
    """
    s = float(np.clip(strength, 0, 1))
    if s <= 0:
        return img
    out = gray_world_wb(img, 0.8)
    out = auto_levels(out)
    out = clahe_luma(out, clip=1.5 + 1.5 * s)
    out = boost_saturation(out, 1.0 + 0.45 * s)
    return cv2.addWeighted(out, s, img, 1 - s, 0)


def desepia(img: np.ndarray) -> np.ndarray:
    """Sepya/sararma tonunu giderip nötr griye çevirir (renklendirme öncesi)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
