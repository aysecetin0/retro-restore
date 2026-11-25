"""Eski fotoğraf kusurları: çizik ve toz lekesi giderme (klasik yöntem).

Yaklaşım:
1. Morfolojik top-hat / black-hat ile ince, yüksek kontrastlı yapılar
   (beyaz çizikler, koyu tozlar) aday maske olarak çıkarılır.
2. Uzunluk/incelik ve alan ölçütleriyle maske filtrelenir (gerçek kenarların
   silinmemesi için).
3. Telea (2004) hızlı yürüme (fast marching) inpainting ile doldurulur.

Not: Bu modül opsiyoneldir. Derin öğrenme tabanlı alternatif olarak
"Bringing Old Photos Back to Life" (Wan et al., CVPR 2020) dokümanda
karşılaştırılmıştır.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_defects(img: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = float(np.clip(sensitivity, 0.05, 1.0))
    k = max(3, int(round(min(gray.shape) / 120)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
    white = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    resp = cv2.max(white, black)
    thr = np.percentile(resp, 99.6 - 1.5 * s)
    thr = max(thr, 40 - 25 * s)
    mask = (resp > thr).astype(np.uint8) * 255

    # Bileşen filtresi: büyük bloklar gerçek nesnedir, silmeyelim.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    max_blob = gray.size * 0.002     # kompakt lekeler (toz) için üst sınır
    max_line = gray.size * 0.03      # ince-uzun çizikler için üst sınır
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        elong = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)
        thin = elong > 4 or fill < 0.35
        if (thin and area <= max_line) or area <= min(max_blob, 60):
            keep[labels == i] = 255
    return cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)


def remove_scratches(img: np.ndarray, sensitivity: float = 0.5):
    """(onarılmış görüntü, maske) döndürür."""
    mask = detect_defects(img, sensitivity)
    if mask.max() == 0:
        return img, mask
    radius = max(2, int(round(min(img.shape[:2]) / 300)))
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA), mask
