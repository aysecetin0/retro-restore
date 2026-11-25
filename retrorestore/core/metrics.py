"""Nicel değerlendirme metrikleri.

* PSNR / SSIM: referanslı (ground-truth gerekir), Y (luma) kanalında hesaplanır
  — süper çözünürlük literatüründeki yaygın uygulama.
* LPIPS: algısal benzerlik (isteğe bağlı, `pip install lpips` gerekir).
* Keskinlik (Laplacian varyansı) ve gürültü tahmini: referanssız, eski
  fotoğraflar gibi GT'si olmayan örneklerde kullanılır.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_y(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return ycc[:, :, 0].astype(np.float64)


def _align(a: np.ndarray, b: np.ndarray, crop: int):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    if crop > 0 and h > 2 * crop and w > 2 * crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return a, b


def psnr(pred: np.ndarray, gt: np.ndarray, crop: int = 4) -> float:
    p, g = _align(_to_y(pred), _to_y(gt), crop)
    mse = np.mean((p - g) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray, crop: int = 4) -> float:
    """Wang et al. (2004) SSIM, 11x11 Gauss penceresi, sigma=1.5."""
    p, g = _align(_to_y(pred), _to_y(gt), crop)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = k @ k.T
    mu1 = cv2.filter2D(p, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(g, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(p ** 2, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(g ** 2, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(p * g, -1, win)[5:-5, 5:-5] - mu12
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def sharpness(img: np.ndarray) -> float:
    """Laplacian varyansı — yüksek değer daha fazla yüksek frekans (detay veya gürültü)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def noise_sigma(img: np.ndarray) -> float:
    """Immerkær (1996) hızlı gürültü standart sapması tahmini."""
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float64)
    h, w = g.shape
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    s = np.abs(cv2.filter2D(g, -1, m))[1:-1, 1:-1].sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def colorfulness(img: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) renk zenginliği ölçüsü."""
    b, g, r = [c.astype(np.float64) for c in cv2.split(img[:, :, :3])]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


_lpips_model = None


def lpips_distance(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """LPIPS (AlexNet). Paket yoksa None döner."""
    global _lpips_model
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    p, g = _align(pred, gt, 0)

    def t(x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)

    with torch.no_grad():
        return float(_lpips_model(t(p), t(g)).item())


def no_reference_report(img: np.ndarray) -> dict:
    return {
        "sharpness": round(sharpness(img), 2),
        "noise_sigma": round(noise_sigma(img), 3),
        "colorfulness": round(colorfulness(img), 2),
    }
