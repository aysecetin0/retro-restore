"""Yöntem karşılaştırma (benchmark).

Aynı girdi üzerinde birden çok yöntemi çalıştırır; süre ve kalite metriklerini
toplar, sunuma/rapora doğrudan konabilecek etiketli bir karşılaştırma ızgarası
(grid) üretir.

* Ground-truth (GT) varsa: PSNR↑, SSIM↑, (ops.) LPIPS↓
* GT yoksa (gerçek eski fotoğraf): referanssız ölçüler (keskinlik, gürültü σ)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..core import metrics
from ..core.device import TORCH_AVAILABLE
from ..methods import classical, opencv_sr
from ..methods.realesrgan import ModelUnavailable, available_models, get_upsampler
from ..models_registry import OPENCV_SR_MODELS, SR_MODELS
from ..utils.io import imwrite

METHOD_LABELS = {
    "nearest": "En yakın komşu",
    "bicubic": "Bicubic",
    "lanczos": "Lanczos-4",
    "classical": "Klasik (NLM+Lanczos+USM)",
    "fsrcnn": "FSRCNN (CNN)",
    "edsr": "EDSR (CNN)",
    "realesr-general-x4v3": "Real-ESRGAN General v3",
    "RealESRGAN_x4plus": "Real-ESRGAN x4plus",
    "RealESRGAN_x2plus": "Real-ESRGAN x2plus",
}


def list_methods() -> list[dict]:
    have = available_models()
    out = [{"id": m, "label": METHOD_LABELS[m], "available": True, "family": "klasik"}
           for m in ("nearest", "bicubic", "lanczos", "classical")]
    for k in OPENCV_SR_MODELS:
        out.append({"id": k, "label": METHOD_LABELS[k], "available": opencv_sr.opencv_sr_available(k),
                    "family": "CNN (PSNR odaklı)"})
    for k, v in SR_MODELS.items():
        if v.get("hidden"):
            continue
        out.append({"id": k, "label": METHOD_LABELS[k], "available": TORCH_AVAILABLE and have.get(k, False),
                    "family": "GAN (algısal)"})
    return out


def run_method(method: str, img: np.ndarray, scale: float, denoise: float = 0.5) -> np.ndarray:
    if method in ("nearest", "bicubic", "lanczos"):
        return classical.resize_by(img, scale, method)
    if method == "classical":
        return classical.classical_sr(img, scale, denoise=denoise * 0.8)
    if method in OPENCV_SR_MODELS:
        if not opencv_sr.opencv_sr_available(method):
            raise ModelUnavailable(f"{method} modeli yok")
        out = opencv_sr.upscale(img, method)
        sc = OPENCV_SR_MODELS[method]["scale"]
        if abs(sc - scale) > 1e-6:
            h, w = img.shape[:2]
            out = cv2.resize(out, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return out
    if method in SR_MODELS:
        up = get_upsampler(method, denoise)
        out, _ = up.enhance(img, outscale=scale)
        return out
    raise ValueError(f"Bilinmeyen yöntem: {method}")


def _label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    bar_h = 46 if sub else 30
    bar = np.full((bar_h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def _ascii(s: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return s.translate(tr)


def make_grid(tiles: list[tuple[str, str, np.ndarray]], crop_box=None, cell: int = 300, cols: int = 4) -> np.ndarray:
    """tiles: [(başlık, alt başlık, görüntü)]; hepsi aynı boyutta olmalı."""
    cells = []
    for title, sub, im in tiles:
        if crop_box:
            x, y, w, h = crop_box
            im = im[y:y + h, x:x + w]
        im = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_NEAREST if im.shape[0] < cell else cv2.INTER_AREA)
        cells.append(_label(im, _ascii(title), _ascii(sub)))
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def auto_crop_box(img: np.ndarray, frac: float = 0.35):
    """En çok detay içeren bölgeyi (Laplacian enerjisi) kırpma kutusu olarak seçer."""
    h, w = img.shape[:2]
    cs = int(min(h, w) * frac)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (cs, cs))
    sub = e[cs // 2: cs // 2 + h - cs + 1, cs // 2: cs // 2 + w - cs + 1]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return int(x), int(y), cs, cs


def run_benchmark(lr: np.ndarray, out_dir: Path, scale: float = 4, gt: np.ndarray | None = None,
                  methods: list[str] | None = None, denoise: float = 0.5, progress=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    avail = {m["id"]: m for m in list_methods()}
    methods = methods or [m for m, d in avail.items() if d["available"] and m != "nearest"]
    target_size = (gt.shape[1], gt.shape[0]) if gt is not None else (round(lr.shape[1] * scale), round(lr.shape[0] * scale))

    rows, tiles = [], []
    for i, m in enumerate(methods):
        if progress:
            progress(i / max(1, len(methods)), f"{METHOD_LABELS.get(m, m)} çalışıyor")
        row = {"method": m, "label": METHOD_LABELS.get(m, m), "family": avail.get(m, {}).get("family", "")}
        try:
            t = time.perf_counter()
            out = run_method(m, lr, scale, denoise)
            row["seconds"] = round(time.perf_counter() - t, 3)
            out = classical.resize_to(out, target_size, "bicubic")
            fname = f"{i:02d}_{m}.png"
            imwrite(out_dir / fname, out)
            row["file"] = fname
            if gt is not None:
                row["psnr"] = round(metrics.psnr(out, gt), 3)
                row["ssim"] = round(metrics.ssim(out, gt), 4)
                lp = metrics.lpips_distance(out, gt)
                if lp is not None:
                    row["lpips"] = round(lp, 4)
            row.update(metrics.no_reference_report(out))
            sub = (f"PSNR {row['psnr']:.2f} / SSIM {row['ssim']:.3f}" if gt is not None
                   else f"keskinlik {row['sharpness']:.0f}") + f" / {row['seconds']:.2f}s"
            tiles.append((row["label"], sub, out))
            row["status"] = "ok"
        except ModelUnavailable as exc:
            row["status"] = "atlandi"
            row["note"] = str(exc)
        except Exception as exc:  # bir yöntemin hatası tüm karşılaştırmayı bozmasın
            row["status"] = "hata"
            row["note"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    grid_file = None
    if tiles:
        ref = tiles[0][2]
        box = auto_crop_box(gt if gt is not None else ref)
        extra = [("Ground Truth", "referans", gt)] if gt is not None else []
        grid = make_grid(tiles + extra, crop_box=box)
        imwrite(out_dir / "comparison_grid.jpg", grid, 92)
        grid_file = "comparison_grid.jpg"
        if gt is not None:
            imwrite(out_dir / "gt.png", gt)
    if progress:
        progress(1.0, "Tamamlandı")
    return {"rows": rows, "grid": grid_file, "has_gt": gt is not None, "scale": scale,
            "lr_size": [lr.shape[1], lr.shape[0]], "target_size": list(target_size)}
