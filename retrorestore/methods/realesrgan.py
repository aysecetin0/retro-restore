"""Real-ESRGAN çıkarım (inference) motoru.

Özellikler
----------
* Döşemeli (tiled) çıkarım + döşeme kenar dolgusu (tile_pad) → dikiş izi yok,
  bellek kullanımı döşeme boyutuyla sınırlı.
* Çıktı tensörü CPU'da biriktirilir → büyük görüntülerde VRAM taşmaz.
* CUDA'da FP16 (yarı hassasiyet) → ~2x hız, ~yarı bellek.
* OOM (bellek yetersiz) olursa döşeme boyutu yarıya indirilerek yeniden denenir.
* `realesr-general-x4v3` için DNI (Deep Network Interpolation) ile gürültü
  giderme gücü 0–1 arasında sürekli ayarlanabilir.
* Model önbelleği: aynı model tekrar tekrar diskten yüklenmez.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
import numpy as np

from ..config import settings
from ..core.device import TORCH_AVAILABLE, empty_cache, get_device_info
from ..models_registry import SR_MODELS

if TORCH_AVAILABLE:
    import torch
    from torch.nn import functional as F

    from .realesrgan_arch import RRDBNet, SRVGGNetCompact


class ModelUnavailable(RuntimeError):
    """PyTorch ya da ağırlık dosyası yok — çağıran klasik yönteme düşmeli."""


def model_path(name: str) -> Path:
    return settings.models_dir / SR_MODELS[name]["filename"]


def available_models() -> dict[str, bool]:
    return {k: model_path(k).exists() for k, v in SR_MODELS.items() if not v.get("hidden")}


def pick_model(requested: str, scale: float) -> str:
    """'auto' seçimini donanıma ve ölçeğe göre somut modele çevirir."""
    if requested != "auto":
        return requested
    info = get_device_info()
    have = available_models()
    if info.device == "cpu":
        order = ["realesr-general-x4v3", "RealESRGAN_x2plus", "RealESRGAN_x4plus"]
    elif scale <= 2:
        order = ["RealESRGAN_x2plus", "RealESRGAN_x4plus", "realesr-general-x4v3"]
    else:
        order = ["RealESRGAN_x4plus", "realesr-general-x4v3", "RealESRGAN_x2plus"]
    for name in order:
        if have.get(name):
            return name
    return order[0]


def _load_state(path: Path) -> dict:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13
        ckpt = torch.load(str(path), map_location="cpu")
    for key in ("params_ema", "params"):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


class RealESRGANer:
    def __init__(self, name: str, denoise_strength: float = 0.5, tile: int | None = None):
        if not TORCH_AVAILABLE:
            raise ModelUnavailable("PyTorch kurulu değil.")
        if name not in SR_MODELS:
            raise ValueError(f"Bilinmeyen model: {name}")
        spec = SR_MODELS[name]
        path = model_path(name)
        if not path.exists():
            raise ModelUnavailable(
                f"Ağırlık bulunamadı: {path.name}. `python scripts/download_models.py` çalıştırın.")

        info = get_device_info()
        self.name = name
        self.scale = spec["scale"]
        self.device = torch.device(info.device)
        self.half = info.fp16
        self.tile = info.default_tile if tile is None or tile < 0 else tile
        self.tile_pad = settings.tile_pad
        self.pre_pad = settings.pre_pad
        self.denoise_strength = float(denoise_strength)

        net = (RRDBNet(**spec["arch_kwargs"]) if spec["arch"] == "rrdb"
               else SRVGGNetCompact(**spec["arch_kwargs"]))
        state = _load_state(path)

        # DNI: w * (denoise model) + (1-w) * (no-denoise model)
        partner = spec.get("dni_partner")
        self.dni_used = False
        if partner and self.denoise_strength < 0.999 and model_path(partner).exists():
            w = float(np.clip(self.denoise_strength, 0, 1))
            other = _load_state(model_path(partner))
            state = {k: w * v + (1 - w) * other[k] for k, v in state.items()}
            self.dni_used = True

        net.load_state_dict(state, strict=True)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        if self.half:
            net = net.half()
        self.model = net.to(self.device)

    # ------------------------------------------------------------------
    def _to_tensor(self, img_bgr: np.ndarray):
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # CPU'da float32 tutulur (half + reflect pad CPU'da desteklenmeyebilir);
        # FP16'ya dönüşüm her döşeme GPU'ya taşınırken yapılır.
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

    @staticmethod
    def _pad(t, right: int, bottom: int):
        if right == 0 and bottom == 0:
            return t
        h, w = t.shape[-2:]
        mode = "reflect" if (right < w and bottom < h) else "replicate"
        return F.pad(t, (0, right, 0, bottom), mode=mode)

    def _run(self, t):
        dtype = torch.float16 if self.half else torch.float32
        return self.model(t.to(self.device, dtype=dtype)).float().cpu()

    def _tile_process(self, img, tile: int):
        b, c, h, w = img.shape
        s = self.scale
        out = torch.zeros((b, c, h * s, w * s), dtype=torch.float32)
        tiles_x, tiles_y = math.ceil(w / tile), math.ceil(h / tile)
        for iy in range(tiles_y):
            for ix in range(tiles_x):
                x0, y0 = ix * tile, iy * tile
                x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
                px0, py0 = max(x0 - self.tile_pad, 0), max(y0 - self.tile_pad, 0)
                px1, py1 = min(x1 + self.tile_pad, w), min(y1 + self.tile_pad, h)
                with torch.inference_mode():
                    o = self._run(img[:, :, py0:py1, px0:px1])
                ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = \
                    o[:, :, oy0:oy0 + (y1 - y0) * s, ox0:ox0 + (x1 - x0) * s]
        return out

    def _infer(self, img, tile: int):
        if tile and (img.shape[-1] > tile or img.shape[-2] > tile):
            return self._tile_process(img, tile)
        with torch.inference_mode():
            return self._run(img)

    # ------------------------------------------------------------------
    def enhance(self, img_bgr: np.ndarray, outscale: float | None = None) -> tuple[np.ndarray, dict]:
        """BGR uint8 -> BGR uint8. (çıktı, bilgi sözlüğü) döndürür."""
        h, w = img_bgr.shape[:2]
        t = self._to_tensor(img_bgr)

        # 1) Ön dolgu: kenar artefaktlarını azaltır
        t = self._pad(t, self.pre_pad, self.pre_pad)
        # 2) Mod dolgusu: x2/x1 modelleri pixel-unshuffle için çift boyut ister
        mod = {2: 2, 1: 4}.get(self.scale, 1)
        ph, pw = t.shape[-2], t.shape[-1]
        mh, mw = (mod - ph % mod) % mod, (mod - pw % mod) % mod
        t = self._pad(t, mw, mh)

        tile = self.tile
        if tile:
            tile = max(64, tile - tile % 4)
        retries = 0
        while True:
            try:
                out = self._infer(t, tile)
                break
            except RuntimeError as exc:  # torch.cuda.OutOfMemoryError da RuntimeError'dır
                if "out of memory" not in str(exc).lower() or retries >= 4:
                    raise
                empty_cache()
                tile = 512 if not tile else max(64, (tile // 2) - (tile // 2) % 4)
                retries += 1

        s = self.scale
        # Dolgular yalnızca sağ/alt kenara eklendiği için tek kırpma ikisini de kaldırır
        out = out[:, :, : h * s, : w * s]
        arr = out.squeeze(0).clamp_(0, 1).numpy().transpose(1, 2, 0)
        arr = cv2.cvtColor((arr * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        if outscale is not None and abs(outscale - s) > 1e-6:
            nw, nh = round(w * outscale), round(h * outscale)
            interp = cv2.INTER_AREA if outscale < s else cv2.INTER_LANCZOS4
            arr = cv2.resize(arr, (nw, nh), interpolation=interp)

        info = {"model": self.name, "model_scale": s, "device": str(self.device),
                "fp16": self.half, "tile": tile, "oom_retries": retries,
                "dni": self.dni_used, "denoise_strength": self.denoise_strength}
        if retries:
            self.tile = tile  # sonraki karelerde küçük döşemeyle devam et
        return arr, info


# ---------------------------------------------------------------------------
# Model önbelleği (tek seferde tek model bellekte tutulur → bellek dostu)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cache: dict[tuple, RealESRGANer] = {}


def get_upsampler(name: str, denoise_strength: float = 0.5, tile: int | None = None) -> RealESRGANer:
    ds = round(float(denoise_strength), 2) if SR_MODELS[name].get("dni_partner") else 1.0
    key = (name, ds, tile)
    with _cache_lock:
        if key not in _cache:
            _cache.clear()
            empty_cache()
            _cache[key] = RealESRGANer(name, ds, tile)
        return _cache[key]
