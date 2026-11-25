"""Donanım algılama ve bellek güvenliği politikası.

* PyTorch kurulu değilse uygulama yine çalışır; derin öğrenme adımları klasik
  yöntemlere düşer (fallback) ve bu durum arayüzde uyarı olarak gösterilir.
* CUDA > Apple MPS > CPU sırasıyla otomatik seçim yapılır.
* Döşeme (tile) boyutu boş VRAM'e göre seçilir; OOM durumunda yarıya indirilir.
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

from ..config import settings

try:  # PyTorch isteğe bağlıdır
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - ortamdan bağımsız
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class DeviceInfo:
    torch_available: bool
    torch_version: str | None
    device: str                 # cuda | mps | cpu
    device_name: str
    total_vram_gb: float | None
    free_vram_gb: float | None
    fp16: bool
    default_tile: int
    cpu_count: int
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def _pick_device() -> str:
    want = settings.device.lower()
    if not TORCH_AVAILABLE:
        return "cpu"
    if want in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if want in ("mps", "auto") and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def auto_tile_size(device: str, free_gb: float | None) -> int:
    """VRAM'e göre güvenli döşeme boyutu (piksel, girdi uzayında)."""
    if settings.tile_size >= 0:
        return settings.tile_size
    if device == "cuda" and free_gb is not None:
        if free_gb >= 12:
            return 0          # döşemesiz (tek geçiş)
        if free_gb >= 6:
            return 512
        if free_gb >= 3:
            return 384
        return 192
    if device == "mps":
        return 384
    return 256                # CPU: RAM tepe noktasını sınırlar


_cached: DeviceInfo | None = None


def get_device_info(refresh: bool = False) -> DeviceInfo:
    global _cached
    if _cached is not None and not refresh:
        return _cached
    dev = _pick_device()
    name, total, free = platform.processor() or platform.machine(), None, None
    if TORCH_AVAILABLE:
        if settings.cpu_threads > 0:
            torch.set_num_threads(settings.cpu_threads)
        if dev == "cuda":
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            total, free = round(total_b / 1024**3, 2), round(free_b / 1024**3, 2)
        elif dev == "mps":
            name = "Apple Silicon (MPS)"
    _cached = DeviceInfo(
        torch_available=TORCH_AVAILABLE,
        torch_version=getattr(torch, "__version__", None) if TORCH_AVAILABLE else None,
        device=dev,
        device_name=name or "CPU",
        total_vram_gb=total,
        free_vram_gb=free,
        fp16=bool(settings.fp16 and dev == "cuda"),
        default_tile=auto_tile_size(dev, free),
        cpu_count=os.cpu_count() or 1,
        platform=f"{platform.system()} {platform.release()}",
    )
    return _cached


def empty_cache() -> None:
    if not TORCH_AVAILABLE:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass
