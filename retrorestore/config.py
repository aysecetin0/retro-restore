"""Merkezi yapılandırma.

Tüm değerler ortam değişkenleriyle (RR_ önekiyle) geçersiz kılınabilir.
Örnek:  RR_MAX_VIDEO_SECONDS=60 python run.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"RR_{name}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on", "evet"}
    return cast(raw)


@dataclass
class Settings:
    # --- Klasörler -------------------------------------------------------
    root_dir: Path = ROOT_DIR
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", ROOT_DIR / "models")))
    dataset_dir: Path = field(default_factory=lambda: Path(_env("DATASET_DIR", ROOT_DIR / "dataset")))
    work_dir: Path = field(default_factory=lambda: Path(_env("WORK_DIR", ROOT_DIR / "outputs" / "jobs")))

    # --- Web sunucusu ----------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8000, int))
    max_upload_mb: int = field(default_factory=lambda: _env("MAX_UPLOAD_MB", 200, int))
    job_ttl_hours: float = field(default_factory=lambda: _env("JOB_TTL_HOURS", 24.0, float))

    # --- Donanım / bellek güvenliği -------------------------------------
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))          # auto | cuda | mps | cpu
    tile_size: int = field(default_factory=lambda: _env("TILE", -1, int))         # -1 = otomatik, 0 = döşemesiz
    tile_pad: int = 10
    pre_pad: int = 10
    fp16: bool = field(default_factory=lambda: _env("FP16", True, bool))          # yalnızca CUDA'da etkin
    cpu_threads: int = field(default_factory=lambda: _env("CPU_THREADS", 0, int)) # 0 = torch varsayılanı

    # --- Girdi limitleri -------------------------------------------------
    max_input_pixels: int = field(default_factory=lambda: _env("MAX_INPUT_PIXELS", 6_000_000, int))
    max_output_pixels: int = field(default_factory=lambda: _env("MAX_OUTPUT_PIXELS", 40_000_000, int))
    max_video_seconds: float = field(default_factory=lambda: _env("MAX_VIDEO_SECONDS", 120.0, float))
    max_video_frames: int = field(default_factory=lambda: _env("MAX_VIDEO_FRAMES", 3600, int))
    max_video_output_side: int = field(default_factory=lambda: _env("MAX_VIDEO_OUTPUT_SIDE", 1920, int))

    # --- Video kodlama ---------------------------------------------------
    video_crf: int = field(default_factory=lambda: _env("VIDEO_CRF", 18, int))
    video_preset: str = field(default_factory=lambda: _env("VIDEO_PRESET", "medium"))

    image_exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    video_exts: tuple = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
