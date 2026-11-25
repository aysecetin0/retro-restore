"""Komut satırı arayüzü — web arayüzü olmadan toplu işlem.

Örnekler:
    python enhance_cli.py dataset/degraded/old_photo/astronaut.png -o out.png --scale 2
    python enhance_cli.py eski_klip.mp4 -o yeni.mp4 --scale 2 --model realesr-general-x4v3
    python enhance_cli.py klasor/ -o cikti/ --color-mode colorize
"""
import argparse
import json
from pathlib import Path

from retrorestore.config import settings
from retrorestore.core.trace import PipelineTrace
from retrorestore.pipeline.image_pipeline import EnhanceOptions, enhance_image
from retrorestore.pipeline.video_pipeline import process_video
from retrorestore.utils.io import imread, imwrite


def main():
    ap = argparse.ArgumentParser(description="RetroRestore CLI")
    ap.add_argument("input", type=Path, help="görüntü, video veya klasör")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=2)
    ap.add_argument("--method", default="realesrgan", help="realesrgan|classical|lanczos|bicubic|none")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--denoise", type=float, default=0.5)
    ap.add_argument("--color-mode", default="auto", help="auto|restore|colorize|none")
    ap.add_argument("--color-strength", type=float, default=0.8)
    ap.add_argument("--scratch", action="store_true", help="çizik/toz giderme")
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--preview-seconds", type=float, default=None)
    ap.add_argument("--trace", action="store_true", help="adım izini ve ara görüntüleri kaydet")
    a = ap.parse_args()

    opts = EnhanceOptions.from_dict(dict(
        scale=a.scale, sr_method=a.method, model=a.model, denoise=a.denoise,
        color_mode=a.color_mode, color_strength=a.color_strength,
        scratch_removal=a.scratch, sharpen=a.sharpen))

    inputs = sorted(p for p in a.input.iterdir() if p.is_file()) if a.input.is_dir() else [a.input]
    for src in inputs:
        ext = src.suffix.lower()
        dst = a.output / src.name if a.input.is_dir() else a.output
        trace_dir = dst.with_suffix("").parent / (dst.stem + "_trace") if a.trace else None
        trace = PipelineTrace(snapshot_dir=trace_dir)
        if ext in settings.video_exts:
            work = dst.parent / (dst.stem + "_work")
            info = process_video(src, work, opts, trace,
                                 progress=lambda p, m: print(f"\r[{p*100:5.1f}%] {m}", end="", flush=True),
                                 preview_seconds=a.preview_seconds)
            print()
            dst.parent.mkdir(parents=True, exist_ok=True)
            (work / "result.mp4").replace(dst.with_suffix(".mp4"))
        elif ext in settings.image_exts:
            out, info = enhance_image(imread(src), opts, trace)
            imwrite(dst, out)
        else:
            continue
        print(f"✔ {src.name} → {dst}  ({trace.total_seconds:.2f} sn)")
        for w in trace.warnings:
            print(f"  ! {w}")
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace.save(trace_dir / "trace.json")
            (trace_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
