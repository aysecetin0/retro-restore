"""Video iyileştirme boru hattı.

    video ─► ffprobe (fps, süre, ses?) ─► ffmpeg: CFR kare çıkarma (PNG, kayıpsız)
          ─► her kare: enhance_image(...)  (renk kararı + ab geçmişi kareler arası taşınır)
          ─► ffmpeg: H.264 kodlama + orijinal ses akışı (AAC) ─► çıktı.mp4

Ses–görüntü senkronu nasıl korunuyor?
  * Kareler, kaynağın ortalama kare hızında *sabit kare hızına (CFR)* dönüştürülerek
    çıkarılır (değişken kare hızlı telefon videolarında da güvenli).
  * Yeniden birleştirmede aynı kare hızı ve aynı kare sayısı kullanılır → görüntü
    akışının süresi kaynakla aynıdır.
  * Ses yeniden örneklenmeden, kaynak dosyadan aynı başlangıç/süre ile alınır;
    `-shortest` ile uçlardaki milisaniyelik farklar kırpılır.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

from ..config import settings
from ..core.trace import PipelineTrace, TraceStep
from ..utils.io import imread, imwrite
from .image_pipeline import EnhanceOptions, FrameState, enhance_image

ProgressCB = Callable[[float, str], None]


class Cancelled(Exception):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Komut başarısız ({p.returncode}): {' '.join(cmd[:6])}...\n{p.stderr[-1500:]}")
    return p.stdout


_encoders: str | None = None


def _has_encoder(name: str) -> bool:
    global _encoders
    if _encoders is None:
        try:
            _encoders = _run(["ffmpeg", "-hide_banner", "-encoders"])
        except Exception:
            _encoders = ""
    return f" {name} " in _encoders


def _video_codec_args() -> list[str]:
    if _has_encoder("libx264"):
        return ["-c:v", "libx264", "-preset", settings.video_preset, "-crf", str(settings.video_crf)]
    if _has_encoder("libopenh264"):
        return ["-c:v", "libopenh264", "-b:v", "8M"]
    return ["-c:v", "mpeg4", "-q:v", "2"]


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("Dosyada video akışı bulunamadı.")
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    def frac(s):
        try:
            f = Fraction(s)
            return f if f > 0 else None
        except Exception:
            return None

    fps = frac(v.get("avg_frame_rate")) or frac(v.get("r_frame_rate")) or Fraction(25)
    if fps > 120:  # bozuk üstveri koruması
        fps = frac(v.get("r_frame_rate")) or Fraction(30)
    fps = fps.limit_denominator(1001)
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "fps": f"{fps.numerator}/{fps.denominator}", "fps_float": round(float(fps), 3),
        "duration": round(duration, 3), "est_frames": int(round(duration * float(fps))),
        "video_codec": v.get("codec_name"), "has_audio": a is not None,
        "audio_codec": a.get("codec_name") if a else None, "rotation": rotation,
    }


def process_video(src: Path, job_dir: Path, opts: EnhanceOptions, trace: PipelineTrace,
                  progress: ProgressCB | None = None, cancel: threading.Event | None = None,
                  preview_seconds: float | None = None, keep_frames: bool = False) -> dict:
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg bulunamadı. Video işleme için ffmpeg/ffprobe kurulmalı.")
    progress = progress or (lambda p, m: None)
    frames_in, frames_out = job_dir / "frames_in", job_dir / "frames_out"
    for d in (frames_in, frames_out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    # ------------------------------------------------------------- 1) Probe
    with trace.step("probe", "Video analizi (ffprobe)", "video",
                    "Çözünürlük, kare hızı, süre ve ses akışı okunur.") as st:
        meta = probe(src)
        st.params = dict(meta)

    duration = meta["duration"] or settings.max_video_seconds
    limit = min(duration, settings.max_video_seconds)
    if preview_seconds:
        limit = min(limit, float(preview_seconds))
    if limit < duration - 0.05:
        trace.warn(f"Video {duration:.1f} sn; işlem süresi/bellek için ilk {limit:.1f} sn işlenecek.")

    # Çıktı çözünürlüğü sınırı (işlem süresini ve dosya boyutunu kontrol altında tutar)
    w, h = meta["width"], meta["height"]
    scale = opts.scale
    max_side = settings.max_video_output_side
    if max(w, h) * scale > max_side:
        new_scale = max(1.0, max_side / max(w, h))
        trace.warn(f"Video çıktısı {max_side}px kenar sınırını aşıyordu; ölçek {scale}x → {new_scale:.2f}x.")
        scale = new_scale
    fopts = EnhanceOptions(**{**opts.as_dict(), "scale": scale})

    # ------------------------------------------------------------- 2) Kareler
    with trace.step("extract", "Kare çıkarma (ffmpeg → PNG)", "video",
                    "Sabit kare hızında (CFR) kayıpsız PNG karelerine ayrılır.",
                    {"fps": meta["fps"], "sure_sn": round(limit, 2)}) as st:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{limit:.3f}",
              "-vf", f"fps={meta['fps']}", "-pix_fmt", "rgb24", "-start_number", "0",
              str(frames_in / "%08d.png")])
        frames = sorted(frames_in.glob("*.png"))
        if len(frames) > settings.max_video_frames:
            for f in frames[settings.max_video_frames:]:
                f.unlink()
            frames = frames[: settings.max_video_frames]
            trace.warn(f"Kare sayısı {settings.max_video_frames} ile sınırlandı.")
        if not frames:
            raise RuntimeError("Videodan kare çıkarılamadı.")
        st.params["kare_sayisi"] = len(frames)
    progress(0.05, f"{len(frames)} kare çıkarıldı")

    # ------------------------------------------------------------- 3) Kare kare işleme
    state = FrameState()
    n = len(frames)
    t0 = time.perf_counter()
    sample_idx = {0, n // 2, n - 1}
    samples_dir = job_dir / "frame_samples"
    infer_step = TraceStep(key="frames", title="Kare kare iyileştirme", stage="inference",
                           description="Her kare görüntü boru hattından geçer. İlk karenin "
                                       "ara adımları aşağıda ayrıca gösterilir.",
                           params={"kare_sayisi": n, "olcek": round(scale, 3)})
    frame_info = {}
    for i, fp in enumerate(frames):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        img = imread(fp)
        # İlk kare için ayrıntılı iz (ara görüntülerle), diğerleri için hafif iz
        sub = trace if i == 0 else PipelineTrace(snapshot_dir=None)
        out, frame_info = enhance_image(img, fopts, trace=sub, state=state)
        if i == 0:
            for wmsg in sub.warnings:
                trace.warn(wmsg)
        imwrite(frames_out / fp.name, out)
        if i in sample_idx:
            imwrite(samples_dir / f"in_{i:06d}.jpg", img, 90)
            imwrite(samples_dir / f"out_{i:06d}.jpg", out, 90)
        done = i + 1
        el = time.perf_counter() - t0
        eta = el / done * (n - done)
        progress(0.05 + 0.85 * done / n, f"Kare {done}/{n} · {el / done:.2f} sn/kare · kalan ~{eta:.0f} sn")
    infer_step.seconds = round(time.perf_counter() - t0, 3)
    infer_step.params["ort_sn_kare"] = round(infer_step.seconds / n, 3)
    infer_step.params["renk_karari"] = state.color_decision
    trace.add(infer_step)

    # ------------------------------------------------------------- 4) Birleştirme
    result = job_dir / "result.mp4"
    with trace.step("assemble", "Video birleştirme + ses (ffmpeg)", "video",
                    "Kareler aynı fps ile H.264'e kodlanır; orijinal ses akışı AAC olarak "
                    "aynı zaman aralığından eklenir → senkron korunur.",
                    {"fps": meta["fps"], "ses": meta["has_audio"]}) as st:
        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", meta["fps"],
               "-i", str(frames_out / "%08d.png")]
        if meta["has_audio"]:
            cmd += ["-t", f"{limit:.3f}", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", *_video_codec_args(),
                "-movflags", "+faststart", str(result)]
        _run(cmd)
        out_meta = probe(result)
        st.params.update({"cikti": f"{out_meta['width']}x{out_meta['height']}",
                          "cikti_sure_sn": out_meta["duration"]})
    progress(0.95, "Karşılaştırma önizlemesi hazırlanıyor")

    # Tarayıcıda yan yana oynatmak için girdinin H.264 önizlemesi
    preview = job_dir / "input_preview.mp4"
    try:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{limit:.3f}",
               "-vf", f"fps={meta['fps']},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
               *_video_codec_args(), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(preview)]
        _run(cmd)
    except Exception as exc:  # önizleme zorunlu değil
        trace.warn(f"Girdi önizlemesi oluşturulamadı: {exc}")

    if not keep_frames:
        shutil.rmtree(frames_in, ignore_errors=True)
        shutil.rmtree(frames_out, ignore_errors=True)

    return {"video": meta, "processed_seconds": round(limit, 3), "frames": n,
            "effective_scale": round(scale, 3), "output": out_meta,
            "sec_per_frame": infer_step.params["ort_sn_kare"],
            "color_decision": state.color_decision, "sr": frame_info.get("sr")}
