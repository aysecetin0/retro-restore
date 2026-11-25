"""Test / doğrulama veri setini üretir (tekrar üretilebilir, seed'li).

Kullanım
--------
    python scripts/make_dataset.py                     # dahili örnek görüntülerle
    python scripts/make_dataset.py --source /yol/DIV2K_valid_HR --max-side 768
    python scripts/make_dataset.py --no-video

Üretilen yapı
-------------
dataset/
├── manifest.json            ← her bozulmuş örnek ↔ GT eşleşmesi + uygulanan bozulma zinciri
├── ground_truth/            ← temiz yüksek çözünürlüklü referanslar (HR)
├── degraded/
│   ├── bicubic_x4/          ← klasik SR kıyas ortamı: yalnızca bicubic küçültme
│   ├── realworld_x4/        ← Real-ESRGAN tarzı 2. derece bozulma (bulanıklık+gürültü+JPEG+yeniden boyut) ×2 tur
│   ├── realworld_x2/
│   ├── old_photo/           ← eski fotoğraf simülasyonu: sepya, solma, gren, çizik, toz, bulanıklık
│   └── grayscale/           ← renklendirme testi (GT = renkli orijinal)
├── videos/
│   ├── gt/                  ← temiz klip (saniyede bir "flaş + bip" senkron işaretçisi)
│   └── degraded/            ← düşük çözünürlük, S/B, gürültü, titreme, çizik, ağır sıkıştırma
├── real_old/                ← gerçek eski fotoğraflarınızı buraya koyun (GT yok)
└── results/                 ← scripts/evaluate.py çıktıları

Dahili kaynak görüntüler scikit-image örnek veri setindendir (astronaut ve rocket:
NASA, kamu malı; coffee, chelsea, camera: CC0).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrorestore.utils.io import imread, imwrite  # noqa: E402

GROUP_DESC = {
    "bicubic_x4": "Yalnızca bicubic ile 4× küçültme (SRCNN/EDSR'ın eğitildiği 'ideal' bozulma)",
    "realworld_x4": "Real-ESRGAN tarzı yüksek dereceli bozulma, 4× (bulanıklık, gürültü, JPEG, yeniden boyut ×2 tur)",
    "realworld_x2": "Real-ESRGAN tarzı yüksek dereceli bozulma, 2×",
    "old_photo": "Eski fotoğraf simülasyonu, 2×: sepya, solma, vinyet, gren, çizik, toz, bulanıklık, JPEG",
    "grayscale": "Siyah-beyaz sürüm (renklendirme testi), aynı çözünürlük",
}


# ---------------------------------------------------------------------------
# Kaynaklar
# ---------------------------------------------------------------------------
def builtin_sources() -> dict[str, np.ndarray]:
    from skimage import data
    out = {}
    for name in ("astronaut", "coffee", "chelsea", "rocket", "camera"):
        im = getattr(data, name)()
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        out[name] = im
    out["test_pattern"] = test_pattern()
    return out


def test_pattern(size: int = 512) -> np.ndarray:
    """Çözünürlük/aliasing testi: zone plate, ince çizgiler, metin, renk rampası."""
    img = np.full((size, size, 3), 235, np.uint8)
    y, x = np.mgrid[0:size // 2, 0:size // 2].astype(np.float32)
    r2 = (x - size / 4) ** 2 + (y - size / 4) ** 2
    zp = (127 + 120 * np.cos(np.pi * r2 / (size * 0.9))).astype(np.uint8)
    img[: size // 2, : size // 2] = zp[..., None]
    for i in range(0, size // 2, 6):
        cv2.line(img, (size // 2 + i, 0), (size // 2 + i, size // 4), (30, 30, 30), 1)
        cv2.line(img, (size // 2, size // 4 + i // 2), (size, size // 4 + i // 2), (60, 60, 160), 1)
    for k, s in enumerate((0.4, 0.55, 0.7, 0.9)):
        cv2.putText(img, "RetroRestore 1234", (12, size // 2 + 40 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, s, (20, 20, 20), 1, cv2.LINE_AA)
    ramp = np.linspace(0, 179, size // 2, dtype=np.uint8)
    hsv = np.stack([np.tile(ramp, (size // 2, 1)), np.full((size // 2, size // 2), 200, np.uint8),
                    np.tile(np.linspace(80, 255, size // 2, dtype=np.uint8)[:, None], (1, size // 2))], -1)
    img[size // 2:, size // 2:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def prep_gt(img: np.ndarray, max_side: int, mod: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return img[: h - h % mod, : w - w % mod]


# ---------------------------------------------------------------------------
# Bozulma operatörleri (Real-ESRGAN makalesindeki boru hattının sadeleştirilmiş hali)
# ---------------------------------------------------------------------------
def blur(img, rng, log):
    if rng.random() < 0.5:
        s = rng.uniform(0.4, 2.2)
        log.append(f"gauss_blur(σ={s:.2f})")
        return cv2.GaussianBlur(img, (0, 0), s)
    sx, sy, ang = rng.uniform(0.5, 2.5), rng.uniform(0.3, 1.2), rng.uniform(0, 180)
    log.append(f"aniso_blur(σx={sx:.2f},σy={sy:.2f},θ={ang:.0f}°)")
    k = 21
    yy, xx = np.mgrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1].astype(np.float32)
    t = np.deg2rad(ang)
    xr, yr = xx * np.cos(t) + yy * np.sin(t), -xx * np.sin(t) + yy * np.cos(t)
    ker = np.exp(-0.5 * (xr ** 2 / sx ** 2 + yr ** 2 / sy ** 2))
    return cv2.filter2D(img, -1, ker / ker.sum())


def resize(img, scale, rng, log, size=None):
    interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    name = {cv2.INTER_AREA: "area", cv2.INTER_LINEAR: "bilinear", cv2.INTER_CUBIC: "bicubic"}[interp]
    h, w = img.shape[:2]
    size = size or (max(1, round(w * scale)), max(1, round(h * scale)))
    log.append(f"resize({name},{size[0]}x{size[1]})")
    return cv2.resize(img, size, interpolation=interp)


def noise(img, rng, log, max_sigma=20):
    f = img.astype(np.float32)
    if rng.random() < 0.6:
        s = rng.uniform(2, max_sigma)
        gray = rng.random() < 0.4
        n = rng.normal(0, s, f.shape[:2] + (1,) if gray else f.shape)
        log.append(f"gauss_noise(σ={s:.1f}{',gri' if gray else ''})")
        f = f + n
    else:
        sc = rng.uniform(0.05, 2.0)
        log.append(f"poisson_noise(ölçek={sc:.2f})")
        vals = 2 ** np.ceil(np.log2(len(np.unique(img)))) * 10 / sc
        f = rng.poisson(np.clip(f, 0, 255) / 255 * vals) / vals * 255
    return np.clip(f, 0, 255).astype(np.uint8)


def jpeg(img, rng, log, lo=30, hi=90):
    q = int(rng.integers(lo, hi))
    log.append(f"jpeg(q={q})")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def realworld(gt, scale, rng):
    """İkinci derece bozulma: [blur → resize → noise → jpeg] × 2, sonunda hedef boyut."""
    log: list[str] = []
    h, w = gt.shape[:2]
    x = blur(gt, rng, log)
    x = resize(x, rng.uniform(0.5, 1.2), rng, log)
    x = noise(x, rng, log)
    x = jpeg(x, rng, log)
    # 2. tur
    if rng.random() < 0.8:
        x = blur(x, rng, log)
    x = resize(x, rng.uniform(0.6, 1.1) / scale * (w / x.shape[1]) * scale, rng, log)
    x = noise(x, rng, log, max_sigma=12)
    x = resize(x, 1, rng, log, size=(w // scale, h // scale))
    x = jpeg(x, rng, log, 30, 85)
    return x, log


def bicubic_lr(gt, scale):
    h, w = gt.shape[:2]
    return cv2.resize(gt, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC), [f"bicubic_down(×{scale})"]


def add_scratches(img, rng, n_lines=(6, 14), n_dust=(80, 200), log=None):
    out = img.copy()
    h, w = out.shape[:2]
    nl = int(rng.integers(*n_lines))
    for _ in range(nl):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        L, ang = rng.uniform(0.1, 0.6) * max(h, w), rng.uniform(0, np.pi)
        x1, y1 = int(x0 + L * np.cos(ang)), int(y0 + L * np.sin(ang))
        c = int(rng.choice([235, 245, 30]))
        cv2.line(out, (x0, y0), (x1, y1), (c, c, c), 1, cv2.LINE_AA)
    nd = int(rng.integers(*n_dust))
    for _ in range(nd):
        c = int(rng.choice([240, 20]))
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)), (c, c, c), -1)
    if log is not None:
        log.append(f"cizik({nl})+toz({nd})")
    return out


def old_photo(gt, rng, scale=2):
    log: list[str] = []
    h, w = gt.shape[:2]
    x = blur(gt, rng, log)
    x = cv2.resize(x, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    log.append(f"downscale(×{scale})")
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    tone = np.array([0.55, 0.78, 0.95], np.float32) if rng.random() < 0.7 else np.array([0.9, 0.9, 0.9], np.float32)
    log.append("sepya_ton" if tone[0] < 0.8 else "gri_ton")
    fade = rng.uniform(0.25, 0.45)
    log.append(f"solma(kontrast×{1 - fade:.2f})")
    g = g * (1 - fade) + fade * 0.55
    hh, ww = g.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    vig = 1 - 0.45 * (((xx - ww / 2) / ww) ** 2 + ((yy - hh / 2) / hh) ** 2) * 2.2
    g = g * vig
    log.append("vinyet")
    x = np.clip(g[..., None] * tone[None, None] * 255 + 25 * (1 - tone[None, None]), 0, 255).astype(np.uint8)
    x = noise(x, rng, log, max_sigma=14)
    x = add_scratches(x, rng, log=log)
    x = jpeg(x, rng, log, 45, 85)
    return x, log


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def write_wav(path: Path, seconds: float, sr: int = 44100):
    """Arka plan tonu + her saniyenin başında 1 kHz 'bip' (senkron işaretçisi)."""
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.08 * np.sin(2 * np.pi * 220 * t)
    beep = (t % 1.0) < 0.08
    sig = sig + 0.5 * np.sin(2 * np.pi * 1000 * t) * beep
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def ken_burns(src, out_w, out_h, n, i):
    h, w = src.shape[:2]
    p = i / max(1, n - 1)
    zoom = 1.0 + 0.35 * p
    cw, ch = int(w / zoom), int(w / zoom * out_h / out_w)
    ch = min(ch, h); cw = min(cw, w)
    x0 = int((w - cw) * (0.15 + 0.7 * p)); y0 = int((h - ch) * (0.5 - 0.2 * p))
    return cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (out_w, out_h), interpolation=cv2.INTER_AREA)


def make_videos(src_img, out_dir: Path, rng, seconds=6, fps=24):
    if not shutil.which("ffmpeg"):
        print("! ffmpeg yok, videolar atlandı")
        return []
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "degraded").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "gt").mkdir(parents=True); (tmp / "dg").mkdir()
    n = seconds * fps
    W, H = 640, 360
    for i in range(n):
        f = ken_burns(src_img, W, H, n, i)
        if i % fps < 2:   # bip ile aynı anda beyaz çerçeve flaşı
            cv2.rectangle(f, (0, 0), (W - 1, H - 1), (255, 255, 255), 10)
        cv2.imwrite(str(tmp / "gt" / f"{i:05d}.png"), f)
        # bozulma: küçük, S/B-sepya, titreme, gren, dikey film çizikleri
        d = cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = g * (0.62 + 0.08 * np.sin(i * 1.7) + rng.uniform(-0.04, 0.04)) + 40
        g += rng.normal(0, 9, g.shape)
        d = np.clip(g[..., None] * np.array([0.82, 0.93, 1.0]), 0, 255).astype(np.uint8)
        for _ in range(int(rng.integers(0, 3))):
            x = int(rng.integers(0, W // 2)); cv2.line(d, (x, 0), (x + int(rng.integers(-3, 4)), H // 2), (225, 225, 225), 1)
        cv2.imwrite(str(tmp / "dg" / f"{i:05d}.png"), d)
    wav = tmp / "a.wav"
    write_wav(wav, seconds)
    gt_p, dg_p = out_dir / "gt" / "clip_clean.mp4", out_dir / "degraded" / "clip_old_film.mp4"
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "gt" / "%05d.png"),
                    "-i", str(wav), *enc, "-crf", "18", "-c:a", "aac", "-shortest", str(gt_p)], check=True)
    # Eski video hissi için düşük bit hızı (blok artefaktları)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "dg" / "%05d.png"),
                    "-i", str(wav), *enc, "-b:v", "180k", "-c:a", "aac", "-b:a", "64k", "-shortest", str(dg_p)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return [{
        "name": "clip_old_film", "gt": "videos/gt/clip_clean.mp4", "degraded": "videos/degraded/clip_old_film.mp4",
        "info": f"{seconds} sn · {fps} fps · GT {W}x{H} · bozulmuş {W // 2}x{H // 2} · ses: 220 Hz + saniyede bir 1 kHz bip",
        "degradations": ["downscale(×2)", "S/B+sepya", "parlaklık titremesi", "gren(σ=9)", "dikey film çizikleri",
                         "H.264 180 kbit/s"],
        "sync_test": "Her saniyenin ilk 2 karesinde beyaz çerçeve flaşı ve aynı anda 80 ms bip vardır; "
                     "çıktıda flaş ile bip hâlâ çakışıyorsa ses–görüntü senkronu korunmuştur.",
    }]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="Kendi HR görüntü klasörünüz (ör. DIV2K_valid_HR)")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.source:
        files = sorted(p for p in a.source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))[: a.limit]
        sources = {p.stem: imread(p) for p in files}
    else:
        sources = builtin_sources()

    out = a.out
    for sub in ["ground_truth", *[f"degraded/{g}" for g in GROUP_DESC], "real_old", "results"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    items = []
    for name, src in sources.items():
        gt = prep_gt(src, a.max_side)
        gt_rel = f"ground_truth/{name}.png"
        imwrite(out / gt_rel, gt)
        is_gray_src = float(np.abs(gt[:, :, 0].astype(int) - gt[:, :, 2]).mean()) < 1
        jobs = [("bicubic_x4", lambda g: bicubic_lr(g, 4)),
                ("realworld_x4", lambda g: realworld(g, 4, rng)),
                ("realworld_x2", lambda g: realworld(g, 2, rng)),
                ("old_photo", lambda g: old_photo(g, rng))]
        if not is_gray_src and name != "test_pattern":
            jobs.append(("grayscale", lambda g: (cv2.cvtColor(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), ["to_gray"])))
        for group, fn in jobs:
            lr, log = fn(gt)
            rel = f"degraded/{group}/{name}.png"
            imwrite(out / rel, lr)
            items.append({"group": group, "name": name, "gt": gt_rel, "degraded": rel,
                          "size_gt": f"{gt.shape[1]}x{gt.shape[0]}", "size_lr": f"{lr.shape[1]}x{lr.shape[0]}",
                          "scale": round(gt.shape[1] / lr.shape[1], 3), "degradations": log})
        print(f"✔ {name}: {gt.shape[1]}x{gt.shape[0]}")

    videos = []
    if not a.no_video:
        vsrc = prep_gt(sources.get("rocket", next(iter(sources.values()))), 1280)
        videos = make_videos(vsrc, out / "videos", rng)
        print(f"✔ {len(videos)} video")

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "seed": a.seed,
                "source": str(a.source) if a.source else "scikit-image örnekleri (kamu malı / CC0)",
                "groups": GROUP_DESC, "images": items, "videos": videos}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifesto: {out / 'manifest.json'}  ({len(items)} çift)")


if __name__ == "__main__":
    main()
