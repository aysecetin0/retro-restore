"""Önceden eğitilmiş model ağırlıklarını models/ klasörüne indirir.

    python scripts/download_models.py                 # Real-ESRGAN (önerilen set) + renklendirme
    python scripts/download_models.py --all           # + EDSR / FSRCNN karşılaştırma modelleri
    python scripts/download_models.py --only realesr-general-x4v3
    python scripts/download_models.py --list

Bir adres çalışmazsa sıradaki yedek denenir. Hepsi başarısız olursa dosyayı
tarayıcıdan indirip aynı adla models/ klasörüne koyabilirsiniz.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrorestore.config import settings  # noqa: E402
from retrorestore.models_registry import COLORIZATION_FILES, OPENCV_SR_MODELS, SR_MODELS  # noqa: E402

DEFAULT_SR = ["realesr-general-x4v3", "realesr-general-wdn-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x2plus"]


def download(urls: list[str], dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"  ✓ zaten var: {dst.name}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for url in urls:
        try:
            print(f"  ↓ {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "RetroRestore/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r    {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if tmp.stat().st_size < 1024:   # LFS işaretçisi veya hata sayfası
                raise IOError("dosya çok küçük")
            tmp.replace(dst)
            print(f"  ✓ {dst.name}")
            return True
        except Exception as exc:
            print(f"\n  ✗ başarısız: {exc}")
            tmp.unlink(missing_ok=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="EDSR/FSRCNN dahil her şey")
    ap.add_argument("--only", nargs="*", help="yalnızca bu model adları")
    ap.add_argument("--no-colorization", action="store_true")
    ap.add_argument("--colorization", action="store_true", help="yalnızca renklendirme modeli")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, v in {**SR_MODELS, **OPENCV_SR_MODELS}.items():
            print(f"{k:28s} {v['filename']:36s} {v.get('label', '')}")
        print(f"{'colorization':28s} {'(3 dosya)':36s} Zhang et al. 2016")
        return

    md = settings.models_dir
    failed = []
    jobs: list[tuple[str, list[str], Path]] = []
    if not a.colorization:
        names = a.only or DEFAULT_SR
        for n in names:
            spec = SR_MODELS.get(n) or OPENCV_SR_MODELS.get(n)
            if spec:
                jobs.append((n, spec["urls"], md / spec["filename"]))
        if a.all:
            jobs += [(k, v["urls"], md / v["filename"]) for k, v in OPENCV_SR_MODELS.items()]
    if (a.colorization or not a.no_colorization) and not a.only:
        jobs += [(f"colorization/{k}", v["urls"], md / v["filename"]) for k, v in COLORIZATION_FILES.items()]

    for name, urls, dst in jobs:
        print(f"[{name}]")
        if not download(urls, dst):
            failed.append((name, dst.name, urls[0]))

    print("\nÖzet:")
    if failed:
        print("Aşağıdakiler indirilemedi — tarayıcıdan indirip models/ klasörüne koyun:")
        for n, f, u in failed:
            print(f"  - {f}  ←  {u}")
        sys.exit(1)
    print("Tüm modeller hazır ✓")


if __name__ == "__main__":
    main()
