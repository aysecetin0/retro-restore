"""Standart süper çözünürlük test setlerini indirir (büyük ölçekli değerlendirme için).

    python scripts/download_benchmarks.py            # DIV2K doğrulama seti (100 görüntü, ~430 MB)

Ardından:
    python scripts/make_dataset.py --source benchmarks/DIV2K_valid_HR --prefix div2k_ --append --no-video --max-side 768 --limit 100
    python scripts/evaluate.py --prefix div2k_

DIV2K: Agustsson & Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution:
Dataset and Study", CVPRW 2017. Yalnızca akademik araştırma amaçlı kullanım.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks"

DIV2K_URLS = [
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/data/DIV2K_valid_HR.zip",
]


def fetch(urls: list[str], dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1e6:
        print(f"✓ zaten var: {dst.name}")
        return
    for url in urls:
        try:
            print(f"↓ {url}")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "RetroRestore"}), timeout=120) as r, \
                    open(dst, "wb") as f:
                total, got = int(r.headers.get("Content-Length") or 0), 0
                while chunk := r.read(1 << 20):
                    f.write(chunk); got += len(chunk)
                    if total:
                        print(f"\r  {got / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if zipfile.is_zipfile(dst):
                return
            raise IOError("zip değil")
        except Exception as exc:
            print(f"\n✗ {exc}")
            dst.unlink(missing_ok=True)
    sys.exit("DIV2K indirilemedi. https://data.vision.ee.ethz.ch/cvl/DIV2K/ adresinden DIV2K_valid_HR.zip "
             "dosyasını indirip benchmarks/ klasörüne koyun ve tekrar çalıştırın.")


def main():
    OUT.mkdir(exist_ok=True)
    z = OUT / "DIV2K_valid_HR.zip"
    fetch(DIV2K_URLS, z)
    target = OUT / "DIV2K_valid_HR"
    if not target.exists() or len(list(target.glob("*.png"))) < 100:
        print("Zip açılıyor…")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(OUT)
    n = len(list(target.glob("*.png")))
    print(f"✓ {target}  ({n} görüntü)")


if __name__ == "__main__":
    main()
