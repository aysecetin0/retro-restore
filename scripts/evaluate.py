"""Veri seti üzerinde toplu nicel değerlendirme.

    python scripts/evaluate.py                          # tüm gruplar, tüm uygun yöntemler
    python scripts/evaluate.py --groups realworld_x4 bicubic_x4
    python scripts/evaluate.py --methods bicubic lanczos realesr-general-x4v3

Çıktılar (dataset/results/):
    metrics.csv      — her görüntü × yöntem satırı
    metrics.json     — grup/yöntem ortalamaları (web arayüzü bunu okur)
    summary.md       — rapora yapıştırılabilir Markdown tablo
    grid_<grup>_<ad>.jpg — etiketli karşılaştırma ızgaraları
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrorestore.config import settings  # noqa: E402
from retrorestore.core.device import get_device_info  # noqa: E402
from retrorestore.pipeline.benchmark import METHOD_LABELS, list_methods, run_benchmark  # noqa: E402
from retrorestore.utils.io import imread  # noqa: E402

SR_GROUPS = ("bicubic_x4", "realworld_x4", "realworld_x2", "old_photo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(SR_GROUPS))
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--grids", type=int, default=2, help="grup başına kaydedilecek ızgara sayısı")
    a = ap.parse_args()

    ds = settings.dataset_dir
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    out = ds / "results"
    out.mkdir(exist_ok=True)
    methods = a.methods or [m["id"] for m in list_methods() if m["available"] and m["id"] != "nearest"]
    print("Yöntemler:", ", ".join(methods))

    rows, grids = [], []
    grid_count = defaultdict(int)
    items = [i for i in manifest["images"] if i["group"] in a.groups]
    for k, it in enumerate(items, 1):
        lr = imread(ds / it["degraded"])[:, :, :3]
        gt = imread(ds / it["gt"])[:, :, :3]
        tmp = out / "_tmp"
        res = run_benchmark(lr, tmp, scale=it["scale"], gt=gt, methods=methods)
        for r in res["rows"]:
            if r.get("status") == "ok":
                rows.append({"group": it["group"], "image": it["name"], "method": r["method"],
                             "label": r["label"], "psnr": r["psnr"], "ssim": r["ssim"],
                             "lpips": r.get("lpips"), "seconds": r["seconds"]})
        if res.get("grid") and grid_count[it["group"]] < a.grids:
            name = f"grid_{it['group']}_{it['name']}.jpg"
            (tmp / res["grid"]).replace(out / name)
            grids.append(name)
            grid_count[it["group"]] += 1
        print(f"[{k}/{len(items)}] {it['group']}/{it['name']} ✓")
    import shutil
    shutil.rmtree(out / "_tmp", ignore_errors=True)

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["group"], r["method"])].append(r)
    summary = []
    for (g, m), rs in agg.items():
        lp = [r["lpips"] for r in rs if r["lpips"] is not None]
        summary.append({"group": g, "method": m, "label": METHOD_LABELS.get(m, m), "n": len(rs),
                        "psnr": round(float(np.mean([r["psnr"] for r in rs])), 3),
                        "ssim": round(float(np.mean([r["ssim"] for r in rs])), 4),
                        "lpips": round(float(np.mean(lp)), 4) if lp else None,
                        "seconds": round(float(np.mean([r["seconds"] for r in rs])), 4)})
    order = {g: i for i, g in enumerate(SR_GROUPS)}
    summary.sort(key=lambda s: (order.get(s["group"], 99), -s["psnr"]))
    info = get_device_info()
    (out / "metrics.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M"), "device": f"{info.device} ({info.device_name})",
        "n_images": len(items), "methods": methods, "summary": summary, "grids": grids,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["| Grup | Yöntem | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Süre (sn) |", "|---|---|---:|---:|---:|---:|"]
    for s in summary:
        md.append(f"| {s['group']} | {s['label']} | {s['psnr']:.2f} | {s['ssim']:.4f} | "
                  f"{s['lpips'] if s['lpips'] is not None else '—'} | {s['seconds']:.3f} |")
    (out / "summary.md").write_text(
        f"# Değerlendirme sonuçları\n\nCihaz: {info.device} · {time.strftime('%Y-%m-%d %H:%M')}\n\n" + "\n".join(md) + "\n",
        encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
