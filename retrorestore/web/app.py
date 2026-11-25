"""Flask web uygulaması ve REST API.

Uç noktalar (endpoints)
-----------------------
GET  /                         → arayüz (SPA)
GET  /api/system               → donanım, modeller, limitler (şeffaflık)
GET  /api/methods              → karşılaştırılabilir yöntemler
GET  /api/dataset              → veri seti manifestosu (GT ↔ bozulmuş eşleşmeleri)
GET  /api/report               → scripts/evaluate.py sonuçları (dataset/results)
POST /api/jobs                 → iş oluştur (dosya yükle VEYA veri setinden örnek seç)
GET  /api/jobs                 → tüm işler
GET  /api/jobs/<id>            → durum, ilerleme, adım izi (trace)
POST /api/jobs/<id>/cancel     → iptal
DELETE /api/jobs/<id>          → sil
GET  /files/jobs/<id>/<yol>    → iş klasöründeki dosyalar (sonuç, ara adımlar)
GET  /files/dataset/<yol>      → veri seti dosyaları
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .. import __version__
from ..config import settings
from ..core.device import get_device_info
from ..methods.colorize import colorizer_ready
from ..methods.realesrgan import available_models
from ..models_registry import SR_MODELS
from ..pipeline.benchmark import list_methods
from ..pipeline.image_pipeline import EnhanceOptions
from ..pipeline.video_pipeline import ffmpeg_available
from .jobs import Job, JobManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _safe_child(base: Path, rel: str) -> Path:
    """Yol geçişi (path traversal) saldırılarına karşı güvenli birleştirme."""
    p = (base / rel).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        abort(404)
    if not p.is_file():
        abort(404)
    return p


def _media_kind(name: str) -> str | None:
    ext = Path(name).suffix.lower()
    if ext in settings.image_exts:
        return "image"
    if ext in settings.video_exts:
        return "video"
    return None


def _find_gt(sample_rel: str) -> str | None:
    """Manifestoya bakarak bozulmuş örneğin ground-truth eşini bulur."""
    mf = settings.dataset_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None
    for item in data.get("images", []):
        if item.get("degraded") == sample_rel and item.get("gt"):
            return item["gt"]
    return None


def create_app() -> Flask:
    settings.ensure_dirs()
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    manager = JobManager()
    app.extensions["rr_jobs"] = manager

    # ------------------------------------------------------------ hatalar
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error=f"Dosya çok büyük (en fazla {settings.max_upload_mb} MB)."), 413

    @app.errorhandler(HTTPException)
    def http_err(e):
        if request.path.startswith("/api/"):
            return jsonify(error=e.description), e.code
        return e

    # ------------------------------------------------------------ sayfa
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # ------------------------------------------------------------ bilgi
    @app.get("/api/system")
    def system():
        info = get_device_info().as_dict()
        models = {k: {"label": v["label"], "scale": v["scale"], "params_m": v["params_m"],
                      "available": available_models().get(k, False), "notes": v.get("notes", "")}
                  for k, v in SR_MODELS.items() if not v.get("hidden")}
        return jsonify(
            version=__version__, device=info, models=models,
            colorizer_ready=colorizer_ready(), ffmpeg=ffmpeg_available(),
            defaults=EnhanceOptions().as_dict(),
            limits={"max_upload_mb": settings.max_upload_mb,
                    "max_input_pixels": settings.max_input_pixels,
                    "max_output_pixels": settings.max_output_pixels,
                    "max_video_seconds": settings.max_video_seconds,
                    "max_video_frames": settings.max_video_frames,
                    "max_video_output_side": settings.max_video_output_side},
        )

    @app.get("/api/methods")
    def methods():
        return jsonify(methods=list_methods())

    @app.get("/api/dataset")
    def dataset():
        mf = settings.dataset_dir / "manifest.json"
        data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        # Manifestoda olmayan, kullanıcının eklediği dosyalar (ör. real_old/)
        extra = []
        for sub in ("real_old",):
            d = settings.dataset_dir / sub
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and _media_kind(f.name):
                        extra.append(str(f.relative_to(settings.dataset_dir)).replace("\\", "/"))
        data["extra"] = extra
        return jsonify(data)

    @app.get("/api/report")
    def report():
        f = settings.dataset_dir / "results" / "metrics.json"
        if not f.exists():
            return jsonify(available=False)
        return jsonify(available=True, **json.loads(f.read_text(encoding="utf-8")))

    # ------------------------------------------------------------ işler
    @app.post("/api/jobs")
    def create_job():
        form = request.form.to_dict()
        kind = form.pop("kind", "enhance")
        if kind not in ("enhance", "benchmark"):
            abort(400, "Geçersiz iş türü.")
        sample = form.pop("sample", "").strip()
        upload = request.files.get("file")
        jid = manager.new_id()
        jd = manager.job_dir(jid)

        if upload and upload.filename:
            name = upload.filename
            media = _media_kind(name)
            if not media:
                abort(400, "Desteklenmeyen dosya türü.")
            jd.mkdir(parents=True, exist_ok=True)
            input_file = "input" + Path(secure_filename(name) or "x.png").suffix.lower()
            upload.save(jd / input_file)
            gt_rel = None
        elif sample:
            src = _safe_child(settings.dataset_dir, sample)
            media = _media_kind(src.name)
            if not media:
                abort(400, "Desteklenmeyen örnek.")
            jd.mkdir(parents=True, exist_ok=True)
            name = sample
            input_file = "input" + src.suffix.lower()
            shutil.copy2(src, jd / input_file)
            gt_rel = _find_gt(sample)
        else:
            abort(400, "Dosya yükleyin veya veri setinden örnek seçin.")

        if kind == "benchmark" and media != "image":
            shutil.rmtree(jd, ignore_errors=True)
            abort(400, "Yöntem karşılaştırması yalnızca görüntüler içindir.")
        if media == "video" and not ffmpeg_available():
            shutil.rmtree(jd, ignore_errors=True)
            abort(400, "Sunucuda ffmpeg yok; video işlenemez.")

        try:  # parametreleri erken doğrula → hatalı istek kuyruğa girmesin
            if kind == "enhance":
                EnhanceOptions.from_dict(form)
        except (ValueError, TypeError) as exc:
            shutil.rmtree(jd, ignore_errors=True)
            abort(400, f"Geçersiz parametre: {exc}")

        gt_file = None
        if gt_rel:
            gsrc = settings.dataset_dir / gt_rel
            if gsrc.exists():
                gt_file = "gt" + gsrc.suffix.lower()
                shutil.copy2(gsrc, jd / gt_file)

        if "preview_seconds" in form:
            try:
                form["preview_seconds"] = float(form["preview_seconds"]) or None
            except ValueError:
                form["preview_seconds"] = None
        job = manager.submit(Job(id=jid, kind=kind, media=media, input_name=name,
                                 input_file=input_file, params=form, gt_file=gt_file))
        return jsonify(job.public()), 201

    @app.get("/api/jobs")
    def jobs():
        return jsonify(jobs=manager.list())

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = manager.get(jid)
        if not j:
            abort(404, "İş bulunamadı.")
        return jsonify(j.public())

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        if not manager.cancel(jid):
            abort(404, "İş bulunamadı.")
        return jsonify(ok=True)

    @app.delete("/api/jobs/<jid>")
    def delete(jid):
        manager.delete(jid)
        return jsonify(ok=True)

    # ------------------------------------------------------------ dosyalar
    @app.get("/files/jobs/<jid>/<path:rel>")
    def job_file(jid, rel):
        if not manager.get(jid):
            abort(404)
        p = _safe_child(manager.job_dir(jid), rel)
        dl = request.args.get("download") == "1"
        name = None
        if dl:
            j = manager.get(jid)
            stem = Path(j.input_name).stem
            name = f"{secure_filename(stem) or 'sonuc'}_retrorestore{p.suffix}"
        return send_file(p, as_attachment=dl, download_name=name, conditional=True)

    @app.get("/files/dataset/<path:rel>")
    def dataset_file(rel):
        return send_file(_safe_child(settings.dataset_dir, rel), conditional=True)

    return app
