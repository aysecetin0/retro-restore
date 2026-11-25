"""Temel testler:  python -m pytest -q

PyTorch veya model ağırlıkları olmadan da geçer (klasik fallback test edilir).
"""
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from retrorestore.config import settings
from retrorestore.core import metrics
from retrorestore.core.trace import PipelineTrace
from retrorestore.methods import classical
from retrorestore.methods.old_photo import remove_scratches
from retrorestore.pipeline.image_pipeline import EnhanceOptions, enhance_image
from retrorestore.pipeline.video_pipeline import ffmpeg_available


def _img(h=96, w=128, gray=False):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w]
    im = np.stack([(x * 2) % 255, (y * 3) % 255, ((x + y) * 2) % 255], -1).astype(np.uint8)
    im = cv2.GaussianBlur(im, (0, 0), 1) + rng.integers(0, 10, im.shape, dtype=np.uint8)
    if gray:
        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return im


def test_metrics_identity():
    im = _img()
    assert metrics.psnr(im, im) == float("inf")
    assert metrics.ssim(im, im) == pytest.approx(1.0, abs=1e-6)


def test_metrics_order():
    im = _img()
    noisy = np.clip(im + np.random.default_rng(1).normal(0, 15, im.shape), 0, 255).astype(np.uint8)
    assert metrics.psnr(noisy, im) < 30
    assert metrics.noise_sigma(noisy) > metrics.noise_sigma(im)


@pytest.mark.parametrize("method", ["bicubic", "lanczos", "classical", "realesrgan", "none"])
def test_image_pipeline_shapes(method):
    im = _img()
    out, info = enhance_image(im, EnhanceOptions(scale=2, sr_method=method))
    exp = (192, 256) if method != "none" else (96, 128)
    assert out.shape[:2] == exp and out.dtype == np.uint8


def test_trace_records_steps(tmp_path):
    tr = PipelineTrace(snapshot_dir=tmp_path)
    enhance_image(_img(), EnhanceOptions(scale=2, scratch_removal=True, sharpen=0.5), tr)
    keys = [s.key for s in tr.steps]
    assert keys[0] == "preprocess" and "sr" in keys and "sharpen" in keys
    assert all((tmp_path / s.snapshot).exists() for s in tr.steps if s.snapshot)


def test_grayscale_detection_and_alpha():
    assert classical.is_grayscale(_img(gray=True))
    assert not classical.is_grayscale(_img())
    rgba = np.dstack([_img(), np.full((96, 128), 200, np.uint8)])
    out, _ = enhance_image(rgba, EnhanceOptions(scale=2, sr_method="bicubic"))
    assert out.shape == (192, 256, 4)


def test_scratch_removal_reduces_defects():
    im = cv2.GaussianBlur(_img(200, 200), (0, 0), 3)
    bad = im.copy()
    cv2.line(bad, (10, 20), (190, 170), (250, 250, 250), 1)
    fixed, mask = remove_scratches(bad, 0.6)
    assert mask.sum() > 0
    assert metrics.psnr(fixed, im) > metrics.psnr(bad, im)


def test_output_limit_reduces_scale(monkeypatch):
    monkeypatch.setattr(settings, "max_output_pixels", 100 * 100)
    tr = PipelineTrace(None)
    out, info = enhance_image(_img(), EnhanceOptions(scale=4, sr_method="bicubic"), tr)
    assert info["effective_scale"] < 4 and tr.warnings


def test_options_validation():
    o = EnhanceOptions.from_dict({"scale": "9", "denoise": "2", "scratch_removal": "true"})
    assert o.scale == 4 and o.denoise == 1 and o.scratch_removal is True
    with pytest.raises(ValueError):
        EnhanceOptions.from_dict({"sr_method": "magic"})


# --------------------------------------------------------------------- web
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "jobs")
    from retrorestore.web import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _wait(client, jid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError


def test_api_image_job(client):
    ok, buf = cv2.imencode(".png", _img())
    r = client.post("/api/jobs", data={"file": (io.BytesIO(buf.tobytes()), "eski foto.png"),
                                       "scale": "2", "sr_method": "classical"},
                    content_type="multipart/form-data")
    assert r.status_code == 201, r.get_json()
    j = _wait(client, r.get_json()["id"])
    assert j["status"] == "done", j.get("error")
    assert client.get(f"/files/jobs/{j['id']}/{j['result_file']}").status_code == 200
    assert j["trace"]["steps"]


def test_api_rejects_bad_input(client):
    r = client.post("/api/jobs", data={"file": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = client.post("/api/jobs", data={"sample": "../../etc/passwd"}, content_type="multipart/form-data")
    assert r.status_code == 404


def test_api_system(client):
    s = client.get("/api/system").get_json()
    assert "device" in s and "limits" in s


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg yok")
def test_api_video_job_keeps_audio_and_fps(client, tmp_path):
    import subprocess
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=12:duration=1.5",
                    "-f", "lavfi", "-i", "sine=frequency=1000:duration=1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    r = client.post("/api/jobs", data={"file": (open(src, "rb"), "v.mp4"), "scale": "2", "sr_method": "bicubic",
                                       "color_mode": "none"}, content_type="multipart/form-data")
    j = _wait(client, r.get_json()["id"])
    assert j["status"] == "done", j.get("error")
    out = j["info"]["output"]
    assert out["has_audio"] and out["width"] == 192 and abs(out["fps_float"] - 12) < 0.01
    assert abs(out["duration"] - 1.5) < 0.2
