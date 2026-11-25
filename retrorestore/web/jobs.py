"""Arka plan iş kuyruğu.

* Tek işçi iş parçacığı (worker thread): GPU/RAM aynı anda tek ağır işe
  ayrılır → bellek taşması (OOM) ve thrashing önlenir. İstekler sıraya girer.
* Her işin kendi klasörü vardır: outputs/jobs/<id>/
      input.<ext>, result.<ext>, trace.json, job.json, steps/*.jpg, ...
* İlerleme ve adım izi (trace) arayüz tarafından periyodik olarak okunur.
* Süresi dolan (TTL) iş klasörleri otomatik silinir.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import settings
from ..core.device import empty_cache
from ..core.metrics import no_reference_report, psnr, ssim
from ..core.trace import PipelineTrace
from ..pipeline.benchmark import run_benchmark
from ..pipeline.image_pipeline import EnhanceOptions, enhance_image
from ..pipeline.video_pipeline import Cancelled, process_video
from ..utils.io import imread, imwrite


@dataclass
class Job:
    id: str
    kind: str                     # enhance | benchmark
    media: str                    # image | video
    input_name: str
    input_file: str
    params: dict
    gt_file: str | None = None
    status: str = "queued"        # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "Sırada"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result_file: str | None = None
    info: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    error: str | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["queue_position"] = None
        d["elapsed"] = round((self.finished or time.time()) - self.started, 2) if self.started else None
        return d


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="rr-worker", daemon=True)
        self._worker.start()
        threading.Thread(target=self._janitor, name="rr-janitor", daemon=True).start()

    # ------------------------------------------------------------------ API
    def job_dir(self, job_id: str) -> Path:
        return settings.work_dir / job_id

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    def submit(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            self.cancel_events[job.id] = threading.Event()
        self._persist(job)
        self.q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        queued.sort(key=lambda j: j.created)
        out = []
        for j in sorted(self.jobs.values(), key=lambda j: j.created, reverse=True):
            d = j.public()
            if j.status == "queued":
                d["queue_position"] = queued.index(j) + 1
            out.append(d)
        return out

    def cancel(self, job_id: str) -> bool:
        ev = self.cancel_events.get(job_id)
        job = self.jobs.get(job_id)
        if not ev or not job:
            return False
        ev.set()
        if job.status == "queued":
            job.status, job.message = "cancelled", "İptal edildi"
        return True

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    # ------------------------------------------------------------ internals
    def _persist(self, job: Job) -> None:
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job.public(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status, job.started, job.message = "running", time.time(), "Başladı"
        jd = self.job_dir(job.id)
        cancel = self.cancel_events[job.id]

        def on_trace(tr: PipelineTrace):
            job.trace = tr.to_dict()

        def progress(p: float, msg: str):
            job.progress = round(float(p), 4)
            job.message = msg

        trace = PipelineTrace(snapshot_dir=jd, on_update=on_trace)
        try:
            if job.kind == "benchmark":
                self._run_benchmark(job, jd, progress)
            elif job.media == "video":
                opts = EnhanceOptions.from_dict(job.params)
                info = process_video(jd / job.input_file, jd, opts, trace, progress, cancel,
                                     preview_seconds=job.params.get("preview_seconds") or None)
                job.result_file = "result.mp4"
                job.info = info
            else:
                self._run_image(job, jd, trace, progress)
            job.status, job.progress, job.message = "done", 1.0, "Tamamlandı"
        except Cancelled:
            job.status, job.message = "cancelled", "İptal edildi"
        except Exception as exc:
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            job.message = "Hata"
            (jd / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            job.finished = time.time()
            job.trace = trace.to_dict()
            if trace.steps:
                trace.save(jd / "trace.json")
            self._persist(job)
            empty_cache()

    def _run_image(self, job: Job, jd: Path, trace: PipelineTrace, progress) -> None:
        progress(0.05, "Görüntü okunuyor")
        img = imread(jd / job.input_file)
        opts = EnhanceOptions.from_dict(job.params)
        progress(0.15, "Boru hattı çalışıyor")
        out, info = enhance_image(img, opts, trace)
        fmt = job.params.get("output_format", "png").lower()
        fmt = fmt if fmt in ("png", "jpg", "webp") else "png"
        if out.ndim == 3 and out.shape[2] == 4 and fmt == "jpg":
            fmt = "png"
        job.result_file = f"result.{fmt}"
        with trace.step("save", "Çıktıyı kaydetme", "io", f"{fmt.upper()} olarak kaydedilir.") as st:
            imwrite(jd / job.result_file, out, quality=95)
            st.out_shape = list(out.shape)
        info["metrics_input"] = no_reference_report(img[:, :, :3])
        info["metrics_output"] = no_reference_report(out[:, :, :3])
        if job.gt_file:
            gt = imread(jd / job.gt_file)[:, :, :3]
            import cv2
            o = cv2.resize(out[:, :, :3], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            info["metrics_vs_gt"] = {"psnr": round(psnr(o, gt), 3), "ssim": round(ssim(o, gt), 4)}
        job.info = info

    def _run_benchmark(self, job: Job, jd: Path, progress) -> None:
        lr = imread(jd / job.input_file)[:, :, :3]
        gt = imread(jd / job.gt_file)[:, :, :3] if job.gt_file else None
        scale = float(job.params.get("scale", 4))
        if gt is not None:  # GT varsa ölçek GT/LR oranından türetilir
            scale = round(gt.shape[1] / lr.shape[1], 3)
        methods = job.params.get("methods") or None
        if isinstance(methods, str):
            methods = [m for m in methods.split(",") if m]
        res = run_benchmark(lr, jd / "bench", scale=scale, gt=gt, methods=methods,
                            denoise=float(job.params.get("denoise", 0.5)), progress=progress)
        job.info = res
        job.result_file = f"bench/{res['grid']}" if res.get("grid") else None

    def _janitor(self) -> None:
        while True:
            time.sleep(600)
            limit = time.time() - settings.job_ttl_hours * 3600
            for jid, job in list(self.jobs.items()):
                if job.status in ("done", "error", "cancelled") and (job.finished or 0) < limit:
                    self.delete(jid)
