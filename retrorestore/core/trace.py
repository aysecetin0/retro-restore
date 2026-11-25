"""Şeffaflık katmanı: boru hattının her adımını kayıt altına alır.

Her adım için ad, açıklama, parametreler, süre, girdi/çıktı boyutu ve
(isteğe bağlı) ara görüntünün küçük bir önizlemesi saklanır. Bu kayıt hem
web arayüzündeki "Arka Plan Akışı" panelinde hem de iş klasöründeki
`trace.json` dosyasında görülebilir.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..utils.io import imwrite, make_thumbnail


@dataclass
class TraceStep:
    key: str
    title: str
    stage: str                    # preprocess | inference | postprocess | video | io
    description: str = ""
    params: dict = field(default_factory=dict)
    in_shape: list | None = None
    out_shape: list | None = None
    seconds: float = 0.0
    snapshot: str | None = None   # iş klasörüne göre göreli yol
    note: str = ""
    status: str = "ok"            # ok | skipped | fallback | error


class PipelineTrace:
    def __init__(self, snapshot_dir: Path | None = None,
                 on_update: Callable[["PipelineTrace"], None] | None = None,
                 save_snapshots: bool = True):
        self.steps: list[TraceStep] = []
        self.warnings: list[str] = []
        self.snapshot_dir = snapshot_dir
        self.save_snapshots = save_snapshots and snapshot_dir is not None
        self.on_update = on_update
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
        self._notify()

    def add(self, step: TraceStep) -> TraceStep:
        self.steps.append(step)
        self._notify()
        return step

    @contextmanager
    def step(self, key: str, title: str, stage: str, description: str = "",
             params: dict | None = None, image_in: np.ndarray | None = None):
        st = TraceStep(key=key, title=title, stage=stage, description=description,
                       params=dict(params or {}))
        if image_in is not None:
            st.in_shape = list(image_in.shape)
        t = time.perf_counter()
        try:
            yield st
        except Exception as exc:  # adım başarısızsa kaydı yine tut
            st.status = "error"
            st.note = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            st.seconds = round(time.perf_counter() - t, 4)
            self.add(st)

    def snapshot(self, st: TraceStep, img: np.ndarray) -> None:
        st.out_shape = list(img.shape)
        if not self.save_snapshots:
            return
        idx = len(self.steps) + 1
        rel = f"steps/{idx:02d}_{st.key}.jpg"
        imwrite(self.snapshot_dir / rel, make_thumbnail(img, 640), quality=88)
        st.snapshot = rel

    # ------------------------------------------------------------------
    @property
    def total_seconds(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [asdict(s) for s in self.steps],
            "warnings": list(self.warnings),
            "total_seconds": self.total_seconds,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _notify(self) -> None:
        if self.on_update:
            try:
                self.on_update(self)
            except Exception:
                pass
