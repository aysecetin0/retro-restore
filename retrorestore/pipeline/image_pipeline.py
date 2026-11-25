"""Görüntü iyileştirme boru hattı.

    Girdi
      │
      ├─ 1. Ön işleme      : doğrulama, alfa ayırma, boyut/bellek limiti, renk analizi
      ├─ 2. Çizik/toz      : (ops.) morfolojik tespit + Telea inpainting
      ├─ 3. Ön denoise     : (ops.) NLM — çok gürültülü girdiler için
      ├─ 4. Süper çözünürlük: Real-ESRGAN (döşemeli) | Klasik | Bicubic | Lanczos
      ├─ 5. Renk           : restorasyon (soluk renk) | renklendirme (S/B) | yok
      ├─ 6. Son işleme     : (ops.) keskinleştirme, alfa birleştirme
      ▼
    Çıktı  +  trace.json (her adımın süresi, parametresi, ara görüntüsü)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..config import settings
from ..core.trace import PipelineTrace
from ..methods import classical
from ..methods.colorize import ColorizerUnavailable, colorizer_ready, get_colorizer
from ..methods.old_photo import remove_scratches
from ..methods.realesrgan import ModelUnavailable, get_upsampler, pick_model
from ..utils.io import merge_alpha, split_alpha

SR_METHODS = ("realesrgan", "classical", "lanczos", "bicubic", "none")
COLOR_MODES = ("auto", "restore", "colorize", "none")


@dataclass
class EnhanceOptions:
    scale: float = 2.0               # 1, 2, 3, 4
    sr_method: str = "realesrgan"    # SR_METHODS
    model: str = "auto"              # auto | realesr-general-x4v3 | RealESRGAN_x4plus | RealESRGAN_x2plus
    denoise: float = 0.5             # 0..1 gürültü giderme gücü
    pre_denoise: bool = False        # SR öncesi klasik NLM
    scratch_removal: bool = False
    scratch_sensitivity: float = 0.5
    color_mode: str = "auto"         # COLOR_MODES
    color_strength: float = 0.8      # 0..1.5
    sharpen: float = 0.0             # 0..1.5 unsharp miktarı
    tile: int = -1                   # -1 otomatik
    temporal_smoothing: float = 0.6  # yalnızca video renklendirmede

    @classmethod
    def from_dict(cls, d: dict) -> "EnhanceOptions":
        names = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in (d or {}).items():
            if k not in names or v is None or v == "":
                continue
            default = getattr(cls, k)
            if isinstance(default, bool):
                kw[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
            elif isinstance(default, int):
                kw[k] = int(float(v))
            elif isinstance(default, float):
                kw[k] = float(v)
            else:
                kw[k] = str(v)
        o = cls(**kw)
        o.validate()
        return o

    def validate(self) -> None:
        if self.sr_method not in SR_METHODS:
            raise ValueError(f"sr_method geçersiz: {self.sr_method}")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode geçersiz: {self.color_mode}")
        self.scale = float(np.clip(self.scale, 1, 4))
        self.denoise = float(np.clip(self.denoise, 0, 1))
        self.color_strength = float(np.clip(self.color_strength, 0, 1.5))
        self.sharpen = float(np.clip(self.sharpen, 0, 1.5))
        self.scratch_sensitivity = float(np.clip(self.scratch_sensitivity, 0.05, 1))
        self.temporal_smoothing = float(np.clip(self.temporal_smoothing, 0, 0.95))

    def as_dict(self) -> dict:
        return asdict(self)


class FrameState:
    """Video kareleri arasında taşınan durum (renk titremesini azaltmak için)."""

    def __init__(self):
        self.prev_ab = None
        self.color_decision: str | None = None   # ilk karede verilen karar sabitlenir


def enhance_image(img: np.ndarray, opts: EnhanceOptions, trace: PipelineTrace | None = None,
                  state: FrameState | None = None, max_input_pixels: int | None = None) -> tuple[np.ndarray, dict]:
    trace = trace or PipelineTrace(snapshot_dir=None)
    info: dict = {"options": opts.as_dict()}
    max_in = max_input_pixels or settings.max_input_pixels

    # ------------------------------------------------------------------ 1
    with trace.step("preprocess", "Ön işleme ve analiz", "preprocess",
                    "Kanal düzenleme (gri→BGR, alfa ayırma), bellek güvenliği için "
                    "piksel limiti kontrolü ve renk/gürültü analizi.", image_in=img) as st:
        bgr, alpha = split_alpha(img)
        h, w = bgr.shape[:2]
        if h * w > max_in:
            f = (max_in / (h * w)) ** 0.5
            bgr = classical.resize_by(bgr, f, "area")
            trace.warn(f"Girdi {w}x{h} çok büyük; bellek güvenliği için {bgr.shape[1]}x{bgr.shape[0]} boyutuna küçültüldü.")
        scale = opts.scale
        hh, ww = bgr.shape[:2]
        if hh * ww * scale * scale > settings.max_output_pixels:
            scale = max(1.0, (settings.max_output_pixels / (hh * ww)) ** 0.5)
            scale = float(int(scale * 2) / 2) or 1.0
            trace.warn(f"Çıktı limiti aşılıyordu; ölçek {opts.scale}x → {scale}x düşürüldü.")
        gray = classical.is_grayscale(bgr)
        from ..core.metrics import noise_sigma
        st.params = {"girdi": f"{w}x{h}", "alfa": alpha is not None,
                     "siyah_beyaz_veya_sepya": gray, "tahmini_gurultu_sigma": round(noise_sigma(bgr), 2),
                     "etkin_olcek": scale}
        trace.snapshot(st, bgr)
    info.update(input_size=[w, h], effective_scale=scale, grayscale=gray)

    # Renk kararı (video için ilk karede sabitlenir)
    decision = state.color_decision if state and state.color_decision else None
    if decision is None:
        decision = opts.color_mode
        if decision == "auto":
            decision = "colorize" if gray else "restore"
        if decision == "colorize" and not colorizer_ready():
            trace.warn("Renklendirme modeli bulunamadı; yalnızca klasik ton/kontrast iyileştirme uygulanacak.")
            decision = "restore_gray" if gray else "restore"
        if state:
            state.color_decision = decision
    info["color_decision"] = decision

    # ------------------------------------------------------------------ 2
    if opts.scratch_removal:
        with trace.step("scratch", "Çizik / toz giderme", "preprocess",
                        "Top-hat + black-hat morfolojisiyle ince kusurlar maskelenir, "
                        "Telea inpainting ile doldurulur.",
                        {"hassasiyet": opts.scratch_sensitivity}, bgr) as st:
            bgr, mask = remove_scratches(bgr, opts.scratch_sensitivity)
            st.params["maskelenen_piksel_orani_%"] = round(100 * float((mask > 0).mean()), 3)
            trace.snapshot(st, bgr)

    if decision in ("colorize", "restore_gray"):
        with trace.step("desepia", "Sepya/sararma nötrleştirme", "preprocess",
                        "Tek tonlu görüntü nötr griye çevrilir (renklendirme modeli L kanalı bekler).",
                        image_in=bgr) as st:
            bgr = classical.desepia(bgr)
            trace.snapshot(st, bgr)

    # ------------------------------------------------------------------ 3
    if opts.pre_denoise and opts.denoise > 0:
        with trace.step("pre_denoise", "Ön gürültü giderme (NLM)", "preprocess",
                        "Non-Local Means: benzer yamaların ağırlıklı ortalaması.",
                        {"güç": opts.denoise}, bgr) as st:
            bgr = classical.denoise_nlm(bgr, opts.denoise * 0.6)
            trace.snapshot(st, bgr)

    # ------------------------------------------------------------------ 4
    method = opts.sr_method
    if method == "realesrgan":
        name = pick_model(opts.model, scale)
        with trace.step("sr", "Süper çözünürlük — Real-ESRGAN", "inference",
                        "Döşemeli (tiled) derin ağ çıkarımı. Model gürültü, bulanıklık ve "
                        "sıkıştırma artefaktlarını giderirken çözünürlüğü artırır.",
                        {"model": name, "hedef_olcek": scale, "gurultu_gucu": opts.denoise}, bgr) as st:
            try:
                up = get_upsampler(name, opts.denoise, opts.tile)
                bgr, sr_info = up.enhance(bgr, outscale=scale)
                st.params.update(sr_info)
                info["sr"] = sr_info
            except ModelUnavailable as exc:
                st.status = "fallback"
                st.note = f"{exc} → Klasik yönteme (NLM + Lanczos + Unsharp) düşüldü."
                trace.warn(st.note)
                bgr = classical.classical_sr(bgr, scale, denoise=opts.denoise * 0.8)
                info["sr"] = {"model": "classical-fallback"}
            trace.snapshot(st, bgr)
    elif method == "classical":
        with trace.step("sr", "Klasik iyileştirme (NLM + Lanczos + Unsharp)", "inference",
                        "Öğrenme gerektirmeyen taban çizgisi.", {"olcek": scale, "denoise": opts.denoise}, bgr) as st:
            bgr = classical.classical_sr(bgr, scale, denoise=opts.denoise * 0.8)
            trace.snapshot(st, bgr)
        info["sr"] = {"model": "classical"}
    elif method in ("bicubic", "lanczos"):
        with trace.step("sr", f"İnterpolasyon — {method.title()}", "inference",
                        "Saf interpolasyon: yeni bilgi üretmez, gürültüyü de büyütür.",
                        {"olcek": scale}, bgr) as st:
            bgr = classical.resize_by(bgr, scale, method)
            trace.snapshot(st, bgr)
        info["sr"] = {"model": method}
    else:
        info["sr"] = {"model": "none"}
        if opts.denoise > 0:
            with trace.step("denoise", "Gürültü giderme (NLM)", "inference", "", {"güç": opts.denoise}, bgr) as st:
                bgr = classical.denoise_nlm(bgr, opts.denoise)
                trace.snapshot(st, bgr)

    # ------------------------------------------------------------------ 5
    if decision == "colorize":
        with trace.step("colorize", "Renklendirme (Zhang et al. 2016)", "postprocess",
                        "L kanalından 313 renk sınıfı olasılığı tahmin edilir, ab kanalları "
                        "tam çözünürlüklü L ile birleştirilir.",
                        {"renk_gucu": opts.color_strength,
                         "zamansal_yumusatma": opts.temporal_smoothing if state else 0}, bgr) as st:
            try:
                col = get_colorizer()
                prev = state.prev_ab if state else None
                bgr, ab = col.colorize(bgr, strength=max(opts.color_strength, 0.1) * 1.15,
                                       prev_ab=prev, temporal=opts.temporal_smoothing if state else 0)
                if state:
                    state.prev_ab = ab
            except ColorizerUnavailable as exc:
                st.status = "fallback"
                st.note = str(exc)
                trace.warn(str(exc))
            trace.snapshot(st, bgr)
    elif decision in ("restore", "restore_gray"):
        with trace.step("color", "Renk / ton restorasyonu", "postprocess",
                        "Gri-dünya beyaz dengesi → otomatik seviye → L kanalında CLAHE → doygunluk.",
                        {"güç": opts.color_strength}, bgr) as st:
            if decision == "restore_gray":
                bgr = classical.clahe_luma(classical.auto_levels(bgr), clip=1.5 + opts.color_strength)
            else:
                bgr = classical.restore_faded_colors(bgr, min(opts.color_strength, 1.0))
            trace.snapshot(st, bgr)

    # ------------------------------------------------------------------ 6
    if opts.sharpen > 0:
        with trace.step("sharpen", "Keskinleştirme (Unsharp Mask)", "postprocess", "",
                        {"miktar": opts.sharpen}, bgr) as st:
            bgr = classical.unsharp_mask(bgr, amount=opts.sharpen)
            trace.snapshot(st, bgr)

    out = merge_alpha(bgr, alpha)
    info["output_size"] = [out.shape[1], out.shape[0]]
    return out, info
