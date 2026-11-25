/* RetroRestore — arayüz mantığı (framework'süz, derleme gerektirmez) */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const IMG_EXT = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
const VID_EXT = /\.(mp4|mov|avi|mkv|webm|m4v|mpe?g)$/i;

const state = { system: null, dataset: null, file: null, sample: "", media: null, jobId: null, poll: null };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}
const jobUrl = (id, rel) => `/files/jobs/${id}/${rel}`;
const dsUrl = (rel) => `/files/dataset/${rel.split("/").map(encodeURIComponent).join("/")}`;
const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "—");

/* ------------------------------------------------------------------ sekmeler */
$$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "history") loadHistory();
  if (name === "compare") loadReport();
}

/* ------------------------------------------------------------------ sistem */
async function loadSystem() {
  const s = await api("/api/system");
  state.system = s;
  const d = s.device;
  const badge = $("#sysBadge");
  badge.textContent = d.torch_available ? `${d.device.toUpperCase()} · ${d.device_name}` : "CPU · klasik mod (PyTorch yok)";
  badge.classList.toggle("gpu", d.device !== "cpu");

  const sel = $("#modelSelect");
  for (const [k, m] of Object.entries(s.models)) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = `${m.label}${m.available ? "" : " — indirilmedi"}`;
    o.disabled = !m.available;
    sel.appendChild(o);
  }

  $("#sysView").innerHTML = `<div class="kv">
    <div><b>${esc(d.device.toUpperCase())}</b><span>Çıkarım cihazı</span></div>
    <div><b>${esc(d.device_name)}</b><span>Donanım</span></div>
    <div><b>${d.total_vram_gb ?? "—"} GB</b><span>VRAM (toplam)</span></div>
    <div><b>${d.fp16 ? "Açık" : "Kapalı"}</b><span>FP16 yarı hassasiyet</span></div>
    <div><b>${d.default_tile || "döşemesiz"}</b><span>Otomatik döşeme (tile)</span></div>
    <div><b>${d.torch_version ?? "yok"}</b><span>PyTorch</span></div>
    <div><b>${s.ffmpeg ? "Var" : "Yok"}</b><span>FFmpeg (video)</span></div>
    <div><b>${s.colorizer_ready ? "Hazır" : "İndirilmedi"}</b><span>Renklendirme modeli</span></div>
    <div><b>${s.limits.max_upload_mb} MB</b><span>Maks. yükleme</span></div>
    <div><b>${(s.limits.max_input_pixels / 1e6).toFixed(1)} MP</b><span>Maks. girdi</span></div>
    <div><b>${s.limits.max_video_seconds} sn</b><span>Maks. video süresi</span></div>
    <div><b>${s.limits.max_video_output_side}px</b><span>Maks. video kenarı</span></div>
  </div>`;
  $("#modelsView").innerHTML = `<table class="tbl"><thead><tr><th>Model</th><th>Ölçek</th><th>Param.</th><th>Durum</th></tr></thead><tbody>${
    Object.entries(s.models).map(([k, m]) => `<tr><td><b>${esc(m.label)}</b><br><small class="muted">${esc(m.notes)}</small></td>
      <td>${m.scale}×</td><td class="num">${m.params_m}M</td><td>${m.available ? "✔ hazır" : "✖ indirilmedi"}</td></tr>`).join("")
  }</tbody></table><p class="muted">Model indirmek için: <code>python scripts/download_models.py</code></p>`;
  if (!d.torch_available || !Object.values(s.models).some((m) => m.available)) {
    $("#dzHint").innerHTML += `<br><span style="color:var(--warn)">Real-ESRGAN hazır değil — işlem klasik yönteme düşecek (arayüzde bildirilir).</span>`;
  }
}

/* ------------------------------------------------------------------ veri seti */
async function loadDataset() {
  const ds = await api("/api/dataset");
  state.dataset = ds;
  const all = [];
  (ds.images || []).forEach((i) => all.push({ path: i.degraded, label: `[${i.group}] ${i.name}` }));
  (ds.videos || []).forEach((v) => all.push({ path: v.degraded, label: `[video] ${v.name}` }));
  (ds.extra || []).forEach((p) => all.push({ path: p, label: `[gerçek] ${p.split("/").pop()}` }));
  for (const sel of [$("#sampleSelect"), $("#benchSample")]) {
    all.filter((x) => sel.id === "sampleSelect" || IMG_EXT.test(x.path)).forEach((x) => {
      const o = document.createElement("option");
      o.value = x.path; o.textContent = x.label; sel.appendChild(o);
    });
  }
  renderDataset(ds);
}

function renderDataset(ds) {
  const v = $("#datasetView");
  if (!ds.images && !ds.videos) {
    v.innerHTML = `<p class="error">Veri seti henüz oluşturulmamış. Çalıştırın: <code>python scripts/make_dataset.py</code></p>`;
    return;
  }
  const groups = {};
  (ds.images || []).forEach((i) => (groups[i.group] ||= []).push(i));
  const desc = ds.groups || {};
  let html = `<p><b>${(ds.images || []).length}</b> görüntü çifti · <b>${(ds.videos || []).length}</b> video · seed: <code>${esc(ds.seed)}</code> · oluşturma: ${esc(ds.created || "")}</p>`;
  for (const [g, items] of Object.entries(groups)) {
    html += `<div class="ds-group"><h3>${esc(g)} <small class="muted">— ${esc(desc[g] || "")}</small></h3><div class="ds-grid">`;
    for (const it of items) {
      html += `<div class="ds-item"><div class="pair">
          <img src="${dsUrl(it.degraded)}" alt="bozulmuş" title="Bozulmuş: ${esc(it.degraded)}">
          ${it.gt ? `<img src="${dsUrl(it.gt)}" alt="GT" title="Ground truth: ${esc(it.gt)}">` : `<div></div>`}
        </div><div class="body"><b>${esc(it.name)}</b>
          <span>${esc(it.size_lr || "")} → GT ${esc(it.size_gt || "—")}</span>
          <span class="deg">${esc((it.degradations || []).join(" → "))}</span></div>
        <div class="acts"><button class="btn small" data-use="${esc(it.degraded)}">İyileştir</button>
          <button class="btn small" data-bench="${esc(it.degraded)}">Karşılaştır</button></div></div>`;
    }
    html += `</div></div>`;
  }
  if ((ds.videos || []).length) {
    html += `<div class="ds-group"><h3>videos <small class="muted">— kısa test klipleri (ses senkron testi dahil)</small></h3><div class="ds-grid">`;
    for (const it of ds.videos) {
      html += `<div class="ds-item"><div class="pair"><video src="${dsUrl(it.degraded)}" muted controls preload="metadata"></video>
        ${it.gt ? `<video src="${dsUrl(it.gt)}" muted controls preload="metadata"></video>` : "<div></div>"}</div>
        <div class="body"><b>${esc(it.name)}</b><span>${esc(it.info || "")}</span>
        <span class="deg">${esc((it.degradations || []).join(" → "))}</span></div>
        <div class="acts"><button class="btn small" data-use="${esc(it.degraded)}">İyileştir</button></div></div>`;
    }
    html += `</div></div>`;
  }
  if ((ds.extra || []).length) {
    html += `<div class="ds-group"><h3>real_old <small class="muted">— gerçek eski fotoğraflar (GT yok)</small></h3><div class="ds-grid">${
      ds.extra.map((p) => `<div class="ds-item"><img src="${dsUrl(p)}" alt=""><div class="body"><b>${esc(p.split("/").pop())}</b></div>
      <div class="acts"><button class="btn small" data-use="${esc(p)}">İyileştir</button>
      ${IMG_EXT.test(p) ? `<button class="btn small" data-bench="${esc(p)}">Karşılaştır</button>` : ""}</div></div>`).join("")}</div></div>`;
  }
  v.innerHTML = html;
  $$("[data-use]", v).forEach((b) => b.addEventListener("click", () => { selectSample(b.dataset.use); showTab("enhance"); }));
  $$("[data-bench]", v).forEach((b) => b.addEventListener("click", () => { $("#benchSample").value = b.dataset.bench; showTab("compare"); }));
}

/* ------------------------------------------------------------------ girdi seçimi */
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && selectFile(ev.dataTransfer.files[0]));
$("#fileInput").addEventListener("change", (ev) => ev.target.files[0] && selectFile(ev.target.files[0]));
$("#sampleSelect").addEventListener("change", (ev) => ev.target.value && selectSample(ev.target.value));

function selectFile(f) {
  if (!IMG_EXT.test(f.name) && !VID_EXT.test(f.name)) return alertBox("Desteklenmeyen dosya türü.");
  state.file = f; state.sample = ""; state.media = VID_EXT.test(f.name) ? "video" : "image";
  $("#sampleSelect").value = "";
  showInputPreview(URL.createObjectURL(f), f.name, `${(f.size / 1048576).toFixed(2)} MB`);
}
function selectSample(path) {
  state.sample = path; state.file = null; state.media = VID_EXT.test(path) ? "video" : "image";
  $("#sampleSelect").value = path;
  showInputPreview(dsUrl(path), path, "veri seti örneği");
}
function showInputPreview(url, name, meta) {
  const box = $("#inputPreview");
  box.innerHTML = state.media === "video"
    ? `<video src="${url}" controls muted></video>`
    : `<img src="${url}" alt="girdi">`;
  const el = box.firstElementChild;
  const setMeta = (dim) => { box.querySelector(".meta")?.remove(); box.insertAdjacentHTML("beforeend", `<div class="meta">${esc(name)} · ${esc(meta)}${dim ? " · " + dim : ""}</div>`); };
  setMeta("");
  el.addEventListener(state.media === "video" ? "loadedmetadata" : "load", () =>
    setMeta(state.media === "video" ? `${el.videoWidth}×${el.videoHeight} · ${el.duration.toFixed(1)} sn` : `${el.naturalWidth}×${el.naturalHeight}`));
  $$(".video-only").forEach((d) => (d.style.display = state.media === "video" ? "" : "none"));
  $("#runBtn").disabled = false;
}

/* ------------------------------------------------------------------ ayar çıktıları */
const outputs = { scale: ["#scaleOut", (v) => `${v}×`], denoise: ["#denoiseOut"], color_strength: ["#colorOut"],
  scratch_sensitivity: ["#scratchOut"], sharpen: ["#sharpOut"], temporal_smoothing: ["#tempOut"] };
for (const [name, [sel, f]] of Object.entries(outputs)) {
  const inp = $(`[name=${name}]`);
  const upd = () => ($(sel).textContent = f ? f(inp.value) : Number(inp.value).toFixed(2));
  inp.addEventListener("input", upd); upd();
}
$("[name=sr_method]").addEventListener("change", (e) => { $("#modelSelect").disabled = e.target.value !== "realesrgan"; });

/* ------------------------------------------------------------------ iş gönderme */
$("#optsForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData();
  const form = $("#optsForm");
  for (const el of form.elements) {
    if (!el.name) continue;
    if (el.type === "checkbox") fd.append(el.name, el.checked ? "true" : "false");
    else fd.append(el.name, el.value);
  }
  fd.append("kind", "enhance");
  if (state.file) fd.append("file", state.file); else fd.append("sample", state.sample);
  $("#runBtn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    startPolling(job.id);
  } catch (e) { alertBox(e.message); }
  finally { $("#runBtn").disabled = false; }
});

function alertBox(msg) {
  $("#jobCard").classList.remove("hidden");
  $("#warnings").innerHTML = `<div class="error">${esc(msg)}</div>`;
}

function startPolling(id) {
  state.jobId = id;
  clearInterval(state.poll);
  $("#jobCard").classList.remove("hidden");
  $("#resultView").innerHTML = ""; $("#metricsView").innerHTML = ""; $("#traceView").innerHTML = ""; $("#warnings").innerHTML = "";
  $("#downloadBtn").classList.add("hidden"); $("#cancelBtn").classList.remove("hidden");
  $("#jobCard").scrollIntoView({ behavior: "smooth" });
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${id}`);
      renderJob(j);
      if (["done", "error", "cancelled"].includes(j.status)) clearInterval(state.poll);
    } catch (e) { clearInterval(state.poll); }
  };
  tick(); state.poll = setInterval(tick, 1000);
}
$("#cancelBtn").addEventListener("click", () => state.jobId && api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }));

function renderJob(j) {
  $("#progBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  const st = { queued: "Sırada", running: "Çalışıyor", done: "Tamamlandı", error: "Hata", cancelled: "İptal edildi" }[j.status];
  $("#progMsg").textContent = `${st} · ${j.message || ""}${j.elapsed ? ` · ${j.elapsed} sn` : ""}`;
  const warns = (j.trace?.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
  $("#warnings").innerHTML = warns + (j.error ? `<div class="error">${esc(j.error)}</div>` : "");
  renderTrace(j);
  if (j.status !== "done") return;
  $("#cancelBtn").classList.add("hidden");
  const dl = $("#downloadBtn");
  dl.href = jobUrl(j.id, j.result_file) + "?download=1"; dl.classList.remove("hidden");
  if ($("#resultView").dataset.job === j.id) return;   // bir kez çiz
  $("#resultView").dataset.job = j.id;
  if (j.media === "video") renderVideoResult(j); else renderImageResult(j);
  renderMetrics(j);
}

function renderTrace(j) {
  const steps = j.trace?.steps || [];
  $("#traceView").innerHTML = steps.map((s, i) => `<li class="s-${s.status}">
    ${s.snapshot ? `<img src="${jobUrl(j.id, s.snapshot)}" alt="" data-zoom>` : `<div class="noimg">${esc(s.stage)}</div>`}
    <div><div class="t"><span class="stage ${esc(s.stage)}">${esc(s.stage)}</span><b>${i + 1}. ${esc(s.title)}</b>
      ${s.status !== "ok" ? `<span class="stage" style="background:var(--warn)">${esc(s.status)}</span>` : ""}
      <span class="sec">${fmt(s.seconds, 3)} sn</span></div>
      ${s.description ? `<div class="desc">${esc(s.description)}</div>` : ""}
      ${s.note ? `<div class="note">${esc(s.note)}</div>` : ""}
      <div class="params">${esc(Object.entries(s.params || {}).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ·  "))}
      ${s.in_shape ? `<br>girdi ${esc(s.in_shape.join("×"))}` : ""}${s.out_shape ? ` → çıktı ${esc(s.out_shape.join("×"))}` : ""}</div>
    </div></li>`).join("") + (j.trace?.total_seconds ? `<li><div class="noimg">toplam</div><div><b>Toplam süre: ${fmt(j.trace.total_seconds, 2)} sn</b></div></li>` : "");
}

function renderImageResult(j) {
  const before = jobUrl(j.id, j.input_file), after = jobUrl(j.id, j.result_file);
  $("#resultView").innerHTML = `
    <div class="view-tools"><span class="muted">Kaydırıcıyı sürükleyin · sol: orijinal (aynı boyuta büyütülmüş, piksel görünümü) · sağ: iyileştirilmiş</span></div>
    <div class="compare" id="cmp">
      <img src="${after}" alt="sonra">
      <div class="before"><img src="${before}" alt="önce"></div>
      <div class="handle"></div><span class="lbl l">Önce</span><span class="lbl r">Sonra</span>
      <input type="range" min="0" max="100" value="50" aria-label="karşılaştırma">
    </div>
    <div class="view-tools"><a class="btn small" href="${before}" target="_blank">Orijinali aç</a>
      <a class="btn small" href="${after}" target="_blank">Sonucu tam boyut aç</a>
      ${j.gt_file ? `<a class="btn small" href="${jobUrl(j.id, j.gt_file)}" target="_blank">Ground truth</a>` : ""}</div>`;
  const cmp = $("#cmp"), r = $("input", cmp);
  const set = () => { $(".before", cmp).style.clipPath = `inset(0 ${100 - r.value}% 0 0)`; $(".handle", cmp).style.left = `${r.value}%`; };
  r.addEventListener("input", set); set();
}

function renderVideoResult(j) {
  const prev = jobUrl(j.id, "input_preview.mp4"), out = jobUrl(j.id, j.result_file);
  $("#resultView").innerHTML = `<div class="side">
      <figure><video id="vA" src="${prev}" controls muted></video><figcaption>Önce (orijinal)</figcaption></figure>
      <figure><video id="vB" src="${out}" controls></video><figcaption>Sonra (iyileştirilmiş, sesli)</figcaption></figure>
    </div><div class="view-tools"><button class="btn small" id="syncPlay">▶ İkisini senkron oynat</button></div>
    <h3>Örnek kareler <small class="muted">(ilk · orta · son)</small></h3><div id="frameSamples" class="side"></div>`;
  const a = $("#vA"), b = $("#vB");
  $("#syncPlay").onclick = () => { a.currentTime = b.currentTime = 0; a.play(); b.play(); };
  b.addEventListener("seeked", () => { if (Math.abs(a.currentTime - b.currentTime) > 0.1) a.currentTime = b.currentTime; });
  const n = j.info?.frames || 1;
  const ids = [...new Set([0, Math.floor(n / 2), n - 1])];
  $("#frameSamples").innerHTML = ids.map((i) => {
    const k = String(i).padStart(6, "0");
    return `<figure><img src="${jobUrl(j.id, `frame_samples/in_${k}.jpg`)}" data-zoom><figcaption>Kare ${i} — önce</figcaption></figure>
            <figure><img src="${jobUrl(j.id, `frame_samples/out_${k}.jpg`)}" data-zoom><figcaption>Kare ${i} — sonra</figcaption></figure>`;
  }).join("");
}

function renderMetrics(j) {
  const i = j.info || {};
  const cards = [];
  if (j.media === "video") {
    const v = i.video || {};
    cards.push([`${v.width}×${v.height}`, "Girdi"], [`${i.output?.width}×${i.output?.height}`, "Çıktı"],
      [`${v.fps_float} fps`, "Kare hızı (korundu)"], [i.frames, "İşlenen kare"],
      [`${fmt(i.sec_per_frame, 3)} sn`, "Kare başına süre"], [v.has_audio ? "Evet (AAC)" : "Yok", "Ses"],
      [`${i.processed_seconds} sn`, "İşlenen süre"], [i.color_decision, "Renk kararı"], [i.sr?.model ?? "—", "SR modeli"]);
  } else {
    const a = i.metrics_input || {}, b = i.metrics_output || {};
    cards.push([`${i.input_size?.join("×")}`, "Girdi"], [`${i.output_size?.join("×")}`, "Çıktı"],
      [`${i.effective_scale}×`, "Etkin ölçek"], [i.sr?.model ?? "—", "SR modeli"], [i.color_decision, "Renk kararı"],
      [`${fmt(a.noise_sigma)} → ${fmt(b.noise_sigma)}`, "Gürültü σ (önce→sonra)"],
      [`${fmt(a.colorfulness, 1)} → ${fmt(b.colorfulness, 1)}`, "Renk zenginliği"]);
    if (i.metrics_vs_gt) cards.push([`${i.metrics_vs_gt.psnr} dB`, "PSNR (GT'ye göre)"], [i.metrics_vs_gt.ssim, "SSIM (GT'ye göre)"]);
  }
  $("#metricsView").innerHTML = `<div class="kv">${cards.map(([v, k]) => `<div><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("")}</div>`;
}

/* ------------------------------------------------------------------ karşılaştırma */
async function loadMethods() {
  const { methods } = await api("/api/methods");
  $("#methodChecks").innerHTML = methods.map((m) => `<label class="${m.available ? "" : "off"}" title="${esc(m.family)}">
    <input type="checkbox" value="${esc(m.id)}" ${m.available && m.id !== "nearest" ? "checked" : ""} ${m.available ? "" : "disabled"}>
    ${esc(m.label)}${m.available ? "" : " (yok)"}</label>`).join("");
}

$("#benchBtn").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("kind", "benchmark");
  fd.append("scale", $("#benchScale").value);
  fd.append("methods", $$("#methodChecks input:checked").map((c) => c.value).join(","));
  const f = $("#benchFile").files[0];
  if (f) fd.append("file", f);
  else if ($("#benchSample").value) fd.append("sample", $("#benchSample").value);
  else return ($("#benchMsg").textContent = "Bir görüntü seçin.");
  $("#benchResult").innerHTML = "";
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    const t = setInterval(async () => {
      const j = await api(`/api/jobs/${job.id}`);
      $("#benchBar").style.width = `${Math.round(j.progress * 100)}%`;
      $("#benchMsg").textContent = j.error || j.message;
      if (["done", "error", "cancelled"].includes(j.status)) { clearInterval(t); if (j.status === "done") renderBench(j); }
    }, 800);
  } catch (e) { $("#benchMsg").textContent = e.message; }
});

function renderBench(j) {
  const r = j.info, rows = r.rows.filter((x) => x.status === "ok");
  const best = (k, hi = true) => rows.length ? rows.reduce((a, b) => ((hi ? b[k] > a[k] : b[k] < a[k]) ? b : a))[k] : null;
  const bp = best("psnr"), bs = best("ssim"), bl = best("lpips", false);
  const cell = (v, b, d) => `<td class="num ${v != null && v === b ? "best" : ""}">${v != null ? fmt(v, d) : "—"}</td>`;
  $("#benchResult").innerHTML = `
    <p class="muted">LR ${r.lr_size.join("×")} → ${r.target_size.join("×")} (${fmt(r.scale, 2)}×) ·
      ${r.has_gt ? "Ground truth mevcut: PSNR/SSIM ↑ daha iyi, LPIPS ↓ daha iyi." : "GT yok: referanssız ölçüler gösteriliyor."}</p>
    <table class="tbl"><thead><tr><th>Yöntem</th><th>Aile</th>${r.has_gt ? "<th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th>" : ""}
      <th>Keskinlik</th><th>Gürültü σ</th><th>Süre (sn)</th><th></th></tr></thead><tbody>
    ${r.rows.map((x) => x.status !== "ok" ? `<tr><td>${esc(x.label)}</td><td colspan="8" class="muted">${esc(x.status)}: ${esc(x.note)}</td></tr>` :
      `<tr><td><b>${esc(x.label)}</b></td><td>${esc(x.family)}</td>
      ${r.has_gt ? cell(x.psnr, bp, 2) + cell(x.ssim, bs, 4) + cell(x.lpips, bl, 4) : ""}
      <td class="num">${fmt(x.sharpness, 0)}</td><td class="num">${fmt(x.noise_sigma, 2)}</td><td class="num">${fmt(x.seconds, 3)}</td>
      <td><a href="${jobUrl(j.id, "bench/" + x.file)}" target="_blank">aç</a></td></tr>`).join("")}
    </tbody></table>
    ${r.grid ? `<h3>Detay kırpması (en çok doku içeren bölge)</h3><img class="grid-img" src="${jobUrl(j.id, "bench/" + r.grid)}" data-zoom>
    <p><a class="btn small" href="${jobUrl(j.id, "bench/" + r.grid)}?download=1">Izgarayı indir (rapor için)</a></p>` : ""}
    <p class="muted">Not: GAN tabanlı yöntemler (Real-ESRGAN) algısal kaliteyi hedeflediği için PSNR'da bicubic/EDSR'dan düşük
      çıkabilir; gerçek dünya bozulmalarında görsel kalite ve LPIPS'te öne geçer. Bu, sunumda "algı–bozulma ödünleşimi"
      (perception–distortion trade-off, Blau &amp; Michaeli 2018) olarak anlatılmalıdır.</p>`;
}

async function loadReport() {
  const r = await api("/api/report");
  const v = $("#reportView");
  if (!r.available) {
    v.innerHTML = `Henüz toplu değerlendirme yok. Çalıştırın: <code>python scripts/evaluate.py</code> — sonuçlar
      <code>dataset/results/</code> klasörüne (CSV, JSON, Markdown tablo ve ızgaralar) yazılır.`;
    return;
  }
  const s = r.summary || [];
  v.innerHTML = `<p>${esc(r.created || "")} · ${r.n_images} görüntü · ${esc(r.device || "")}</p>
    <table class="tbl"><thead><tr><th>Grup</th><th>Yöntem</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>Süre (sn) ↓</th><th>n</th></tr></thead><tbody>
    ${s.map((x) => `<tr><td>${esc(x.group)}</td><td>${esc(x.label)}</td><td class="num">${fmt(x.psnr, 2)}</td><td class="num">${fmt(x.ssim, 4)}</td>
      <td class="num">${x.lpips != null ? fmt(x.lpips, 4) : "—"}</td><td class="num">${fmt(x.seconds, 3)}</td><td class="num">${x.n}</td></tr>`).join("")}
    </tbody></table>
    ${(r.grids || []).map((g) => `<img class="grid-img" src="${dsUrl("results/" + g)}" data-zoom alt="">`).join("")}`;
}

/* ------------------------------------------------------------------ geçmiş */
async function loadHistory() {
  const { jobs } = await api("/api/jobs");
  $("#historyView").innerHTML = !jobs.length ? `<p class="muted">Henüz iş yok.</p>` :
    `<table class="tbl"><thead><tr><th>ID</th><th>Tür</th><th>Girdi</th><th>Durum</th><th>Süre</th><th></th></tr></thead><tbody>${
      jobs.map((j) => `<tr><td><code>${esc(j.id)}</code></td><td>${esc(j.kind)} / ${esc(j.media)}</td><td>${esc(j.input_name)}</td>
        <td>${esc(j.status)}${j.queue_position ? ` (#${j.queue_position})` : ""}</td><td>${j.elapsed ?? "—"} sn</td>
        <td>${j.kind === "enhance" ? `<button class="btn small" data-open="${esc(j.id)}">Aç</button>` : ""}
            <a class="btn small" href="${jobUrl(j.id, "job.json")}" target="_blank">job.json</a>
            ${j.status === "done" || j.status === "error" ? `<a class="btn small" href="${jobUrl(j.id, "trace.json")}" target="_blank">trace.json</a>` : ""}</td></tr>`).join("")
    }</tbody></table>`;
  $$("[data-open]").forEach((b) => b.addEventListener("click", () => { showTab("enhance"); startPolling(b.dataset.open); }));
}

/* ------------------------------------------------------------------ büyüteç */
document.addEventListener("click", (e) => {
  if (e.target.matches("[data-zoom]")) {
    const lb = document.createElement("div");
    lb.className = "lightbox"; lb.innerHTML = `<img src="${e.target.src}">`;
    lb.onclick = () => lb.remove(); document.body.appendChild(lb);
  }
});

/* ------------------------------------------------------------------ başlat */
(async () => {
  try { await loadSystem(); } catch (e) { $("#sysBadge").textContent = "sunucuya ulaşılamadı"; }
  try { await loadDataset(); } catch (e) { console.error(e); }
  try { await loadMethods(); } catch (e) { console.error(e); }
})();
