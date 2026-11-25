/* RetroRestore — kullanıcı arayüzü (framework'süz, derleme gerektirmez) */
"use strict";
{
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const IMG_EXT = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
const VID_EXT = /\.(mp4|mov|avi|mkv|webm|m4v|mpe?g)$/i;
const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "—");
const jobUrl = (id, rel) => `/files/jobs/${id}/${rel}`;

const state = { file: null, media: null, jobId: null, poll: null };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

/* ---------------------------------------------------------------- sistem */
async function loadSystem() {
  const s = await api("/api/system");
  const d = s.device;
  const badge = $("#sysBadge");
  badge.textContent = d.device === "cpu" ? "CPU ile çalışıyor" : `GPU · ${d.device_name}`;
  badge.classList.toggle("gpu", d.device !== "cpu");
  for (const [k, m] of Object.entries(s.models)) {
    const o = document.createElement("option");
    o.value = k; o.textContent = m.label + (m.available ? "" : " — yüklü değil"); o.disabled = !m.available;
    $("#modelSelect").appendChild(o);
  }
  $("#dzHint").textContent += ` · en fazla ${s.limits.max_upload_mb} MB · video en fazla ${s.limits.max_video_seconds} sn`;
}

/* ---------------------------------------------------------------- girdi */
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && selectFile(ev.dataTransfer.files[0]));
$("#fileInput").addEventListener("change", (ev) => ev.target.files[0] && selectFile(ev.target.files[0]));

function selectFile(f) {
  if (!IMG_EXT.test(f.name) && !VID_EXT.test(f.name)) return showError("Desteklenmeyen dosya türü.");
  state.file = f; state.media = VID_EXT.test(f.name) ? "video" : "image";
  const url = URL.createObjectURL(f);
  const box = $("#inputPreview");
  box.innerHTML = state.media === "video" ? `<video src="${url}" controls muted></video>` : `<img src="${url}" alt="girdi">`;
  const el = box.firstElementChild;
  const meta = (dim) => { box.querySelector(".meta")?.remove(); box.insertAdjacentHTML("beforeend", `<div class="meta">${esc(f.name)} · ${(f.size / 1048576).toFixed(2)} MB${dim ? " · " + dim : ""}</div>`); };
  meta("");
  el.addEventListener(state.media === "video" ? "loadedmetadata" : "load", () =>
    meta(state.media === "video" ? `${el.videoWidth}×${el.videoHeight} · ${el.duration.toFixed(1)} sn` : `${el.naturalWidth}×${el.naturalHeight}`));
  $$(".video-only").forEach((d) => (d.style.display = state.media === "video" ? "" : "none"));
  $("#runBtn").disabled = false;
}

/* ---------------------------------------------------------------- ayarlar */
const outputs = { scale: ["#scaleOut", (v) => `${v}×`], denoise: ["#denoiseOut"], color_strength: ["#colorOut"], sharpen: ["#sharpOut"] };
for (const [name, [sel, f]] of Object.entries(outputs)) {
  const inp = $(`[name=${name}]`);
  const upd = () => ($(sel).textContent = f ? f(inp.value) : Number(inp.value).toFixed(2));
  inp.addEventListener("input", upd); upd();
}
$("[name=sr_method]").addEventListener("change", (e) => { $("#modelSelect").disabled = e.target.value !== "realesrgan"; });

/* ---------------------------------------------------------------- gönder */
$("#optsForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!state.file) return;
  const fd = new FormData();
  for (const el of $("#optsForm").elements) {
    if (!el.name) continue;
    fd.append(el.name, el.type === "checkbox" ? String(el.checked) : el.value);
  }
  fd.append("file", state.file);
  $("#runBtn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    startPolling(job.id);
  } catch (e) { showError(e.message); }
  finally { $("#runBtn").disabled = false; }
});

function showError(msg) {
  $("#jobCard").classList.remove("hidden");
  $("#warnings").innerHTML = `<div class="error">${esc(msg)}</div>`;
}

function startPolling(id) {
  state.jobId = id;
  clearInterval(state.poll);
  $("#jobCard").classList.remove("hidden");
  $("#resultView").innerHTML = ""; $("#resultView").dataset.job = ""; $("#metricsView").innerHTML = ""; $("#warnings").innerHTML = "";
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
  const st = { queued: "Sırada", running: "İşleniyor", done: "Tamamlandı", error: "Hata", cancelled: "İptal edildi" }[j.status];
  const pos = j.status === "queued" && j.queue_position ? ` (sırada ${j.queue_position}.)` : "";
  const msg = j.message && j.message !== st ? ` · ${j.message}` : "";
  $("#progMsg").textContent = `${st}${pos}${msg}${j.elapsed ? ` · ${j.elapsed} sn` : ""}`;
  const warns = (j.trace?.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
  $("#warnings").innerHTML = warns + (j.error ? `<div class="error">İşlem başarısız oldu: ${esc(j.error)}</div>` : "");
  if (j.status !== "done") return;
  $("#cancelBtn").classList.add("hidden");
  const dl = $("#downloadBtn");
  dl.href = jobUrl(j.id, j.result_file) + "?download=1"; dl.classList.remove("hidden");
  if ($("#resultView").dataset.job === j.id) return;
  $("#resultView").dataset.job = j.id;
  if (j.media === "video") renderVideo(j); else renderImage(j);
  renderSummary(j);
}

function renderImage(j) {
  const before = jobUrl(j.id, j.input_file), after = jobUrl(j.id, j.result_file);
  $("#resultView").innerHTML = `
    <p class="muted">Kaydırıcıyı sürükleyin · sol: orijinal · sağ: iyileştirilmiş</p>
    <div class="compare" id="cmp">
      <img src="${after}" alt="sonra">
      <div class="before"><img src="${before}" alt="önce"></div>
      <div class="handle"></div><span class="lbl l">Önce</span><span class="lbl r">Sonra</span>
      <input type="range" min="0" max="100" value="50" aria-label="karşılaştırma">
    </div>
    <div class="view-tools"><a class="btn small" href="${after}" target="_blank">Sonucu tam boyut aç</a></div>`;
  const cmp = $("#cmp"), r = $("input", cmp);
  const set = () => { $(".before", cmp).style.clipPath = `inset(0 ${100 - r.value}% 0 0)`; $(".handle", cmp).style.left = `${r.value}%`; };
  r.addEventListener("input", set); set();
}

function renderVideo(j) {
  const prev = jobUrl(j.id, "input_preview.mp4"), out = jobUrl(j.id, j.result_file);
  $("#resultView").innerHTML = `<div class="side">
      <figure><video id="vA" src="${prev}" controls muted></video><figcaption>Önce</figcaption></figure>
      <figure><video id="vB" src="${out}" controls></video><figcaption>Sonra (sesli)</figcaption></figure>
    </div><div class="view-tools"><button class="btn small" id="syncPlay">▶ İkisini birlikte oynat</button></div>`;
  const a = $("#vA"), b = $("#vB");
  $("#syncPlay").onclick = () => { a.currentTime = b.currentTime = 0; a.play(); b.play(); };
}

function renderSummary(j) {
  const i = j.info || {};
  const color = { colorize: "Renklendirildi", restore: "Renkler canlandırıldı", restore_gray: "Ton/kontrast iyileştirildi" }[i.color_decision] || "Değişmedi";
  const cards = j.media === "video"
    ? [[`${i.video?.width}×${i.video?.height} → ${i.output?.width}×${i.output?.height}`, "Çözünürlük"],
       [`${i.frames} kare · ${i.video?.fps_float} fps`, "İşlenen"], [i.video?.has_audio ? "Korundu" : "Yok", "Ses"], [color, "Renk"]]
    : [[`${i.input_size?.join("×")} → ${i.output_size?.join("×")}`, "Çözünürlük"], [`${i.effective_scale}×`, "Büyütme"], [color, "Renk"],
       [`${fmt(i.metrics_input?.noise_sigma, 1)} → ${fmt(i.metrics_output?.noise_sigma, 1)}`, "Gürültü seviyesi"]];
  $("#metricsView").innerHTML = `<div class="kv">${cards.map(([v, k]) => `<div><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("")}</div>`;
}

loadSystem().catch(() => { $("#sysBadge").textContent = "sunucuya ulaşılamadı"; });
}
