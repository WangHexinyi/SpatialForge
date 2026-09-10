// SpatialForge Embodied Inspector — frontend.
const I18N_MODE = {
  observation: "bilingual", agent: "bilingual", task: "bilingual",
  environment: "bilingual", training: "bilingual", debug: "bilingual", settings: "bilingual",
};
let DICT = { en: {}, zh: {} };
let LANG = localStorage.getItem("sf.lang") || "bilingual";
let STATE = null;

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

// ---------- i18n ----------
async function loadI18n() {
  const r = await fetch("/api/i18n");
  const data = await r.json();
  DICT = data.dictionary;
  renderI18n();
}
function pick(lang) {
  if (lang === "en") return DICT.en;
  if (lang === "zh-CN" || lang === "zh") return DICT.zh;
  return DICT.en; // bilingual prose base
}
function textFor(key, mode) {
  const en = DICT.en[key];
  const zh = DICT.zh[key];
  if (!en && !zh) return "[" + key + "]";
  if (mode === "long") {
    const base = pick(LANG);
    return (base[key] && (base[key].long || base[key].short)) || key;
  }
  // short / default -> bilingual combine when requested
  if (LANG === "bilingual") {
    const es = (en && en.short) || "";
    const zs = (zh && zh.short) || "";
    if (zs && es && zs !== es) return zs + " · " + es;
    return es || zs;
  }
  const base = pick(LANG);
  const e = base[key];
  return (e && (e.short || e.long)) || en?.short || key;
}
function renderI18n() {
  $$("[data-i18n]").forEach((el) => {
    const mode = el.getAttribute("data-i18n-mode") || "short";
    el.textContent = textFor(el.dataset.i18n, mode);
  });
  document.getElementById("lang-select").value = LANG;
  document.getElementById("settings-lang").value = LANG;
}
function setLang(l) {
  LANG = l;
  localStorage.setItem("sf.lang", l);
  renderI18n();
  renderAll();
}

// ---------- tabs ----------
function activateTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
}

// ---------- API ----------
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
}
async function apiGet(path) {
  const r = await fetch(path);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ---------- episode ----------
async function startEpisode() {
  const target = $("#target-input").value.trim() || "mug";
  const backend = $("#backend-select").value;
  const maxSteps = parseInt($("#set-maxsteps").value, 10) || 300;
  STATE = await apiPost("/api/episode/start", {
    target_category: target, backend, max_steps: maxSteps, seed: 0,
  });
  renderAll();
  drawFrame();
}
async function doAction(act) {
  if (!STATE || !STATE.active) return;
  STATE = await apiPost("/api/action/step", { action_type: act });
  renderAll();
  drawFrame();
}
async function doTeacher() {
  if (!STATE || !STATE.active) return;
  const t = await apiGet("/api/teacher");
  $("#v-dbg-teacher").textContent = t.planned ? (t.actions.join(" ") || "(already at view pose)") : "(" + t.detail + ")";
  $("#v-dbg-json").textContent = JSON.stringify(t, null, 2);
  drawGod();
}

// ---------- render ----------
function fmt(step, status) {
  const map = { success: textFor("common.success", "long"), failure: textFor("common.failure", "long"), running: textFor("common.running", "long") };
  return (map[status] || status) + (step !== undefined ? " · step " + step : "");
}
function renderAll() {
  if (!STATE) return;
  const s = STATE;
  const pv = s.privileged || {};
  const taskP = pv.task || {};
  const ag = s.agent_state || {};

  // chips
  $("#chip-env").textContent = s.backend || "—";
  $("#chip-task").textContent = (s.task && s.task.target_category) || "—";
  $("#chip-agent").textContent = s.terminal ? "done" : ag.last_action || "idle";

  // obs meta
  const om = s.observation_meta || {};
  $("#v-obs-frame").textContent = om.frame_id || "—";
  $("#v-obs-step").textContent = om.step + " · ts " + (om.timestamp_ms || "—");
  $("#v-obs-horizon").textContent = (om.camera_horizon_deg ?? "—") + "°";
  const cause = (s.model_input && s.model_input.observation && s.model_input.observation.cause_action) || null;
  $("#v-obs-action").textContent = cause ? (cause.action_type || "—") + (cause.success === null ? "" : cause.success ? " ✓" : " ✕") : "—";
  const histLen = (s.model_input && s.model_input.action_history || []).length;
  $("#v-obs-history").textContent = histLen + " actions (permitted)";
  const tf = $("#obs-target-flag");
  if (om.target_visible === true) { tf.textContent = "● target visible"; tf.className = "target-flag yes"; }
  else if (om.target_visible === false) { tf.textContent = "target not visible"; tf.className = "target-flag no"; }
  else { tf.textContent = ""; tf.className = "target-flag"; }

  // agent
  const pos = ag.position ? ag.position.map((x) => (x === Math.round(x) ? x : x.toFixed(2))).join(",") : "—";
  $("#v-agent-pose").textContent = pos;
  $("#v-agent-yaw").textContent = (ag.rotation_yaw_deg ?? "—") + "° / horizon " + (ag.camera_horizon_deg ?? "—") + "°";
  $("#v-agent-stance").textContent = ag.is_crouching ? "crouch 蹲" : "stand 站";
  $("#v-agent-step").textContent = ag.step_count ?? "—";
  const last = s.task && s.task.step !== undefined ? s.model_input?.action_history?.slice(-1)[0] : null;
  $("#v-agent-last").textContent = last ? (last.action_type + (last.success ? " ✓" : " ✕")) : "—";
  $("#v-agent-coll").textContent = ag.collision ? "collision" : ag.blocked ? "blocked" : "—";
  $("#v-agent-room").textContent = ag.room || "—";

  // task
  $("#v-task-instruction").textContent = (s.task && s.task.instruction) || "—";
  $("#v-task-cat").textContent = (s.task && s.task.target_category) || "—";
  $("#v-task-episode").textContent = s.task ? fmt(s.task.step, s.task.status) : "—";
  $("#v-task-step").textContent = (s.task ? s.task.step : "—") + " / " + (s.task ? s.task.max_steps : "—");
  $("#v-task-success").textContent = s.task && s.task.success !== null ? (s.task.success ? "✓ success 成功" : "✕ failure 失败") : "—";
  $("#episode-status").textContent = fmt(s.task?.step, s.task?.status) + (s.task?.success_reason ? " · " + s.task.success_reason : "");
  if (s.terminal && !($("#btn-start"))) {} 

  // debug (privileged)
  const tg = taskP.target_object_ids || [];
  $("#v-dbg-target-id").textContent = tg.length ? tg.join(",") : "—";
  const tp = (taskP.target_positions || [])[0];
  $("#v-dbg-target-pos").textContent = tp ? tp.map((x) => x.toFixed?.(2) ?? x).join(",") : "—";
  $("#v-dbg-pose").textContent = pos;
  if (s.terminal) {
    // show one summary; keep teacher separate
  }
}
function drawFrame() {
  const r = new Image();
  r.onload = () => {
    const cv = $("#obs-canvas");
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(r, 0, 0, cv.width, cv.height);
  };
  r.src = "/api/frame?t=" + Date.now();
}
function drawGod() {
  if (!$("#show-god").checked) return;
  const cv = $("#god-canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#0b1117"; ctx.fillRect(0, 0, W, H);
  if (!STATE) return;
  const pv = STATE.privileged || {};
  const ag = pv.agent_state || STATE.agent_state || {};
  const taskP = pv.task || {};
  const pos = ag.position;
  if (!pos) return;
  const px = (pos[0] + 0.5) * (W / 8), py = (pos[1] + 0.5) * (H / 8);
  // target
  const tp = (taskP.target_positions || [])[0];
  if (tp) {
    ctx.fillStyle = "#e0574f";
    ctx.beginPath();
    ctx.arc((tp[0] + 0.5) * (W / 8), (tp[1] + 0.5) * (H / 8), 5, 0, Math.PI * 2);
    ctx.fill();
  }
  // agent heading
  const yaw = (ag.rotation_yaw_deg || 0) * Math.PI / 180;
  ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + Math.sin(yaw) * 14, py - Math.cos(yaw) * 14);
  ctx.stroke();
  ctx.fillStyle = "#4aa3ff";
  ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
  $("#god-meta").textContent = "agent " + pos.map((x) => x.toFixed?.(1) ?? x).join(",") + "  yaw " + (ag.rotation_yaw_deg ?? "—") + "°" + (tp ? "  target ●" : "");
}

// ---------- teacher-rollout episode replay (researcher-only) ----------
let EPISODE = null;
async function refreshEpisodeList() {
  const data = await apiGet("/api/episodes/list");
  const sel = $("#ep-select");
  sel.innerHTML = "";
  (data.episodes || []).forEach((e) => {
    const o = document.createElement("option");
    o.value = e.episode_id;
    o.textContent = e.episode_id + " · " + e.category + " · " + (e.status || "?");
    sel.appendChild(o);
  });
}
async function loadEpisode() {
  const id = $("#ep-select").value;
  if (!id) return;
  EPISODE = await apiPost("/api/episodes/load", { episode_id: id });
  const h = EPISODE.header || {};
  $("#ep-header").textContent =
    "episode " + id + " · house " + (h.house_id || "—") +
    " · target " + (h.category || "—") +
    " · status " + (h.status || "—") + " success=" + h.success +
    (h.model_controlled ? " · model_controlled=yes" : " · teacher") +
    (h.spawn && h.spawn.initially_visible !== undefined ? " · spawn_visible=" + h.spawn.initially_visible : "") +
    (h.student_domain ? " · student_domain=" + h.student_domain + " leak=" + (h.student_leak_checked || "—") : "") +
    (h.model_terminal_reason ? " · terminal_reason=" + h.model_terminal_reason : "") +
    (h.invalid_decision_count ? " · invalid=" + h.invalid_decision_count : "");
  const tl = $("#ep-timeline");
  tl.innerHTML = "";
  (EPISODE.timeline || []).forEach((e, i) => {
    const b = document.createElement("div");
    b.className = "step" + (e.terminal ? " term" : "") + (e.authoritative_target_visible ? " vis" : "");
    b.textContent = "#" + e.idx + (e.action ? " " + e.action : " start");
    b.dataset.idx = e.idx;
    b.addEventListener("click", () => selectEpisodeStep(e.idx));
    tl.appendChild(b);
  });
  renderDecisions();
  if (EPISODE.timeline && EPISODE.timeline.length) selectEpisodeStep(0);
  else renderLive(EPISODE);
}
function renderDecisions() {
  const box = $("#ep-decisions");
  const dec = EPISODE.decisions || [];
  if (!dec.length) { box.textContent = ""; return; }
  const rows = dec.map((d) => {
    const f = d.frame || "—";
    return "<b>#" + (d.decision_idx ?? "?") + "</b> frame=" + f +
      " act=" + (d.model_action || "∅") +
      (d.invalid ? " [invalid:" + (d.parse_reason || "") + "]" : "") +
      (d.executed ? " exec" : "") + " lat=" + (d.latency_ms ?? "—") + "ms" +
      " vis=" + (d.authoritative_target_visible ? "●" : "○") +
      '<span class="raw"> ' + (d.model_raw || "").slice(0, 80) + "</span>";
  }).join("<br>");
  box.innerHTML = "<b>model decisions (researcher only):</b><br>" + rows;
}
function renderLive(EP) {
  if (!EP.in_progress) return;
  const d = EP.live_decision || {};
  const img = new Image();
  img.onload = () => {
    const cv = $("#ep-canvas"), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
  };
  img.src = "/api/episodes/frame?frame=" + encodeURIComponent(EP.live_frame || "frame_0000.png") + "&t=" + Date.now();
  $("#ep-fields").innerHTML =
    "<dt>LIVE step</dt><dd>" + (d.step ?? "—") + " decision#" + (EP.header?.decision_idx ?? "—") + "</dd>" +
    "<dt>model action</dt><dd>" + (d.model_action || "—") + (d.invalid ? " [invalid]" : "") + "</dd>" +
    "<dt>raw</dt><dd>" + (d.model_raw || "—").slice(0, 120) + "</dd>" +
    "<dt>latency</dt><dd>" + (d.latency_ms ?? "—") + "ms</dd>" +
    "<dt>target visible(priv)</dt><dd>" + (d.authoritative_target_visible ? "yes" : "no") + "</dd>";
}
async function selectEpisodeStep(idx) {
  if (!EPISODE) return;
  const st = await apiGet("/api/episodes/state?step=" + idx);
  $$("#ep-timeline .step").forEach((b) => b.classList.toggle("sel", b.dataset.idx == idx));
  const img = new Image();
  img.onload = () => {
    const cv = $("#ep-canvas"), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
  };
  img.src = "/api/episodes/frame?step=" + idx + "&t=" + Date.now();
  const f = $("#ep-fields");
  const ag = st.agent || {};
  const pos = ag.position ? ag.position.map((x) => (x === Math.round(x) ? x : x.toFixed(2))).join(",") : "—";
  const h = st.header || {};
  const tp = h.target_positions && h.target_positions[0] ? h.target_positions[0].map((x)=>x.toFixed?.(2)??x).join(",") : "—";
  f.innerHTML =
    "<dt>step</dt><dd>" + st.idx + " (" + st.type + ")</dd>" +
    "<dt>action</dt><dd>" + (st.action || "—") + " · ok=" + (st.action_success === null ? "—" : st.action_success) + "</dd>" +
    "<dt>target visible</dt><dd>" + (st.authoritative_target_visible ? "● yes" : "no") + "</dd>" +
    "<dt>pose</dt><dd>" + pos + " · yaw " + (ag.rotation_yaw_deg ?? "—") + "°</dd>" +
    "<dt>target id(priv)</dt><dd>" + (h.target_object_ids || "—") + "</dd>" +
    "<dt>target pos(priv)</dt><dd>" + tp + "</dd>" +
    "<dt>terminal</dt><dd>" + (st.terminal ? "yes" : "no") + "</dd>";
}

// ---------- model input modal ----------
async function showModelInput() {
  if (!STATE) return;
  const el = $("#model-json-modal");
  el.textContent = "MODEL INPUT (permitted only)\n" + JSON.stringify(STATE.model_input, null, 2) +
    "\n\nPRIVILEGED (never model input)\n" + JSON.stringify(STATE.privileged, null, 2);
  el.classList.toggle("hidden");
}

// ---------- boot ----------
function bind() {
  $("#lang-select").addEventListener("change", (e) => setLang(e.target.value));
  $("#settings-lang").addEventListener("change", (e) => setLang(e.target.value));
  $$(".tab").forEach((t) => t.addEventListener("click", () => activateTab(t.dataset.tab)));
  $("#btn-start").addEventListener("click", startEpisode);
  $("#btn-teacher").addEventListener("click", doTeacher);
  $$(".act").forEach((b) => b.addEventListener("click", () => doAction(b.dataset.act)));
  $("#show-god").addEventListener("change", () => {
    $("#god-pane").style.display = $("#show-god").checked ? "" : "none";
    if ($("#show-god").checked) drawGod();
  });
  $("#btn-view-modelinput").addEventListener("click", (e) => { e.preventDefault(); showModelInput(); });
  $("#btn-ep-list").addEventListener("click", refreshEpisodeList);
  $("#btn-ep-load").addEventListener("click", loadEpisode);
  setInterval(() => {
    if ($("#ep-live").checked && $("#ep-select").value) loadEpisode();
  }, 2000);
  refreshEpisodeList();
  $("#model-json-modal").addEventListener("click", () => $("#model-json-modal").classList.add("hidden"));
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#model-json-modal").classList.add("hidden");
  });
}
async function boot() {
  await loadI18n();
  bind();
  activateTab("observation");
  // auto-load a house (deterministic default) so UAT can Start immediately
  try {
    const info = await apiPost("/api/house/load", { backend: "deterministic", seed: 0, house_id: "" });
    const h = info.house || {};
    $("#chip-env").textContent = info.backend;
    $("#v-env-house").textContent = h.house_id || "—";
    $("#v-env-split").textContent = h.split || "—";
    $("#v-env-rooms").textContent = h.room_count ?? "—";
    $("#v-env-backend").textContent = h.rendering_backend || "—";
    $("#v-env-cats").textContent = h.object_category_counts ? JSON.stringify(h.object_category_counts) : "—";
  } catch (err) {
    $("#chip-env").textContent = "load err";
  }
}
boot();
