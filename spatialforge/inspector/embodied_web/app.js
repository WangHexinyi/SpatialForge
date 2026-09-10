import * as God3D from "./god3d.js";
import * as Scene2D from "./scene2d.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let DICT = { en: {}, zh: {} };
let LANG = localStorage.getItem("sf.lang") || "bilingual";
let MODE = "live";
let STATE = null;
let EPISODES = [];
let EPISODE = null;
let EP_STATE = null;
let EP_SELECTED = null;
let EP_SELECTED_ID = null;
let EP_PICKER_OPEN = false;
let STUDENT = null;
let LIVE_DECISION = null;
let LIVE_TRACE = [];
let POLL_TIMER = null;
let TEACHER = null;
let HOUSE = null;
let DRAWER = null;

// ---- 3D God View / telemetry state ----
let GOD_MODE = localStorage.getItem("sf.godview") || "unity";
let GOD3D_READY = false;
let SCENE3D = null;
let SCENE3D_KEY = null;        // canonical key the current SCENE3D belongs to
let SCENE3D_GEN = 0;           // latest-request-wins guard for scene fetches
const SCENE3D_CACHE = new Map(); // key -> canonical payload (survives mode switches)
const SCENE3D_INFLIGHT = new Set(); // keys with a fetch in flight
let EPISODE_SEMANTICS = null;
let MODEL_INFO = null;
let TELEMETRY = null;
let EP_LIST_TIMER = null;
let HUD_TIMER = null;
let GOD3D_RAF = null;
// ---- real Unity God View (primary) ----
let UNITY_VIEW = localStorage.getItem("sf.unityview") || "follow";
let UNITY_STATUS = null;
let UNITY_URL = null;          // last requested frame URL
let UNITY_LOADING = false;
let UNITY_ERROR = null;        // last error string (same UNITY_URL only)
let UNITY_LOAD_GEN = 0;        // latest-request-wins token
const UNITY_CACHE = new Map(); // url -> {objectUrl, bytes, ms}
const UNITY_CACHE_MAX = 48;
// Explicit Unity ThirdPartyCamera state. The browser owns this and sends the
// resolved pose to the server; presets are Follow/Overview and the user can
// grab the camera into Free mode with the mouse.
const UNITY_CAM = {
  mode: UNITY_VIEW === "overview" ? "overview" : "follow", // follow | overview | free
  preset: UNITY_VIEW === "overview" ? "overview" : "follow", // last preset, for Reset
  target: [0, 0.9, 0],
  yaw: 0, pitch: 35, distance: 3.0, fov: 70,
  position: [0, 0.9, 0], rotation: [35, 0, 0],
};
let UNITY_DRAG = null;
let UNITY_RENDER_INFLIGHT = false;
let UNITY_RENDER_PENDING = null;
// Interactive camera: pointer state is local/60fps; Unity follows the latest
// pose through a single-flight + latest-pending scheduler. During a drag we
// request a real low-latency preview, then a viewport-sized settle frame.
let UNITY_INTERACTIVE = false;
let UNITY_SETTLE_TIMER = null;
let UNITY_RENDER_RAF = null;
// ---- curated / condensed replay ----
let CONDENSED = localStorage.getItem("sf.condensed") !== "0";
let DISPLAY_ORDER = null;

let PLAYING = false;
let SPEED_MS = 500;
let PLAY_TIMER = null;
let RAF_ID = null;
let GOD_ANIM = { active: false, from: null, to: null, t: 0, start: 0, duration: 500 };
let FRAME_CACHE = new Map();
const FRAME_CACHE_MAX = 240;
let PRELOAD_GEN = 0;
let MARKERS_KEY = null;
let SCRUB_RAF = null;
let SCRUB_TARGET = null;

// Display-only performance instrumentation (never model input). Inspect in the
// browser console with `sfPerfSummary()`.
const PERF = { replayCached: [], replayUncached: [], unity: [], fpv: [] };
window.__sfPerf = PERF;
window.sfPerfSummary = function sfPerfSummary() {
  const stat = (arr) => {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    return {
      n: s.length,
      avg_ms: +(s.reduce((a, b) => a + b, 0) / s.length).toFixed(1),
      p50_ms: +s[Math.floor(s.length * 0.5)].toFixed(1),
      p95_ms: +s[Math.min(s.length - 1, Math.floor(s.length * 0.95))].toFixed(1),
      max_ms: +s[s.length - 1].toFixed(1),
    };
  };
  return {
    replay_step_cached: stat(PERF.replayCached),
    replay_step_uncached: stat(PERF.replayUncached),
    unity_frame: stat(PERF.unity),
    fpv_frame: stat(PERF.fpv),
  };
};

const SPEED_TABLE = { "0.5": 1000, "1": 500, "2": 250, "4": 125 };

async function loadI18n() {
  const r = await fetch("/api/i18n");
  const data = await r.json();
  DICT = data.dictionary;
  renderI18n();
}

function pick(lang) {
  if (lang === "en") return DICT.en;
  if (lang === "zh-CN" || lang === "zh") return DICT.zh;
  return DICT.en;
}

function textFor(key, mode) {
  const en = DICT.en[key];
  const zh = DICT.zh[key];
  if (!en && !zh) return "[" + key + "]";
  if (mode === "long") {
    const base = pick(LANG);
    return (base[key] && (base[key].long || base[key].short)) || key;
  }
  if (LANG === "bilingual") {
    const es = (en && en.short) || "";
    const zs = (zh && zh.short) || "";
    if (zs && es && zs !== es) return zs + " · " + es;
    return es || zs;
  }
  const base = pick(LANG);
  const e = base[key];
  return (e && (e.short || e.long)) || (en && en.short) || key;
}

const T = (key, mode) => textFor(key, mode);

function renderI18n() {
  $$("[data-i18n]").forEach((el) => {
    const mode = el.getAttribute("data-i18n-mode") || "short";
    el.textContent = textFor(el.dataset.i18n, mode);
  });
  const top = $("#lang-select");
  if (top) top.value = LANG;
  const settings = $("#settings-lang");
  if (settings) settings.value = LANG;
  renderPlayer();
}

function setLang(l) {
  LANG = l;
  localStorage.setItem("sf.lang", l);
  renderI18n();
  renderAll();
}

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

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtNum(x) {
  const n = Number(x);
  if (!isFinite(n)) return "—";
  return String(parseFloat(n.toFixed(2)));
}

function fmtPos(p) {
  if (!Array.isArray(p)) return "—";
  return p.map(fmtNum).join(", ");
}

function setText(id, v) {
  const el = $(id);
  if (el) el.textContent = v;
}

function frameName(idx) {
  return "frame_" + String(idx).padStart(4, "0") + ".png";
}

function samePos(a, b) {
  if (!a || !b) return false;
  return Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9 && Math.abs(a[2] - b[2]) < 1e-9;
}

function setMode(mode) {
  MODE = mode;
  if (mode !== "replay") pause();
  const app = $("#app");
  app.classList.toggle("mode-live", mode === "live");
  app.classList.toggle("mode-replay", mode === "replay");
  $("#btn-mode-live").classList.toggle("active", mode === "live");
  $("#btn-mode-replay").classList.toggle("active", mode === "replay");
  renderAll();
}

function renderAll() {
  renderTopbar();
  renderEpisodeHeader();
  renderDecision();
  renderFpvMeta();
  renderGodLabels();
  renderGodWarning();
  renderGod();
  renderPlayer();
  renderMetrics();
  renderPrivileged();
  renderManualStatus();
  if (DRAWER === "debug") renderDebug();
}

function renderTopbar() {
  const env = $("#chip-env");
  const task = $("#chip-task");
  const agent = $("#chip-agent");
  if (MODE === "replay" && EPISODE) {
    const h = EPISODE.header || {};
    env.textContent = h.house_id || (HOUSE && HOUSE.house_id) || "—";
    task.textContent = h.category || "—";
    const total = Math.max(0, (EPISODE.timeline || []).length - 1);
    agent.textContent = "#" + (EP_SELECTED ?? "—") + " / " + total + " · " + (h.model_controlled ? "model" : "teacher");
    return;
  }
  if (EPISODE && EPISODE.in_progress) {
    const h = EPISODE.header || {};
    env.textContent = (HOUSE && HOUSE.house_id) || "—";
    task.textContent = h.category || "—";
    agent.textContent = "live · " + T("god.step") + " " + (h.step ?? "—");
    return;
  }
  if (STATE && STATE.active) {
    env.textContent = STATE.backend || "—";
    task.textContent = (STATE.task && STATE.task.target_category) || "—";
    const st = STATE.terminal
      ? (STATE.task && STATE.task.success ? T("common.success") : T("common.failure"))
      : T("common.running");
    agent.textContent = st + " · " + ((STATE.agent_state && STATE.agent_state.last_action) || "idle");
    return;
  }
  env.textContent = (HOUSE && HOUSE.house_id) || "—";
  task.textContent = "—";
  agent.textContent = "idle";
}

function renderEpisodeHeader() {
  const el = $("#ep-header");
  if (!el) return;
  if (EPISODE && EPISODE.header && !EPISODE.in_progress) {
    const h = EPISODE.header;
    const parts = [];
    if (h.episode_id) parts.push(h.episode_id);
    if (h.house_id) parts.push(h.house_id);
    if (h.category) parts.push(h.category);
    if (h.status) parts.push(h.status + (h.success === true ? " ✓" : h.success === false ? " ✕" : ""));
    if (h.model_controlled) parts.push("model");
    if (h.decision_count !== undefined) parts.push("dec " + h.decision_count + " / invalid " + (h.invalid_decision_count ?? 0));
    if (h.model_terminal_reason) parts.push(h.model_terminal_reason);
    if (h.labels && h.labels.length) parts.push(h.labels.join("/"));
    if (h.render_quality) parts.push("quality " + h.render_quality);
    el.textContent = parts.join(" · ");
    return;
  }
  if (EPISODE && EPISODE.in_progress) {
    const h = EPISODE.header || {};
    el.textContent = [h.episode_id, h.category, h.status, h.goal].filter(Boolean).join(" · ");
    return;
  }
  el.textContent = T("tl.empty");
}

function poseAt(idx) {
  if (!EPISODE) return null;
  const h = EPISODE.header || {};
  const e = (EPISODE.timeline || [])[idx];
  if (e && e.agent && e.agent.position) return e.agent;
  if (idx <= 0 && h.spawn) {
    return { position: h.spawn.position, rotation_yaw_deg: h.spawn.yaw, camera_horizon_deg: h.spawn.horizon };
  }
  return null;
}

function localStepState(idx) {
  const e = (EPISODE.timeline || [])[idx] || {};
  const h = EPISODE.header || {};
  const state = Object.assign({}, e, { idx, header: h, _privileged_researcher_only: true });
  if (!state.agent && idx === 0 && h.spawn) {
    state.agent = { position: h.spawn.position, rotation_yaw_deg: h.spawn.yaw, camera_horizon_deg: h.spawn.horizon };
  }
  return state;
}

function decisionAt(idx) {
  const decs = (EPISODE && EPISODE.decisions) || [];
  const frame = frameName(idx);
  return decs.find((d) => d.frame === frame) || decs.find((d) => d.decision_idx === idx) || null;
}

function decisionPayload() {
  if (MODE === "replay" && EPISODE) {
    const st = EP_STATE || {};
    const dec = (EPISODE.decisions || []).find(
      (d) => d.decision_idx === EP_SELECTED || (st.frame && d.frame === st.frame)
    ) || null;
    return { kind: "replay", st, dec, header: EPISODE.header || {} };
  }
  if (MODE === "live" && EPISODE && EPISODE.in_progress) {
    return { kind: "live", st: LIVE_DECISION || {}, dec: LIVE_DECISION || null, header: EPISODE.header || {} };
  }
  if (STATE && STATE.active) {
    return { kind: "manual", st: STATE, dec: null, header: {} };
  }
  return { kind: "none" };
}

function teacherRef(idx, h) {
  if (h.model_controlled) return T("dec.teacher_na");
  if (!STUDENT || !STUDENT.steps || !STUDENT.steps[idx]) return "—";
  return STUDENT.steps[idx].teacher_action || "—";
}

function renderDecision() {
  const p = decisionPayload();
  const src = $("#decision-src");
  const blank = ["#d-raw", "#d-parsed", "#d-executed", "#d-teacher", "#d-visible", "#d-verifier", "#d-terminal", "#d-latency"];
  if (p.kind === "none") {
    blank.forEach((id) => setText(id, "—"));
    if (src) src.textContent = "—";
    return;
  }
  if (p.kind === "replay") {
    const st = p.st || {};
    const d = p.dec || {};
    const h = p.header || {};
    if (src) src.textContent = h.model_controlled ? "model" : "teacher";
    setText("#d-raw", d.model_raw || "—");
    setText("#d-parsed", d.model_action
      ? d.model_action + (d.invalid ? " · invalid(" + (d.parse_reason || "") + ")" : "")
      : (h.model_controlled ? "—" : T("dec.teacher_na")));
    const ok = st.action_success;
    setText("#d-executed", (st.action || T("tl.start")) + (ok === true ? " ✓" : ok === false ? " ✕" : ""));
    setText("#d-teacher", teacherRef(EP_SELECTED, h));
    setText("#d-visible", st.authoritative_target_visible ? "● " + T("common.yes") : "○ " + T("common.no"));
    setText("#d-verifier", (h.status || "—")
      + (h.success === true ? " ✓" : h.success === false ? " ✕" : "")
      + (h.success_reason ? " · " + h.success_reason : ""));
    setText("#d-terminal", st.terminal ? (h.model_terminal_reason || h.success_reason || "terminal") : "—");
    setText("#d-latency", typeof d.latency_ms === "number" ? d.latency_ms + " ms" : "—");
    return;
  }
  if (p.kind === "live") {
    const d = p.dec || {};
    const h = p.header || {};
    if (src) src.textContent = "live · #" + (d.decision_idx ?? "—");
    setText("#d-raw", d.model_raw || "—");
    setText("#d-parsed", d.model_action
      ? d.model_action + (d.invalid ? " · invalid(" + (d.parse_reason || "") + ")" : "")
      : "—");
    setText("#d-executed", d.executed === true ? (d.model_action || "—") : (d.invalid ? "not executed" : "—"));
    setText("#d-teacher", T("dec.teacher_na"));
    setText("#d-visible", d.authoritative_target_visible ? "● " + T("common.yes") : "○ " + T("common.no"));
    setText("#d-verifier", h.status || "—");
    setText("#d-terminal", "—");
    setText("#d-latency", typeof d.latency_ms === "number" ? d.latency_ms + " ms" : "—");
    return;
  }
  const s = p.st || {};
  if (src) src.textContent = "manual";
  setText("#d-raw", "—");
  setText("#d-parsed", T("dec.manual"));
  const ag = s.agent_state || {};
  setText("#d-executed", ag.last_action
    ? ag.last_action + (ag.last_action_success === true ? " ✓" : ag.last_action_success === false ? " ✕" : "")
    : "—");
  setText("#d-teacher", TEACHER && (TEACHER.actions || []).length ? TEACHER.actions.join(" ") : "—");
  setText("#d-visible", s.observation_meta && s.observation_meta.target_visible ? "● " + T("common.yes") : "○ " + T("common.no"));
  setText("#d-verifier", ((s.task && s.task.status) || "—")
    + (s.task && s.task.success === true ? " ✓" : s.task && s.task.success === false ? " ✕" : "")
    + (s.task && s.task.success_reason ? " · " + s.task.success_reason : ""));
  setText("#d-terminal", s.terminal ? ((s.task && s.task.success_reason) || "—") : "—");
  setText("#d-latency", "—");
}

function setFlag(el, visible) {
  if (!el) return;
  if (visible === true) {
    el.textContent = "● " + T("fpv.visible");
    el.className = "target-flag yes";
  } else if (visible === false) {
    el.textContent = "○ " + T("fpv.not_visible");
    el.className = "target-flag no";
  } else {
    el.textContent = "";
    el.className = "target-flag";
  }
}

function renderFpvMeta() {
  const meta = $("#obs-meta");
  const flag = $("#fpv-visible");
  if (!meta) return;
  if (MODE === "replay" && EP_STATE) {
    const h = EPISODE.header || {};
    let ag = EP_STATE.agent;
    if (!ag && EP_SELECTED === 0 && h.spawn) {
      ag = { rotation_yaw_deg: h.spawn.yaw, camera_horizon_deg: h.spawn.horizon };
    }
    ag = ag || {};
    meta.textContent = "#" + EP_SELECTED + " · " + (EP_STATE.action || T("tl.start"))
      + " · yaw " + fmtNum(ag.rotation_yaw_deg) + "° · horizon " + fmtNum(ag.camera_horizon_deg) + "°";
    setFlag(flag, EP_STATE.authoritative_target_visible);
    return;
  }
  if (MODE === "live" && STATE && STATE.active) {
    const om = STATE.observation_meta || {};
    meta.textContent = T("god.step") + " " + (om.step ?? "—") + " · frame " + (om.frame_id || "—")
      + " · horizon " + fmtNum(om.camera_horizon_deg) + "° · ts " + (om.timestamp_ms || "—");
    setFlag(flag, om.target_visible);
    return;
  }
  if (EPISODE && EPISODE.in_progress) {
    const d = LIVE_DECISION || {};
    meta.textContent = T("live.running") + " · " + T("god.step") + " " + (EPISODE.header && EPISODE.header.step !== undefined ? EPISODE.header.step : "—")
      + " · decision #" + (d.decision_idx ?? "—");
    setFlag(flag, d.authoritative_target_visible);
    return;
  }
  meta.textContent = "—";
  setFlag(flag, null);
}

// Trajectory invariant (mirrors spatialforge/inspector/scene3d.py):
// only connect consecutive, real executed, valid transitions. Setup/spawn/
// reset/teleport and unexplained displacement break the line.
const _TRANSLATION_ACTIONS = new Set(["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]);
const _SETUP_ORIGINS = new Set(["setup", "spawn", "reset", "teleport", "teacher"]);

function distXZ(a, b) {
  return Math.hypot(a[0] - b[0], a[2] - b[2]);
}

function buildTraceSegments(idx) {
  if (!EPISODE) return [];
  const h = EPISODE.header || {};
  const entries = [];
  if (h.spawn && h.spawn.position) {
    entries.push({ idx: -1, action: null, origin: "spawn", pos: h.spawn.position });
  }
  (EPISODE.timeline || []).forEach((e) => {
    if (e.idx <= idx && e.agent && e.agent.position) {
      entries.push({ idx: e.idx, action: e.action, origin: e.action_origin, pos: e.agent.position });
    }
  });
  const segs = [];
  let cur = [];
  let prev = null;
  for (const en of entries) {
    const pos = en.pos;
    if (prev === null) {
      cur = [pos];
    } else {
      const moved = distXZ(prev, pos);
      let brk = null;
      if (moved > 0.05) {
        if (_SETUP_ORIGINS.has(en.origin)) brk = "setup";
        else if (!en.action) brk = "missing_action";
        else {
          const limit = _TRANSLATION_ACTIONS.has(en.action) ? 0.75 : 0.05;
          if (moved > limit) brk = "unexplained_displacement";
        }
      }
      if (brk) {
        if (cur.length) segs.push(cur);
        cur = [pos];
      } else if (cur.length && distXZ(cur[cur.length - 1], pos) > 1e-4) {
        cur.push(pos);
      }
    }
    prev = pos;
  }
  if (cur.length) segs.push(cur);
  return segs;
}

function flattenSegments(segs) {
  const out = [];
  (segs || []).forEach((s) => s.forEach((p) => out.push(p)));
  return out;
}

// The single canonical scene payload consumed by BOTH the Three.js 3D view and
// the 2D top-down diagnostic view. Geometry comes from the backend scene3d
// payload; episode truth comes from `canonicalEpisode()` (the same function the
// 3D view uses). There is no separate 2D-only map logic.
// Identifies which canonical scene payload the current session needs. Used both
// to cache scene geometry across mode switches and to reject stale responses.
function currentSceneKey() {
  if (MODE === "replay" && EPISODE && !EPISODE.in_progress) {
    return "replay:" + ((EPISODE.header && EPISODE.header.episode_id) || "?");
  }
  if (EPISODE && EPISODE.in_progress) {
    return "live:" + ((EPISODE.header && EPISODE.header.episode_id) || "?");
  }
  if (STATE && STATE.active) {
    // task_id is unique per started episode, so a new manual episode on a
    // different house can never reuse the previous house's geometry.
    const taskId = (STATE.task && STATE.task.task_id) || "?";
    return "manual:" + (STATE.backend || "?") + ":" + taskId;
  }
  return null;
}

function canonicalScene() {
  const episode = canonicalEpisode();
  const key = currentSceneKey();
  // Only expose geometry that belongs to the *current* scene key; otherwise a
  // stale payload from a previous episode/mode could be projected as if it were
  // the current world.
  const usable = SCENE3D && key && SCENE3D_KEY === key;
  const scene = (usable && SCENE3D.scene) || null;
  const meta = (usable && SCENE3D.scene_meta) || {};
  return {
    schema: "spatialforge_scene2d.v1",
    scene,
    episode,
    bounds: (scene && scene.bounds) || null,
    source: meta.source || (scene && scene.source) || null,
    unavailable: meta.unavailable || (scene && scene.unavailable) || [],
    scene_key: key,
    scene_expected: key !== null,
  };
}

function lerpNum(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  const d = ((b - a + 540) % 360) - 180;
  return a + d * t;
}

function lerpPose(a, b, t) {
  return {
    position: [
      lerpNum(a.position[0], b.position[0], t),
      lerpNum(a.position[1], b.position[1], t),
      lerpNum(a.position[2], b.position[2], t),
    ],
    rotation_yaw_deg: lerpAngle(Number(a.rotation_yaw_deg) || 0, Number(b.rotation_yaw_deg) || 0, t),
    camera_horizon_deg: lerpNum(Number(a.camera_horizon_deg) || 0, Number(b.camera_horizon_deg) || 0, t),
  };
}

function renderGod() {
  if (GOD_MODE === "unity") {
    renderGodUnity();
    return;
  }
  if (GOD_MODE === "3d" && GOD3D_READY) {
    renderGod3D();
    return;
  }
  renderGod2D();
}

// ---- real Unity God View (primary) ---------------------------------
// The image visibility and the overlay are controlled *independently*. The old
// code hid the <img> right after assigning src (via showUnityStatus) and only
// hid the overlay in img.onload, so a successfully rendered PNG stayed
// display:none over a black panel — the exact P0 black-screen failure.
function setUnityOverlay(kind, text) {
  const st = $("#god-unity-status");
  if (!st) return;
  if (kind === "hidden") {
    st.classList.add("hidden");
    st.classList.remove("error", "loading");
    return;
  }
  st.textContent = text;
  st.classList.remove("hidden");
  st.classList.toggle("error", kind === "error");
  st.classList.toggle("loading", kind === "loading");
}

// Small, non-blocking "rendering…" pill. Used while a frame is being refreshed
// and a previous frame is already on screen, so the viewport is never covered.
function setUnityIndicator(kind, text) {
  const el = $("#god-unity-indicator");
  if (!el) return;
  if (!kind || kind === "hidden") {
    el.classList.add("hidden");
    el.classList.remove("error");
    return;
  }
  el.textContent = text || T("god.unity_rendering");
  el.classList.remove("hidden");
  el.classList.toggle("error", kind === "error");
}

// Settle-frame resolution follows the real viewport (clamped) so the browser
// never upscales a tiny image; devicePixelRatio is honoured up to 2x.
function unityViewportSize() {
  const wrap = $("#god-wrap");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = (wrap && wrap.clientWidth) || 640;
  const ch = (wrap && wrap.clientHeight) || 480;
  return {
    w: Math.round(Math.max(320, Math.min(1600, cw * dpr))),
    h: Math.round(Math.max(240, Math.min(1200, ch * dpr))),
  };
}

// Coalesce render requests to one per animation frame. pointermove never awaits
// network/Unity; it only mutates local state and schedules this.
function unityScheduleRender() {
  if (UNITY_RENDER_RAF) return;
  UNITY_RENDER_RAF = requestAnimationFrame(() => {
    UNITY_RENDER_RAF = null;
    if (GOD_MODE === "unity") renderGodUnity();
  });
}

function unityEndInteraction() {
  UNITY_INTERACTIVE = false;
  if (UNITY_SETTLE_TIMER) {
    clearTimeout(UNITY_SETTLE_TIMER);
    UNITY_SETTLE_TIMER = null;
  }
  unityScheduleRender();
}

function resetUnity() {
  UNITY_LOAD_GEN += 1;
  UNITY_URL = null;
  UNITY_LOADING = false;
  UNITY_ERROR = null;
  UNITY_RENDER_PENDING = null;
  UNITY_INTERACTIVE = false;
  if (UNITY_SETTLE_TIMER) {
    clearTimeout(UNITY_SETTLE_TIMER);
    UNITY_SETTLE_TIMER = null;
  }
  setUnityIndicator("hidden");
}

// ---- Unity ThirdPartyCamera state (browser-owned) -----------------
function unityComputePose() {
  const t = UNITY_CAM.target;
  const yaw = (UNITY_CAM.yaw * Math.PI) / 180;
  const pitch = (UNITY_CAM.pitch * Math.PI) / 180;
  const d = UNITY_CAM.distance;
  // AI2-THOR third-party camera forward is (sin y, cos y) in x/z and looks
  // down by +rotation.x. Place the camera opposite its forward vector.
  UNITY_CAM.position = [
    t[0] - Math.sin(yaw) * Math.cos(pitch) * d,
    t[1] + Math.sin(pitch) * d,
    t[2] - Math.cos(yaw) * Math.cos(pitch) * d,
  ];
  UNITY_CAM.rotation = [UNITY_CAM.pitch, UNITY_CAM.yaw, 0];
}

function unityApplyPreset(name) {
  const canon = canonicalScene();
  const scene = canon.scene;
  const ag = canon.episode && canon.episode.agent;
  if (name === "overview") {
    const b = scene && scene.bounds;
    if (b) {
      const cx = (b.min[0] + b.max[0]) / 2;
      const cz = (b.min[2] + b.max[2]) / 2;
      const span = Math.max(b.max[0] - b.min[0], b.max[2] - b.min[2], 4);
      UNITY_CAM.target = [cx, 1.0, cz];
      UNITY_CAM.distance = span * 0.8;
      UNITY_CAM.pitch = 45;
      UNITY_CAM.yaw = 0;
    } else {
      UNITY_CAM.target = [0, 1.0, 0];
      UNITY_CAM.distance = 6.0;
      UNITY_CAM.pitch = 45;
      UNITY_CAM.yaw = 0;
    }
    UNITY_CAM.mode = "overview";
    UNITY_CAM.preset = "overview";
  } else {
    const p = (ag && ag.position) || [0, 0.9, 0];
    UNITY_CAM.target = [p[0], p[1] + 0.2, p[2]];
    UNITY_CAM.distance = 3.0;
    UNITY_CAM.pitch = 35;
    UNITY_CAM.yaw = Number(ag && ag.rotation_yaw_deg) || 0;
    UNITY_CAM.mode = "follow";
    UNITY_CAM.preset = "follow";
  }
  UNITY_VIEW = UNITY_CAM.preset;
  unityComputePose();
  syncUnityToolButtons();
}

// Follow the agent on step changes, but never reset a Free/Overview camera.
function unityFollowStep() {
  if (UNITY_CAM.mode !== "follow") return;
  const canon = canonicalScene();
  const ag = canon.episode && canon.episode.agent;
  if (ag && ag.position) {
    UNITY_CAM.target = [ag.position[0], ag.position[1] + 0.2, ag.position[2]];
    UNITY_CAM.yaw = Number(ag.rotation_yaw_deg) || 0;
    unityComputePose();
  }
}

function syncUnityToolButtons() {
  $$("#god-unity-tools button").forEach((x) => {
    x.classList.toggle("active", x.dataset.uv === UNITY_CAM.preset && UNITY_CAM.mode !== "free");
  });
  const camHud = $("#god-camera-hud");
  if (camHud) {
    camHud.textContent = (UNITY_CAM.mode === "free" ? T("god.unity_free") : UNITY_CAM.mode)
      + " · yaw " + fmtNum(UNITY_CAM.yaw) + "° · pitch " + fmtNum(UNITY_CAM.pitch)
      + "° · d " + fmtNum(UNITY_CAM.distance);
  }
}

function unityFrameUrl() {
  const live = MODE === "live" && STATE && STATE.active;
  const inProgress = !!(EPISODE && EPISODE.in_progress);
  const base = (live || inProgress)
    ? "/api/unity/live_godview"
    : "/api/episodes/unity_godview";
  const q = [];
  if (base === "/api/episodes/unity_godview") {
    const epId = (EPISODE && EPISODE.header && EPISODE.header.episode_id) || "";
    q.push("episode_id=" + encodeURIComponent(epId));
    q.push("step=" + (EP_SELECTED ?? 0));
  } else {
    // live frames change every action even at the same camera pose; the step is
    // part of the cache key.
    q.push("step=" + ((STATE && STATE.task && STATE.task.step) || (EPISODE && EPISODE.header && EPISODE.header.step) || 0));
  }
  q.push("view=" + encodeURIComponent(UNITY_CAM.mode === "overview" ? "overview" : "follow"));
  q.push("cam_mode=" + encodeURIComponent(UNITY_CAM.mode));
  const P = UNITY_CAM.position;
  const R = UNITY_CAM.rotation;
  q.push("px=" + P[0].toFixed(4), "py=" + P[1].toFixed(4), "pz=" + P[2].toFixed(4));
  q.push("rx=" + R[0].toFixed(3), "ry=" + R[1].toFixed(3), "rz=" + R[2].toFixed(3));
  q.push("fov=" + UNITY_CAM.fov);
  q.push("tx=" + UNITY_CAM.target[0].toFixed(3),
    "ty=" + UNITY_CAM.target[1].toFixed(3),
    "tz=" + UNITY_CAM.target[2].toFixed(3));
  q.push("dist=" + UNITY_CAM.distance.toFixed(3));
  // Interactive drags get a real low-latency preview; the settle frame is
  // rendered at the viewport size so the steady state is sharp.
  q.push("q=" + (UNITY_INTERACTIVE ? "preview" : "full"));
  if (!UNITY_INTERACTIVE) {
    const vp = unityViewportSize();
    q.push("w=" + vp.w, "h=" + vp.h);
  }
  return base + "?" + q.join("&");
}

function cacheUnityFrame(url, objectUrl, meta) {
  if (UNITY_CACHE.has(url)) {
    try { URL.revokeObjectURL(UNITY_CACHE.get(url).objectUrl); } catch (e) { /* ignore */ }
  }
  UNITY_CACHE.set(url, Object.assign({ objectUrl }, meta));
  while (UNITY_CACHE.size > UNITY_CACHE_MAX) {
    const oldest = UNITY_CACHE.keys().next().value;
    try { URL.revokeObjectURL(UNITY_CACHE.get(oldest).objectUrl); } catch (e) { /* ignore */ }
    UNITY_CACHE.delete(oldest);
  }
}

function showCachedUnity(img, url) {
  const hit = UNITY_CACHE.get(url);
  if (!hit) return false;
  if (img.dataset.src !== url) {
    img.dataset.src = url;
    img.src = hit.objectUrl;
  }
  img.classList.remove("hidden");
  return true;
}

// Single-flight with latest-pending coalescing: while a real Unity render is in
// flight, only the newest requested pose is kept. Intermediate drag poses are
// dropped so the server (one serialized engine) never builds a backlog.
function loadUnityFrame(url) {
  if (UNITY_RENDER_INFLIGHT) {
    UNITY_RENDER_PENDING = url;
    return;
  }
  UNITY_RENDER_INFLIGHT = true;
  _fetchUnityFrame(url).finally(() => {
    UNITY_RENDER_INFLIGHT = false;
    const next = UNITY_RENDER_PENDING;
    UNITY_RENDER_PENDING = null;
    if (next && next !== url && GOD_MODE === "unity") loadUnityFrame(next);
  });
}

// Fetch the real Unity frame as a blob so HTTP status and the server's JSON
// error body are surfaced verbatim. Latest-request-wins: a slow earlier request
// can never overwrite a newer step's frame.
async function _fetchUnityFrame(url) {
  const gen = ++UNITY_LOAD_GEN;
  UNITY_LOADING = true;
  UNITY_ERROR = null;
  const img = $("#god-unity-img");
  // If a real frame is already on screen, never cover it with the full-viewport
  // loading panel; show a tiny non-blocking pill only if the render is slow.
  const showing = !!(img && img.dataset.src);
  let indicatorTimer = null;
  if (showing) {
    indicatorTimer = setTimeout(() => setUnityIndicator("loading", T("god.unity_rendering")), 150);
  } else {
    setUnityOverlay("loading", T("god.unity_rendering"));
  }
  const t0 = performance.now();
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      let detail = "HTTP " + res.status;
      try {
        const j = await res.json();
        if (j && j.error) detail += " — " + j.error;
      } catch (e) { /* non-JSON body */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    if (gen !== UNITY_LOAD_GEN) return; // superseded by a newer request
    const objectUrl = URL.createObjectURL(blob);
    const ms = performance.now() - t0;
    PERF.unity.push(ms);
    cacheUnityFrame(url, objectUrl, { bytes: blob.size, ms });
    if (img) {
      img.dataset.src = url;
      img.onload = () => {
        if (indicatorTimer) clearTimeout(indicatorTimer);
        if (gen !== UNITY_LOAD_GEN) return;
        img.classList.remove("hidden");
        setUnityOverlay("hidden");
        setUnityIndicator("hidden");
      };
      img.onerror = () => {
        if (indicatorTimer) clearTimeout(indicatorTimer);
        if (gen !== UNITY_LOAD_GEN) return;
        img.classList.add("hidden");
        img.removeAttribute("src");
        img.dataset.src = "";
        setUnityOverlay("error", T("god.unity_unavailable") + ": PNG decode failed");
      };
      img.src = objectUrl;
    }
  } catch (err) {
    if (indicatorTimer) clearTimeout(indicatorTimer);
    if (gen !== UNITY_LOAD_GEN) return;
    UNITY_ERROR = String((err && err.message) || err);
    if (img && !showing) {
      img.classList.add("hidden");
      img.removeAttribute("src");
      img.dataset.src = "";
      setUnityOverlay("error", T("god.unity_unavailable") + ": " + UNITY_ERROR);
    } else {
      // keep the last good frame; surface the failure without blanking the view
      setUnityIndicator("error", T("god.unity_unavailable") + ": " + UNITY_ERROR);
    }
  } finally {
    if (indicatorTimer) clearTimeout(indicatorTimer);
    if (gen === UNITY_LOAD_GEN) UNITY_LOADING = false;
  }
}

function renderGodLabels() {
  const box = $("#god-labels");
  if (!box) return;
  const labels = (EPISODE && EPISODE.header && EPISODE.header.labels) || [];
  box.innerHTML = labels.map((l) => {
    const cls = ["VALID", "SUCCESS"].includes(l) ? "ok"
      : ["LEGACY", "INVALID_SENSOR", "INVALID_SPAWN", "ANOMALY", "CRASH", "FALSE_DONE"].includes(l) ? "bad"
      : ["TIMEOUT", "STUCK", "POLICY_COLLAPSE"].includes(l) ? "warn" : "info";
    return '<span class="label-chip ' + cls + '">' + esc(l) + "</span>";
  }).join("");
}

function renderGodWarning() {
  const el = $("#god-warning");
  if (!el) return;
  const h = (EPISODE && EPISODE.header) || {};
  const labels = h.labels || [];
  const msgs = [];
  if (labels.includes("LEGACY")) msgs.push(T("warn.legacy"));
  if (labels.includes("INVALID_SENSOR")) msgs.push(T("warn.sensor") + (h.render_quality ? " (" + h.render_quality + ")" : ""));
  if (labels.includes("INVALID_SPAWN") || labels.includes("ANOMALY")) msgs.push(T("warn.trajectory"));
  if (labels.includes("POLICY_COLLAPSE")) msgs.push(T("warn.collapse"));
  if (labels.includes("STUCK")) msgs.push(T("warn.stuck"));
  if (!msgs.length) {
    el.classList.add("hidden");
    return;
  }
  el.textContent = msgs.join("  ·  ");
  el.classList.remove("hidden");
}

function renderGodUnity() {
  const img = $("#god-unity-img");
  const cv3d = $("#god3d-canvas");
  const cv2d = $("#god-canvas");
  if (cv3d) cv3d.style.display = "none";
  if (cv2d) cv2d.classList.add("hidden");
  renderGodWarning();
  renderGodLabels();
  syncUnityToolButtons();

  const canReplay = MODE === "replay" && EPISODE && !EPISODE.in_progress;
  const canLive = MODE === "live" && STATE && STATE.active;
  const canInProgress = !!(EPISODE && EPISODE.in_progress);
  if (!canReplay && !canLive && !canInProgress) {
    if (img) img.classList.add("hidden");
    setUnityOverlay("error", T("god.unity_replay_only"));
    renderGodMetaUnity(null);
    return;
  }
  if (UNITY_STATUS && UNITY_STATUS.available === false) {
    if (img) img.classList.add("hidden");
    setUnityOverlay("error", T("god.unity_unavailable") + (UNITY_STATUS.reason ? " — " + UNITY_STATUS.reason : ""));
    renderGodMetaUnity(null);
    return;
  }
  const url = unityFrameUrl();
  if (UNITY_CACHE.has(url)) {
    if (img && showCachedUnity(img, url)) setUnityOverlay("hidden");
  } else if (UNITY_URL === url && UNITY_LOADING) {
    // request already in flight for this exact frame; keep the loading overlay
  } else if (UNITY_URL === url && UNITY_ERROR) {
    // keep the loud error visible; do not auto-retry on every animation frame.
    // Re-clicking the mode button (or switching view/step) clears it.
  } else {
    UNITY_URL = url;
    loadUnityFrame(url);
  }
  renderGodMetaUnity(EP_STATE);
}

function renderGodMetaUnity(epState) {
  const meta = [];
  const modeLabel = UNITY_CAM.mode === "free" ? T("god.unity_free")
    : (UNITY_CAM.mode === "overview" ? T("god.unity_overview") : T("god.unity_follow"));
  meta.push(T("god.unity_main") + " · " + modeLabel);
  meta.push(MODE === "replay" ? "#" + (EP_SELECTED ?? "—") : T("live.running"));
  if (epState && epState.agent && epState.agent.position) {
    meta.push("pose " + fmtPos(epState.agent.position) + " · yaw " + fmtNum(epState.agent.rotation_yaw_deg) + "°");
  }
  meta.push("cam yaw " + fmtNum(UNITY_CAM.yaw) + "° · pitch " + fmtNum(UNITY_CAM.pitch) + "° · d " + fmtNum(UNITY_CAM.distance));
  setText("#god-meta", meta.join(" · "));
  setText("#god-legend", "● " + T("god.unity_main") + " · " + T("god.view3d") + " " + T("god.diagnostic"));
}

// ---- Unity mouse camera control -----------------------------------
function unityOrbit(dx, dy) {
  UNITY_CAM.yaw = ((UNITY_CAM.yaw - dx * 0.3) % 360 + 360) % 360;
  UNITY_CAM.pitch = Math.min(85, Math.max(-85, UNITY_CAM.pitch + dy * 0.3));
  unityComputePose();
}

function unityPan(dx, dy) {
  const yaw = (UNITY_CAM.yaw * Math.PI) / 180;
  const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const scale = UNITY_CAM.distance * 0.0016;
  const t = UNITY_CAM.target;
  UNITY_CAM.target = [
    t[0] - right[0] * dx * scale,
    Math.min(12, Math.max(0.0, t[1] + dy * scale)),
    t[2] - right[2] * dx * scale,
  ];
  unityComputePose();
}

function bindUnityCamera() {
  const wrap = $("#god-wrap");
  if (!wrap) return;
  const active = () => GOD_MODE === "unity";
  wrap.addEventListener("contextmenu", (e) => { if (active()) e.preventDefault(); });
  wrap.addEventListener("pointerdown", (e) => {
    if (!active()) return;
    if (e.target && e.target.closest && e.target.closest(".tool-group")) return;
    if (e.button !== 0 && e.button !== 2) return;
    UNITY_DRAG = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    UNITY_INTERACTIVE = true;
    if (UNITY_SETTLE_TIMER) {
      clearTimeout(UNITY_SETTLE_TIMER);
      UNITY_SETTLE_TIMER = null;
    }
    try { wrap.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    e.preventDefault();
  });
  wrap.addEventListener("pointermove", (e) => {
    if (!active() || !UNITY_DRAG) return;
    const dx = e.clientX - UNITY_DRAG.x;
    const dy = e.clientY - UNITY_DRAG.y;
    UNITY_DRAG.x = e.clientX;
    UNITY_DRAG.y = e.clientY;
    if (UNITY_DRAG.pan) unityPan(dx, dy);
    else unityOrbit(dx, dy);
    UNITY_CAM.mode = "free";
    // local state + HUD update immediately; the network render is coalesced
    syncUnityToolButtons();
    unityScheduleRender();
  });
  const endDrag = (e) => {
    if (!UNITY_DRAG) return;
    UNITY_DRAG = null;
    try { wrap.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    unityEndInteraction();
  };
  wrap.addEventListener("pointerup", endDrag);
  wrap.addEventListener("pointercancel", endDrag);
  wrap.addEventListener("wheel", (e) => {
    if (!active()) return;
    e.preventDefault();
    UNITY_CAM.distance = Math.min(80, Math.max(0.4, UNITY_CAM.distance * Math.exp(e.deltaY * 0.001)));
    UNITY_CAM.mode = "free";
    UNITY_INTERACTIVE = true;
    unityComputePose();
    syncUnityToolButtons();
    unityScheduleRender();
    if (UNITY_SETTLE_TIMER) clearTimeout(UNITY_SETTLE_TIMER);
    UNITY_SETTLE_TIMER = setTimeout(unityEndInteraction, 180);
  }, { passive: false });
}

function renderGod2D() {
  const cv = $("#god-canvas");
  const wrap = $("#god-wrap");
  if (!cv || !wrap) return;
  const cv3d = $("#god3d-canvas");
  if (cv3d) cv3d.style.display = "none";
  const uimg = $("#god-unity-img");
  if (uimg) uimg.classList.add("hidden");
  setUnityOverlay("hidden");
  cv.classList.remove("hidden");
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(80, wrap.clientWidth);
  const H = Math.max(80, wrap.clientHeight);
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Same canonical payload as the Three.js 3D view. Playback interpolation is
  // applied to the shared episode truth before projection so both views animate
  // from identical coordinates.
  const canon = canonicalScene();
  if (MODE === "replay" && GOD_ANIM.active && GOD_ANIM.from && GOD_ANIM.to && canon.episode) {
    const p = lerpPose(GOD_ANIM.from, GOD_ANIM.to, GOD_ANIM.t);
    canon.episode.agent = p;
    const segs = canon.episode.trace_segments || [];
    if (segs.length && segs[segs.length - 1].length) {
      segs[segs.length - 1][segs[segs.length - 1].length - 1] = p.position;
    } else if (canon.episode.trace && canon.episode.trace.length) {
      canon.episode.trace[canon.episode.trace.length - 1] = p.position;
    }
  }
  const hasScene = !!(canon.scene && ((canon.scene.rooms || []).length || (canon.scene.walls || []).length));
  const hasEpisode = !!(canon.episode && (
    canon.episode.agent || (canon.episode.trace || []).length || (canon.episode.targets || []).length
  ));
  const sceneMissing = !!canon.scene_expected && !hasScene;
  if (sceneMissing) refreshScene3D(); // recover automatically once the payload arrives
  if (!hasScene && !hasEpisode) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#060a0f";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#4d5f70";
    ctx.font = "12px system-ui,sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(T("god.no_data"), W / 2, H / 2);
    if (sceneMissing) {
      ctx.fillStyle = "rgba(224,87,79,0.94)";
      ctx.fillRect(0, 0, W, 24);
      ctx.fillStyle = "#1a0b0a";
      ctx.font = "600 12px system-ui,sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(T("god.scene_unavailable"), 8, 16);
    }
    setText("#god-meta", "—");
    setText("#god-legend", "");
    return;
  }

  const model = Scene2D.buildScene2d(canon, { width: W, height: H });
  Scene2D.drawScene2d(ctx, model, {
    rooms: true,
    walls: !($("#ov-walls") && !$("#ov-walls").checked),
    objects: !($("#ov-objects") && !$("#ov-objects").checked),
    trajectory: !!($("#ov-trajectory") && $("#ov-trajectory").checked),
    fov: !!($("#ov-fov") && $("#ov-fov").checked),
    target: !!($("#ov-target") && $("#ov-target").checked),
  });
  if (sceneMissing) {
    // Never silently present an episode-only point plot as the full 2D map.
    ctx.fillStyle = "rgba(224,87,79,0.94)";
    ctx.fillRect(0, 0, W, 24);
    ctx.fillStyle = "#1a0b0a";
    ctx.font = "600 12px system-ui,sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(T("god.scene_unavailable"), 8, 16);
  }

  const ag = model.agent;
  const meta = [];
  meta.push(MODE === "replay" ? "#" + (EP_SELECTED ?? "—") : T("live.running"));
  meta.push(T("god.plane") + " " + model.plane);
  meta.push(T("god.source") + " " + (model.bounds_source === "scene" ? "scene" : "episode"));
  if (ag) {
    meta.push("pose " + fmtPos(ag.world) + " · yaw " + fmtNum(ag.yaw_deg) + "° · horizon " + fmtNum(ag.horizon_deg) + "°");
  }
  if (canon.episode && canon.episode.visible !== null && canon.episode.visible !== undefined) {
    meta.push("target " + (canon.episode.visible ? "●" : "○"));
  }
  setText("#god-meta", meta.join(" · "));
  setText("#god-legend", "▭ " + T("god.rooms") + " · ─ " + T("god.walls") + " · ▣ " + T("god.objects") + " · ● agent · ◆ " + T("overlay.target") + " · " + T("overlay.fov"));
}

// =====================================================================
// Canonical episode truth (shared by 3D and 2D)
// =====================================================================

function canonicalEpisode() {
  if (MODE === "replay" && EPISODE && !EPISODE.in_progress) {
    const h = EPISODE.header || {};
    const idx = EP_SELECTED ?? 0;
    const segments = buildTraceSegments(idx);
    const trace = flattenSegments(segments);
    let agent = (EP_STATE && EP_STATE.agent) || null;
    if (!agent && idx === 0 && h.spawn) {
      agent = { position: h.spawn.position, rotation_yaw_deg: h.spawn.yaw, camera_horizon_deg: h.spawn.horizon };
    }
    if (agent && agent.position && !samePos(trace[trace.length - 1], agent.position)) trace.push(agent.position);
    return {
      trace,
      trace_segments: segments,
      agent,
      targets: (h.target_positions || []).map((p) => ({ position: p, category: h.category })),
      spawn: h.spawn && h.spawn.position
        ? { position: h.spawn.position, yaw: h.spawn.yaw, horizon: h.spawn.horizon }
        : null,
      visible: EP_STATE ? EP_STATE.authoritative_target_visible : null,
      terminal: isTerminal(idx),
    };
  }
  if (MODE === "live" && EPISODE && EPISODE.in_progress) {
    const d = LIVE_DECISION || {};
    return {
      trace: LIVE_TRACE,
      agent: d.agent || null,
      targets: [],
      spawn: null,
      visible: d.authoritative_target_visible,
      terminal: false,
    };
  }
  if (STATE && STATE.active) {
    const pv = STATE.privileged || {};
    const taskP = pv.task || {};
    return {
      trace: LIVE_TRACE,
      agent: STATE.agent_state || null,
      targets: (taskP.target_positions || []).map((p) => ({ position: p, category: taskP.target_category })),
      spawn: null,
      visible: STATE.observation_meta ? STATE.observation_meta.target_visible : null,
      terminal: !!STATE.terminal,
    };
  }
  return { trace: [], agent: null, targets: [], spawn: null, visible: null, terminal: false };
}

let GOD3D_LAST_SIG = null;
let GOD3D_SCENE_SIG = null;
let GOD3D_OVERLAY_SIG = null;

function renderGod3D() {
  if (!GOD3D_READY) return;
  const cv3d = $("#god3d-canvas");
  const cv2d = $("#god-canvas");
  if (cv3d) cv3d.style.display = "";
  if (cv2d) cv2d.classList.add("hidden");
  const uimg = $("#god-unity-img");
  if (uimg) uimg.classList.add("hidden");
  setUnityOverlay("hidden");
  const overlaySig = [
    !!($("#ov-trajectory") && $("#ov-trajectory").checked),
    !!($("#ov-fov") && $("#ov-fov").checked),
    !!($("#ov-target") && $("#ov-target").checked),
  ].join("");
  if (overlaySig !== GOD3D_OVERLAY_SIG) {
    God3D.setOverlayFlags({
      trajectory: overlaySig[0] === "t",
      fov: overlaySig[1] === "t",
      target: overlaySig[2] === "t",
    });
    GOD3D_OVERLAY_SIG = overlaySig;
  }
  // Same canonical payload as the 2D view.
  const canon = canonicalScene();
  const ep = canon.episode;
  if (GOD_ANIM.active && GOD_ANIM.from && GOD_ANIM.to) {
    ep.agent = lerpPose(GOD_ANIM.from, GOD_ANIM.to, GOD_ANIM.t);
    const segs = ep.trace_segments || [];
    if (segs.length && segs[segs.length - 1].length) {
      segs[segs.length - 1][segs[segs.length - 1].length - 1] = ep.agent.position;
    } else if (ep.trace && ep.trace.length) {
      ep.trace[ep.trace.length - 1] = ep.agent.position;
    } else if (ep.trace) {
      ep.trace.push(ep.agent.position);
    }
  }
  const sceneSig = canon.scene ? JSON.stringify({
    src: canon.source,
    r: (canon.scene.rooms || []).length,
    w: (canon.scene.walls || []).length,
    o: (canon.scene.objects || []).length,
    b: canon.scene.bounds,
  }) : "none";
  if (sceneSig !== GOD3D_SCENE_SIG) {
    God3D.setScene(canon);
    GOD3D_SCENE_SIG = sceneSig;
    GOD3D_LAST_SIG = null;
  }
  const sig = JSON.stringify({
    i: EP_SELECTED,
    n: (ep.trace || []).length,
    a: ep.agent && ep.agent.position,
    v: ep.visible,
    t: ep.terminal,
    anim: GOD_ANIM.active ? Math.round(GOD_ANIM.t * 20) : 0,
  });
  if (sig !== GOD3D_LAST_SIG) {
    God3D.updateEpisode(ep);
    GOD3D_LAST_SIG = sig;
  }
  God3D.resize();
  God3D.render();
  renderGodMeta3D(ep);
}

function renderGodMeta3D(ep) {
  const meta = [];
  meta.push(MODE === "replay" ? "#" + (EP_SELECTED ?? "—") : T("live.running"));
  if (ep.agent && ep.agent.position) {
    meta.push("pose " + fmtPos(ep.agent.position) + " · yaw " + fmtNum(ep.agent.rotation_yaw_deg) + "° · horizon " + fmtNum(ep.agent.camera_horizon_deg) + "°");
  }
  if (ep.visible !== null && ep.visible !== undefined) meta.push("target " + (ep.visible ? "●" : "○"));
  const cs = God3D.cameraState ? God3D.cameraState() : null;
  if (cs) {
    meta.push((cs.mode === "follow" ? T("god.cam_follow") : T("god.unity_free"))
      + " · yaw " + fmtNum(cs.yaw) + "° · pitch " + fmtNum(cs.pitch) + "° · d " + fmtNum(cs.distance));
  }
  setText("#god-meta", meta.join(" · "));
  setText("#god-legend", "● agent · — " + T("overlay.trajectory") + " · ◆ " + T("overlay.target") + " · " + T("overlay.fov"));

  const sem = SCENE3D && SCENE3D.spawn_semantics;
  const semEl = $("#god-semantics");
  if (semEl) {
    if (sem) {
      semEl.textContent = sem.ok ? "● " + T("god.semantics_ok") : "✕ " + T("god.semantics_bad");
      semEl.className = "sem-flag " + (sem.ok ? "ok" : "bad");
      if (!sem.ok) semEl.title = JSON.stringify(sem.violations || []);
    } else {
      semEl.textContent = "—";
      semEl.className = "sem-flag";
    }
  }
  const srcEl = $("#god-source");
  if (srcEl) {
    if (SCENE3D && SCENE3D.scene) {
      const meta2 = SCENE3D.scene_meta || {};
      srcEl.textContent = T("god.source") + ": " + (meta2.source || SCENE3D.scene.source || "—")
        + (meta2.unavailable && meta2.unavailable.length ? " · " + T("god.unavailable") + ": " + meta2.unavailable.length : "");
    } else {
      srcEl.textContent = T("god.geometry_unavailable");
    }
  }
}

async function refreshScene3D() {
  // The canonical scene payload feeds BOTH the 3D view and the 2D diagnostic
  // view, so it must load even when WebGL/Three.js is unavailable.
  const key = currentSceneKey();
  if (!key) return;
  SCENE3D_KEY = key;
  const cached = SCENE3D_CACHE.get(key);
  if (cached) {
    SCENE3D = cached;
    EPISODE_SEMANTICS = cached.spawn_semantics || null;
    GOD3D_LAST_SIG = null;
    GOD3D_SCENE_SIG = null;
    renderGod();
    return;
  }
  if (SCENE3D_INFLIGHT.has(key)) return; // coalesce duplicate requests
  SCENE3D_INFLIGHT.add(key);
  const gen = ++SCENE3D_GEN;
  let payload = null;
  let fromError = false;
  try {
    if (key.startsWith("replay:")) {
      payload = await apiGet("/api/episodes/scene3d");
    } else {
      payload = await apiGet("/api/scene3d");
    }
  } catch (err) {
    payload = { scene: null, unavailable: [String(err.message || err)] };
    fromError = true;
  } finally {
    SCENE3D_INFLIGHT.delete(key);
  }
  // Latest request wins: an older response (e.g. a live /api/scene3d that
  // resolved after the replay scene) must never clobber the current episode.
  if (gen !== SCENE3D_GEN || key !== currentSceneKey()) return;
  SCENE3D = payload;
  if (!fromError) SCENE3D_CACHE.set(key, payload);
  EPISODE_SEMANTICS = payload && payload.spawn_semantics ? payload.spawn_semantics : null;
  GOD3D_LAST_SIG = null;
  GOD3D_SCENE_SIG = null; // force the canonical scene to be re-applied
  if (UNITY_CAM.mode === "overview") unityApplyPreset("overview"); // re-frame new house
  renderGod();
}

function setGodMode(mode) {
  const valid = ["unity", "3d", "2d"];
  GOD_MODE = valid.includes(mode) ? mode : "unity";
  if (GOD_MODE === "3d" && !GOD3D_READY) GOD_MODE = "unity";
  localStorage.setItem("sf.godview", GOD_MODE);
  $$("#god-view-modes button").forEach((b) => b.classList.toggle("active", b.dataset.godview === GOD_MODE));
  const unityTools = $("#god-unity-tools");
  if (unityTools) unityTools.style.display = GOD_MODE === "unity" ? "" : "none";
  const camTools = $("#god-cam-tools");
  if (camTools) camTools.style.display = GOD_MODE === "3d" ? "" : "none";
  if (GOD_MODE === "2d") {
    God3D.setVisible(false);
    stop3DLoop();
    if (!SCENE3D || SCENE3D_KEY !== currentSceneKey()) refreshScene3D();
    renderGod2D();
    return;
  }
  if (GOD_MODE === "unity") {
    God3D.setVisible(false);
    stop3DLoop();
    unityFollowStep();
    unityComputePose();
    syncUnityToolButtons();
    resetUnity();
    renderGod();
    return;
  }
  God3D.setVisible(true);
  start3DLoop();
  refreshScene3D();
}

function start3DLoop() {
  if (GOD3D_RAF) return;
  const frame = () => {
    GOD3D_RAF = requestAnimationFrame(frame);
    if (GOD_MODE === "3d" && GOD3D_READY) {
      if (GOD_ANIM.active) {
        const dt = performance.now() - GOD_ANIM.start;
        GOD_ANIM.t = Math.min(1, dt / Math.max(1, GOD_ANIM.duration));
        if (GOD_ANIM.t >= 1) GOD_ANIM.active = false;
      }
      renderGod3D();
    }
  };
  GOD3D_RAF = requestAnimationFrame(frame);
}

function stop3DLoop() {
  if (GOD3D_RAF) {
    cancelAnimationFrame(GOD3D_RAF);
    GOD3D_RAF = null;
  }
}

function renderHud() {
  const modelEl = $("#god-model");
  const teleEl = $("#god-telemetry");
  if (modelEl) {
    const m = (MODEL_INFO && (MODEL_INFO.model || MODEL_INFO)) || null;
    if (m && (m.model_id || m.model)) {
      modelEl.textContent = T("god.model") + ": " + (m.model_id || m.model)
        + (m.backend ? " · " + m.backend : "")
        + (m.precision ? " · " + m.precision : "")
        + (m.adapter ? " · adapter" : "")
        + (m.visual_tokens ? " · vis " + m.visual_tokens : "");
    } else {
      modelEl.textContent = T("god.model") + ": —";
    }
  }
  if (teleEl) {
    const t = TELEMETRY || {};
    const g = t.gpu || {};
    const r = t.runtime || {};
    const parts = [];
    if (typeof g.utilization_pct === "number") parts.push("GPU " + g.utilization_pct.toFixed(0) + "%");
    if (typeof g.memory_used_mb === "number" && g.memory_total_mb) {
      parts.push("VRAM " + (g.memory_used_mb / 1024).toFixed(1) + "/" + (g.memory_total_mb / 1024).toFixed(0) + "GB");
    }
    if (typeof g.power_w === "number") parts.push(g.power_w.toFixed(0) + "W");
    if (typeof r.env_steps_per_sec === "number") parts.push(r.env_steps_per_sec.toFixed(2) + " env steps/s");
    if (typeof r.episodes_per_hour === "number") parts.push(r.episodes_per_hour.toFixed(1) + " ep/h");
    if (typeof r.model_batch === "number") parts.push("batch " + r.model_batch);
    if (typeof r.decision_p50_ms === "number") parts.push("p50 " + r.decision_p50_ms.toFixed(0) + "ms");
    teleEl.textContent = T("god.telemetry") + ": " + (parts.length ? parts.join(" · ") : "—");
  }
}

async function refreshHud() {
  try {
    MODEL_INFO = await apiGet("/api/model");
  } catch (err) { /* keep previous */ }
  try {
    TELEMETRY = await apiGet("/api/telemetry");
  } catch (err) { /* keep previous */ }
  renderHud();
}

function interval() {
  return SPEED_MS;
}

function isTerminal(idx) {
  const e = EPISODE && EPISODE.timeline ? EPISODE.timeline[idx] : null;
  return !!(e && e.terminal);
}

function setupGodAnim(prevIdx, curIdx) {
  const from = poseAt(prevIdx);
  const to = poseAt(curIdx);
  if (!from || !to || !from.position || !to.position) {
    GOD_ANIM.active = false;
    return;
  }
  GOD_ANIM = { active: true, from, to, t: 0, start: performance.now(), duration: interval() };
}

function startGodLoop() {
  if (RAF_ID) return;
  const frame = () => {
    RAF_ID = requestAnimationFrame(frame);
    if (GOD_ANIM.active) {
      const dt = performance.now() - GOD_ANIM.start;
      GOD_ANIM.t = Math.min(1, dt / Math.max(1, GOD_ANIM.duration));
      if (GOD_ANIM.t >= 1) GOD_ANIM.active = false;
    }
    renderGod();
  };
  RAF_ID = requestAnimationFrame(frame);
}

function stopGodLoop() {
  if (RAF_ID) {
    cancelAnimationFrame(RAF_ID);
    RAF_ID = null;
  }
}

function scheduleTick() {
  if (PLAY_TIMER) clearTimeout(PLAY_TIMER);
  PLAY_TIMER = setTimeout(tick, interval());
}

// ---- curated / condensed replay ------------------------------------
function rebuildDisplayOrder() {
  if (!EPISODE || !EPISODE.timeline) {
    DISPLAY_ORDER = null;
    return;
  }
  const n = EPISODE.timeline.length;
  const cond = EPISODE.condensed;
  if (CONDENSED && cond && Array.isArray(cond.indices) && cond.indices.length) {
    DISPLAY_ORDER = cond.indices.filter((i) => i >= 0 && i < n);
  } else {
    DISPLAY_ORDER = null;
  }
}

function displayIndices() {
  if (DISPLAY_ORDER && DISPLAY_ORDER.length) return DISPLAY_ORDER;
  if (!EPISODE || !EPISODE.timeline) return [];
  return EPISODE.timeline.map((e) => e.idx);
}

function displayPos(idx) {
  const order = displayIndices();
  const p = order.indexOf(idx);
  return p < 0 ? 0 : p;
}

function renderCollapsedRuns() {
  const el = $("#ep-collapsed");
  if (!el) return;
  const cond = EPISODE && EPISODE.condensed;
  const segs = (CONDENSED && cond && cond.segments) || [];
  if (!segs.length) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  el.innerHTML = segs.map((s) =>
    '<span class="run">' + esc(T("ep.collapsed_run"))
      .replace("{action}", esc(s.action || "?"))
      .replace("{count}", String(s.count || 0))
      .replace("{disp}", fmtNum(s.displacement_m || 0)) + "</span>"
  ).join("");
  el.classList.remove("hidden");
}

function tick() {
  PLAY_TIMER = null;
  if (!PLAYING || !EPISODE || !EPISODE.timeline || !EPISODE.timeline.length) return;
  const order = displayIndices();
  const p = displayPos(EP_SELECTED);
  if (isTerminal(EP_SELECTED) || p >= order.length - 1) {
    pause();
    return;
  }
  selectStep(order[p + 1], { playback: true });
  if (isTerminal(EP_SELECTED)) {
    pause();
    return;
  }
  scheduleTick();
}

function play() {
  if (!EPISODE || !EPISODE.timeline || !EPISODE.timeline.length) return;
  if (MODE !== "replay") setMode("replay");
  const order = displayIndices();
  if (EP_SELECTED === null || displayPos(EP_SELECTED) >= order.length - 1) selectStep(order[0]);
  PLAYING = true;
  startGodLoop();
  scheduleTick();
  renderPlayer();
}

function pause() {
  PLAYING = false;
  if (PLAY_TIMER) {
    clearTimeout(PLAY_TIMER);
    PLAY_TIMER = null;
  }
  GOD_ANIM.active = false;
  stopGodLoop();
  renderPlayer();
}

function togglePlay() {
  if (PLAYING) pause();
  else play();
}

function stepPrev() {
  pause();
  const order = displayIndices();
  const p = displayPos(EP_SELECTED);
  if (EP_SELECTED !== null && p > 0) selectStep(order[p - 1]);
}

function stepNext() {
  pause();
  const order = displayIndices();
  if (!order.length) return;
  const p = displayPos(EP_SELECTED);
  if (EP_SELECTED !== null && p < order.length - 1) selectStep(order[p + 1]);
}

function setSpeed(v) {
  SPEED_MS = SPEED_TABLE[String(v)] || 500;
  if (PLAYING) {
    GOD_ANIM.duration = SPEED_MS;
    scheduleTick();
  }
  renderPlayer();
}

function selectStep(idx, opts) {
  opts = opts || {};
  if (!EPISODE || !EPISODE.timeline || !EPISODE.timeline.length) return;
  const n = EPISODE.timeline.length;
  const prev = EP_SELECTED;
  idx = Math.max(0, Math.min(Number(idx), n - 1));
  const cachedImg = FRAME_CACHE.get(idx);
  const cached = !!(cachedImg && cachedImg.complete && cachedImg.naturalWidth);
  const t0 = performance.now();
  EP_SELECTED = idx;
  EP_STATE = localStepState(idx);
  if (opts.playback && prev !== null && prev !== idx) setupGodAnim(prev, idx);
  else GOD_ANIM.active = false;
  unityFollowStep(); // Follow tracks the agent; Free/Overview are untouched
  preloadAround(idx);
  drawEpisodeFrame(idx);
  renderAll();
  (cached ? PERF.replayCached : PERF.replayUncached).push(performance.now() - t0);
}

// Coalesce rapid scrub input to at most one step selection per animation frame;
// the latest target wins so a fast drag never queues a backlog of fetches.
function queueScrub(displayValue) {
  SCRUB_TARGET = Number(displayValue);
  if (SCRUB_RAF) return;
  SCRUB_RAF = requestAnimationFrame(() => {
    SCRUB_RAF = null;
    const v = SCRUB_TARGET;
    SCRUB_TARGET = null;
    if (v === null || v === undefined || isNaN(v)) return;
    const order = displayIndices();
    selectStep(order[v] ?? v);
  });
}

function buildMarkers() {
  const strip = $("#ep-timeline");
  if (!strip || !EPISODE || !EPISODE.timeline) return;
  strip.innerHTML = "";
  const n = EPISODE.timeline.length;
  const decs = EPISODE.decisions || [];
  EPISODE.timeline.forEach((e) => {
    const m = document.createElement("span");
    m.className = "tl-marker"
      + (e.authoritative_target_visible ? " vis" : "")
      + (e.terminal ? " term" : "");
    if (decs.some((d) => d.frame === frameName(e.idx) && d.invalid)) m.className += " inv";
    m.style.left = (n > 1 ? (e.idx / (n - 1)) * 100 : 0) + "%";
    strip.appendChild(m);
  });
}

function renderPlayer() {
  const has = MODE === "replay" && EPISODE && (EPISODE.timeline || []).length > 0;
  const n = has ? EPISODE.timeline.length : 0;
  const scrub = $("#ep-scrub");
  const playBtn = $("#btn-play");
  const prevBtn = $("#btn-prev");
  const nextBtn = $("#btn-next");
  const speed = $("#play-speed");
  const counter = $("#step-counter");
  const fill = $("#scrub-fill");
  if (!scrub) return;
  [prevBtn, nextBtn, playBtn, speed, scrub].forEach((el) => {
    if (el) el.disabled = !has;
  });
  if (playBtn) {
    playBtn.textContent = PLAYING ? "⏸" : "▶";
    playBtn.title = T(PLAYING ? "player.pause" : "player.play");
  }
  if (prevBtn) prevBtn.title = T("player.prev");
  if (nextBtn) nextBtn.title = T("player.next");
  if (speed) speed.title = T("player.speed");
  const order = has ? displayIndices() : [];
  const dpos = has ? displayPos(EP_SELECTED) : 0;
  scrub.max = String(Math.max(0, order.length - 1));
  scrub.value = String(dpos);
  if (counter) {
    if (has) counter.textContent = dpos + " / " + Math.max(0, order.length - 1)
      + (order.length < n ? " (" + n + ")" : "");
    else if (EPISODE && EPISODE.in_progress) counter.textContent = "LIVE";
    else counter.textContent = "0 / 0";
  }
  renderCollapsedRuns();
  if (fill) {
    const pct = n > 1 ? ((EP_SELECTED ?? 0) / (n - 1)) * 100 : 0;
    fill.style.width = pct + "%";
  }
  const key = has && EPISODE.header ? EPISODE.header.episode_id : null;
  if (has && MARKERS_KEY !== key) {
    buildMarkers();
    MARKERS_KEY = key;
  } else if (!has && MARKERS_KEY !== null) {
    const strip = $("#ep-timeline");
    if (strip) strip.innerHTML = "";
    MARKERS_KEY = null;
  }
}

function metricsData() {
  if (MODE === "replay" && EPISODE) {
    const h = EPISODE.header || {};
    const decs = EPISODE.decisions || [];
    const tl = EPISODE.timeline || [];
    const lat = decs.map((d) => d.latency_ms).filter((x) => typeof x === "number");
    const dist = {};
    decs.forEach((d) => {
      const a = d.model_action || "(invalid)";
      dist[a] = (dist[a] || 0) + 1;
    });
    return {
      outcome: (h.status || "—") + (h.success === true ? " ✓" : h.success === false ? " ✕" : ""),
      steps: tl.filter((e) => e.type === "action").length,
      decisions: decs.length,
      invalid: decs.filter((d) => d.invalid).length,
      visible: decs.filter((d) => d.authoritative_target_visible).length,
      avg: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : null,
      terminal: h.model_terminal_reason || h.success_reason || "",
      spawn: h.spawn ? h.spawn.initially_visible : null,
      dist,
    };
  }
  if (MODE === "live" && STATE && STATE.active) {
    const s = STATE;
    return {
      outcome: ((s.task && s.task.status) || "—") + (s.task && s.task.success === true ? " ✓" : s.task && s.task.success === false ? " ✕" : ""),
      steps: (s.task && s.task.step) || 0,
      decisions: 0,
      invalid: 0,
      visible: s.observation_meta && s.observation_meta.target_visible ? 1 : 0,
      avg: null,
      terminal: (s.task && s.task.success_reason) || "",
      spawn: null,
      dist: {},
    };
  }
  if (MODE === "live" && EPISODE && EPISODE.in_progress) {
    const d = LIVE_DECISION || {};
    const dist = {};
    if (d.model_action) dist[d.model_action] = 1;
    return {
      outcome: (EPISODE.header && EPISODE.header.status) || "running",
      steps: (EPISODE.header && EPISODE.header.step) || 0,
      decisions: (d.decision_idx ?? -1) + 1,
      invalid: d.invalid ? 1 : 0,
      visible: d.authoritative_target_visible ? 1 : 0,
      avg: typeof d.latency_ms === "number" ? d.latency_ms : null,
      terminal: "",
      spawn: null,
      dist,
    };
  }
  return null;
}

function metricCard(label, value) {
  return '<div class="metric"><h4>' + esc(label) + '</h4><div class="v">' + esc(value) + "</div></div>";
}

function renderMetrics() {
  const box = $("#metrics-body");
  if (!box) return;
  const m = metricsData();
  if (!m) {
    box.innerHTML = metricCard(T("wb.metrics"), T("metrics.none"));
    return;
  }
  const rows = Object.entries(m.dist).sort((a, b) => b[1] - a[1]);
  const maxN = rows.length ? rows[0][1] : 1;
  const bars = rows.length
    ? rows.map(([a, n]) => '<div class="bar-row"><span class="bar-label">' + esc(a) + '</span><span class="bar-track"><span class="bar" style="width:' + Math.round(100 * n / maxN) + '%"></span></span><span class="bar-n">' + n + "</span></div>").join("")
    : "—";
  box.innerHTML =
    metricCard(T("metrics.outcome"), m.outcome) +
    metricCard(T("metrics.steps"), String(m.steps)) +
    metricCard(T("metrics.decisions"), String(m.decisions)) +
    metricCard(T("metrics.invalid"), String(m.invalid)) +
    metricCard(T("metrics.visible"), String(m.visible)) +
    metricCard(T("metrics.latency"), m.avg === null ? "—" : m.avg.toFixed(1) + " ms") +
    metricCard(T("metrics.terminal"), m.terminal || "—") +
    metricCard(T("metrics.spawn"), m.spawn === null || m.spawn === undefined ? "—" : (m.spawn ? T("common.yes") : T("common.no"))) +
    '<div class="metric wide"><h4>' + esc(T("metrics.actions")) + "</h4>" + bars + "</div>";
}

function privData() {
  if (MODE === "replay" && EPISODE) {
    const h = EPISODE.header || {};
    const st = EP_STATE || {};
    let ag = st.agent;
    if (!ag && EP_SELECTED === 0 && h.spawn) {
      ag = { position: h.spawn.position, rotation_yaw_deg: h.spawn.yaw, camera_horizon_deg: h.spawn.horizon };
    }
    ag = ag || {};
    return [
      ["house", h.house_id || "—"],
      ["target id", (h.target_object_ids || []).join(", ") || "—"],
      ["target pos", (h.target_positions || []).slice(0, 2).map(fmtPos).join(" | ") || "—"],
      ["agent pose", fmtPos(ag.position)],
      ["yaw / horizon", fmtNum(ag.rotation_yaw_deg) + "° / " + fmtNum(ag.camera_horizon_deg) + "°"],
      ["spawn visible", h.spawn ? String(h.spawn.initially_visible) : "—"],
      ["student domain", (h.student_domain || "—") + " · leak=" + (h.student_leak_checked || "—")],
      ["decisions / invalid", (h.decision_count ?? "—") + " / " + (h.invalid_decision_count ?? "—")],
      ["teacher plan", h.teacher_plan ? JSON.stringify(h.teacher_plan) : "—"],
    ];
  }
  if (MODE === "live" && STATE && STATE.active) {
    const pv = STATE.privileged || {};
    const t = pv.task || {};
    const ag = STATE.agent_state || {};
    return [
      ["task id", t.task_id || "—"],
      ["target id", (t.target_object_ids || []).join(", ") || "—"],
      ["target pos", (t.target_positions || []).slice(0, 2).map(fmtPos).join(" | ") || "—"],
      ["agent pose", fmtPos(ag.position)],
      ["reachable cells", String((t.reachable_positions || []).length)],
      ["teacher path", (t.teacher_path || []).join(" ") || "—"],
    ];
  }
  if (EPISODE && EPISODE.in_progress) {
    const h = EPISODE.header || {};
    return [
      ["episode", h.episode_id || "—"],
      ["goal", h.goal || "—"],
      ["status", h.status || "—"],
      ["step", String(h.step ?? "—")],
      ["decision idx", String(h.decision_idx ?? "—")],
    ];
  }
  return [["—", "—"]];
}

function renderPrivileged() {
  const dl = $("#priv-kv");
  if (!dl) return;
  dl.innerHTML = privData()
    .map(([k, v]) => "<dt>" + esc(k) + "</dt><dd>" + esc(v) + "</dd>")
    .join("");
}

function renderDebug() {
  const setJson = (id, obj) => {
    const el = $(id);
    if (el) el.textContent = obj === null || obj === undefined ? "—" : JSON.stringify(obj, null, 2);
  };
  if (MODE === "replay" && EPISODE) {
    const dec = decisionAt(EP_SELECTED);
    setJson("#dbg-priv-json", { header: EPISODE.header, step: EP_STATE });
    setJson("#dbg-model-json", dec ? { question: dec.question } : null);
    setJson("#dbg-teacher-json", STUDENT || null);
    return;
  }
  if (MODE === "live" && STATE && STATE.active) {
    setJson("#dbg-priv-json", STATE.privileged);
    setJson("#dbg-model-json", STATE.model_input);
    setJson("#dbg-teacher-json", TEACHER);
    return;
  }
  if (EPISODE && EPISODE.in_progress) {
    setJson("#dbg-priv-json", { header: EPISODE.header, live_decision: LIVE_DECISION });
    setJson("#dbg-model-json", null);
    setJson("#dbg-teacher-json", null);
    return;
  }
  setJson("#dbg-priv-json", null);
  setJson("#dbg-model-json", null);
  setJson("#dbg-teacher-json", null);
}

function renderManualStatus() {
  const el = $("#episode-status");
  const stateEl = $("#manual-state");
  if (stateEl) {
    if (STATE && STATE.active) {
      stateEl.className = "manual-state live";
      stateEl.textContent = T("manual.live") + " · " + (STATE.backend || "?")
        + " · " + T("god.step") + " " + ((STATE.task && STATE.task.step) ?? "—");
    } else if (EPISODE && !EPISODE.in_progress) {
      stateEl.className = "manual-state replay";
      stateEl.textContent = T("manual.replay");
    } else {
      stateEl.className = "manual-state";
      stateEl.textContent = "—";
    }
  }
  if (!el) return;
  if (STATE && STATE.active) {
    const st = STATE.terminal
      ? (STATE.task && STATE.task.success ? T("common.success") : T("common.failure"))
      : T("common.running");
    el.textContent = st + " · " + T("god.step") + " " + ((STATE.task && STATE.task.step) ?? "—")
      + (STATE.task && STATE.task.success_reason ? " · " + STATE.task.success_reason : "");
  } else {
    el.textContent = "—";
  }
}

// ---- custom episode picker (never a clipped native <select>) --------
function currentEpisodeId() {
  return EP_SELECTED_ID
    || (EPISODE && EPISODE.header && EPISODE.header.episode_id)
    || null;
}

function setSelectedEpisodeId(id) {
  EP_SELECTED_ID = id || null;
  const label = $("#ep-select-label");
  const ep = EPISODES.find((e) => e.episode_id === id);
  if (label) {
    label.textContent = ep
      ? [ep.episode_id, ep.house_id || "?", ep.category || "?", ep.status || "?"].join(" · ")
      : (id || "—");
  }
  $$("#ep-select-menu .ep-picker-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.episodeId === id);
  });
}

function openEpisodePicker(open) {
  EP_PICKER_OPEN = !!open;
  const menu = $("#ep-select-menu");
  const btn = $("#ep-select-btn");
  if (menu) menu.classList.toggle("hidden", !open);
  if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
}

function renderEpisodePicker() {
  const menu = $("#ep-select-menu");
  if (!menu) return;
  menu.innerHTML = "";
  if (!EPISODES.length) {
    menu.innerHTML = '<div class="ep-picker-empty">' + esc(T("ep.none")) + "</div>";
    return;
  }
  const validCount = EPISODES.filter((e) => (e.labels || []).includes("VALID")).length;
  if (validCount <= 1) {
    const note = document.createElement("div");
    note.className = "ep-picker-note";
    note.textContent = T("ep.one_valid");
    menu.appendChild(note);
  }
  EPISODES.forEach((e) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ep-picker-item" + (e.episode_id === currentEpisodeId() ? " active" : "");
    b.dataset.episodeId = e.episode_id;
    const labels = (e.labels || []).map((l) => {
      const cls = ["VALID", "SUCCESS"].includes(l) ? "ok"
        : ["LEGACY", "INVALID_SENSOR", "INVALID_SPAWN", "ANOMALY", "CRASH", "FALSE_DONE"].includes(l) ? "bad"
        : ["TIMEOUT", "STUCK", "POLICY_COLLAPSE"].includes(l) ? "warn" : "info";
      return '<span class="label-chip ' + cls + '">' + esc(l) + "</span>";
    }).join("");
    const meta = [
      e.house_id || "—", e.category || "?",
      (e.success === true ? "SUCCESS" : e.success === false ? "FAILURE" : (e.status || "?")),
      e.model_controlled ? "model" : "teacher",
      (e.steps ?? 0) + " " + T("god.step"),
    ].join(" · ");
    b.innerHTML = '<div class="epi-id">' + esc(e.episode_id)
      + (e.live ? ' <span class="label-chip info">LIVE</span>' : "") + "</div>"
      + '<div class="epi-meta">' + esc(meta) + "</div>"
      + (labels ? '<div class="epi-labels">' + labels + "</div>" : "");
    b.addEventListener("click", () => {
      setSelectedEpisodeId(e.episode_id);
      openEpisodePicker(false);
      loadEpisode();
    });
    menu.appendChild(b);
  });
  setSelectedEpisodeId(currentEpisodeId());
}

function renderEpisodeCards() {
  const box = $("#ep-cards");
  if (!box) return;
  box.innerHTML = "";
  if (!EPISODES.length) {
    box.innerHTML = '<div class="tl-empty">' + esc(T("ep.none")) + "</div>";
    return;
  }
  EPISODES.forEach((e) => {
    const c = document.createElement("div");
    const active = EPISODE && (
      (EPISODE.header && EPISODE.header.episode_id === e.episode_id) ||
      EPISODE.episode_id === e.episode_id
    );
    c.className = "ep-card" + (active ? " active" : "");
    const meta = [e.category || "?", e.status || "?", (e.steps ?? 0) + " " + T("god.step"), e.model_controlled ? "model" : "teacher"];
    const labelHtml = (e.labels || []).map((l) => {
      const cls = ["VALID", "SUCCESS"].includes(l) ? "ok"
        : ["LEGACY", "INVALID_SENSOR", "INVALID_SPAWN", "ANOMALY", "CRASH", "FALSE_DONE"].includes(l) ? "bad"
        : ["TIMEOUT", "STUCK", "POLICY_COLLAPSE"].includes(l) ? "warn" : "info";
      return '<span class="label-chip ' + cls + '">' + esc(l) + "</span>";
    }).join("");
    c.innerHTML = '<div class="ep-card-id">' + esc(e.episode_id) + "</div>"
      + '<div class="ep-card-meta">' + meta.map(esc).join(" · ") + "</div>"
      + (labelHtml ? '<div class="label-chips" style="margin-top:4px">' + labelHtml + "</div>" : "")
      + (e.model_controlled
        ? '<div class="ep-card-sub">decisions ' + (e.decision_count ?? 0) + " · invalid " + (e.invalid_decision_count ?? 0)
          + (e.model_terminal_reason ? " · " + esc(e.model_terminal_reason) : "") + "</div>"
        : "");
    c.addEventListener("click", () => {
      setSelectedEpisodeId(e.episode_id);
      openDrawer(null);
      loadEpisode();
    });
    box.appendChild(c);
  });
}

function openDrawer(name) {
  const drawer = $("#drawer");
  if (!drawer) return;
  if (!name || DRAWER === name) {
    DRAWER = null;
    drawer.classList.remove("open");
    $$(".dtab").forEach((t) => t.classList.remove("active"));
    return;
  }
  DRAWER = name;
  drawer.classList.add("open");
  $$(".dtab").forEach((t) => t.classList.toggle("active", t.dataset.drawer === name));
  $$(".dpanel").forEach((p) => p.classList.toggle("active", p.id === "dp-" + name));
  if (name === "debug") renderDebug();
  if (name === "metrics") renderMetrics();
}

function sizeFpv() {
  const wrap = $("#fpv-wrap");
  if (!wrap || !wrap.parentElement) return;
  const avail = Math.max(160, wrap.parentElement.clientWidth - 20);
  const maxH = Math.max(200, Math.round(window.innerHeight * 0.38));
  const size = Math.round(Math.min(avail, maxH));
  wrap.style.width = size + "px";
  wrap.style.height = size + "px";
  const cv = $("#obs-canvas");
  const dpr = window.devicePixelRatio || 1;
  if (cv && (cv.width !== Math.round(size * dpr) || cv.height !== Math.round(size * dpr))) {
    cv.width = Math.round(size * dpr);
    cv.height = Math.round(size * dpr);
  }
}

function paintFrame(img) {
  const cv = $("#obs-canvas");
  if (!cv || !img || !img.naturalWidth) return;
  const ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
}

function onceImage(img, fn) {
  const handler = () => {
    img.removeEventListener("load", handler);
    img.removeEventListener("error", handler);
    fn();
  };
  img.addEventListener("load", handler);
  img.addEventListener("error", handler);
}

function preloadFrame(idx) {
  if (!EPISODE || !EPISODE.timeline || idx < 0 || idx >= EPISODE.timeline.length) return null;
  if (FRAME_CACHE.has(idx)) return FRAME_CACHE.get(idx);
  const img = new Image();
  img._t0 = performance.now();
  img.addEventListener("load", () => {
    if (img.naturalWidth) PERF.fpv.push(performance.now() - img._t0);
  });
  FRAME_CACHE.set(idx, img);
  while (FRAME_CACHE.size > FRAME_CACHE_MAX) {
    const oldest = FRAME_CACHE.keys().next().value;
    const evicted = FRAME_CACHE.get(oldest);
    if (!evicted || evicted === img) break;
    try { evicted.src = ""; } catch (e) { /* ignore */ }
    FRAME_CACHE.delete(oldest);
  }
  img.src = "/api/episodes/frame?step=" + idx;
  return img;
}

function preloadAround(idx) {
  for (let d = -3; d <= 10; d++) preloadFrame(idx + d);
}

function preloadAll() {
  const gen = ++PRELOAD_GEN;
  const n = EPISODE && EPISODE.timeline ? EPISODE.timeline.length : 0;
  const step = (i) => {
    if (gen !== PRELOAD_GEN || i >= n) return;
    const img = preloadFrame(i);
    const next = () => {
      if (gen === PRELOAD_GEN) setTimeout(() => step(i + 1), 0);
    };
    if (!img || (img.complete && img.naturalWidth)) next();
    else onceImage(img, next);
  };
  step(0);
}

function drawEpisodeFrame(idx) {
  const img = preloadFrame(idx);
  if (!img) return;
  const paint = () => {
    if (EP_SELECTED === idx) paintFrame(img);
  };
  if (img.complete && img.naturalWidth) paint();
  else onceImage(img, paint);
}

function drawFrameFrom(url) {
  const cv = $("#obs-canvas");
  if (!cv) return;
  const img = new Image();
  img.onload = () => paintFrame(img);
  img.onerror = () => {
    const ctx = cv.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#0b1117";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#5f7386";
    ctx.font = Math.round(cv.width / 18) + "px system-ui,sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(T("fpv.none"), cv.width / 2, cv.height / 2);
  };
  img.src = url;
}

function pushTrace(p) {
  if (!Array.isArray(p)) return;
  const last = LIVE_TRACE[LIVE_TRACE.length - 1];
  if (last && last[0] === p[0] && last[1] === p[1] && last[2] === p[2]) return;
  LIVE_TRACE.push(p);
}

async function refreshEpisodeList(opts) {
  const autoload = opts && opts.autoload;
  const silent = opts && opts.silent;
  let data = { episodes: [] };
  try {
    data = await apiGet("/api/episodes/list");
  } catch (err) {
    if (silent) return;
    data = { episodes: [] };
  }
  const previous = currentEpisodeId();
  EPISODES = data.episodes || [];
  let selected = previous && EPISODES.some((e) => e.episode_id === previous) ? previous : null;
  if (!selected && EPISODE && EPISODE.header && EPISODE.header.episode_id) {
    selected = EPISODE.header.episode_id;
  }
  if (!selected && EPISODES.length) selected = EPISODES[0].episode_id;
  setSelectedEpisodeId(selected);
  renderEpisodePicker();
  renderEpisodeCards();
  if (autoload && !EPISODE && EPISODES.length) {
    // server returns episodes already curated (researcher-useful post-fix
    // first); never auto-load a legacy over-exposed or collapsed rollout.
    const preferred =
      EPISODES.find((e) => e.flags && e.flags.researcher_useful) ||
      EPISODES.find((e) => (e.labels || []).includes("VALID")) ||
      EPISODES.find((e) => !e.live) ||
      EPISODES[0];
    setSelectedEpisodeId(preferred.episode_id);
    await loadEpisode();
  }
}

async function loadEpisode() {
  const id = currentEpisodeId();
  if (!id) return;
  let data;
  try {
    data = await apiPost("/api/episodes/load", { episode_id: id });
  } catch (err) {
    setText("#ep-header", String(err.message || err));
    return;
  }
  setSelectedEpisodeId(id);
  if (data.in_progress) {
    pause();
    EPISODE = data;
    EP_STATE = null;
    EP_SELECTED = null;
    STUDENT = null;
    LIVE_DECISION = data.live_decision || null;
    LIVE_TRACE = [];
    if (LIVE_DECISION && LIVE_DECISION.agent) pushTrace(LIVE_DECISION.agent.position);
    $("#ep-live").checked = true;
    startPoll();
    setMode("live");
    drawFrameFrom("/api/episodes/frame?frame=" + encodeURIComponent(data.live_frame || "frame_0000.png") + "&t=" + Date.now());
    refreshScene3D();
    renderAll();
    return;
  }
  pause();
  EPISODE = data;
  STUDENT = data.student_record || null;
  LIVE_DECISION = null;
  $("#ep-live").checked = false;
  stopPoll();
  FRAME_CACHE = new Map();
  PRELOAD_GEN++;
  MARKERS_KEY = null;
  resetUnity();
  rebuildDisplayOrder();
  setMode("replay");
  selectStep(0);
  preloadAround(0);
  preloadAll();
  refreshScene3D();
  // Hide the Unity engine/house startup cost off the interactive path.
  apiGet("/api/unity/prewarm?episode_id=" + encodeURIComponent(id)).catch(() => {});
  renderAll();
}

function startPoll() {
  if (POLL_TIMER) return;
  POLL_TIMER = setInterval(pollLive, 2000);
}

function stopPoll() {
  if (POLL_TIMER) {
    clearInterval(POLL_TIMER);
    POLL_TIMER = null;
  }
}

async function pollLive() {
  const liveBox = $("#ep-live");
  if (!liveBox || !liveBox.checked || !EPISODE) return;
  const id = (EPISODE.header && EPISODE.header.episode_id) || currentEpisodeId();
  if (!id) return;
  let data;
  try {
    data = await apiPost("/api/episodes/load", { episode_id: id });
  } catch (err) {
    return;
  }
  if (data.in_progress) {
    EPISODE = data;
    LIVE_DECISION = data.live_decision || null;
    if (LIVE_DECISION && LIVE_DECISION.agent) pushTrace(LIVE_DECISION.agent.position);
    drawFrameFrom("/api/episodes/frame?frame=" + encodeURIComponent(data.live_frame || "frame_0000.png") + "&t=" + Date.now());
    renderAll();
    return;
  }
  stopPoll();
  liveBox.checked = false;
  await loadEpisode();
}

async function refreshHouses() {
  const sel = $("#house-select");
  if (!sel) return;
  try {
    const data = await apiGet("/api/houses");
    const houses = data.houses || [];
    sel.innerHTML = "";
    houses.forEach((h) => {
      const o = document.createElement("option");
      o.value = h.house_id;
      o.textContent = h.house_id;
      sel.appendChild(o);
    });
    if (!houses.length) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = T("god.geometry_unavailable");
      sel.appendChild(o);
    }
  } catch (err) {
    sel.innerHTML = '<option value="">—</option>';
  }
}

async function startEpisode() {
  const target = $("#target-input").value.trim() || "mug";
  const backend = $("#backend-select").value;
  const maxSteps = parseInt($("#set-maxsteps").value, 10) || 300;
  const houseId = ($("#house-select") && $("#house-select").value) || "";
  setText("#episode-status", T("common.running") + "…");
  let data;
  try {
    data = await apiPost("/api/episode/start", {
      target_category: target, backend, max_steps: maxSteps, seed: 0, house_id: houseId,
    });
  } catch (err) {
    setText("#episode-status", String(err.message || err));
    return;
  }
  pause();
  // Manual control is a distinct session: drop any replay episode so the two
  // state machines can never be mixed.
  EPISODE = null;
  EP_SELECTED = null;
  EP_STATE = null;
  STATE = data;
  TEACHER = null;
  LIVE_TRACE = [];
  if (STATE.agent_state) pushTrace(STATE.agent_state.position);
  stopPoll();
  setMode("live");
  unityFollowStep();
  unityComputePose();
  syncUnityToolButtons();
  drawFrameFrom("/api/frame?t=" + Date.now());
  refreshScene3D();
  renderAll();
}

async function doAction(act) {
  if (!STATE || !STATE.active) return;
  try {
    STATE = await apiPost("/api/action/step", { action_type: act });
  } catch (err) {
    setText("#episode-status", String(err.message || err));
    return;
  }
  if (STATE.agent_state) pushTrace(STATE.agent_state.position);
  unityFollowStep();
  drawFrameFrom("/api/frame?t=" + Date.now());
  renderAll();
}

async function doTeacher() {
  if (!STATE || !STATE.active) return;
  try {
    TEACHER = await apiGet("/api/teacher");
  } catch (err) {
    TEACHER = { error: String(err.message || err) };
  }
  renderAll();
}

function showJsonModal() {
  const el = $("#model-json-modal");
  let text = "";
  if (MODE === "live" && STATE) {
    text = "MODEL INPUT (permitted only)\n" + JSON.stringify(STATE.model_input, null, 2)
      + "\n\nPRIVILEGED (never model input)\n" + JSON.stringify(STATE.privileged, null, 2);
  } else if (MODE === "replay" && EPISODE) {
    const dec = decisionAt(EP_SELECTED);
    text = "REPLAY STEP " + EP_SELECTED + " (researcher-only)\n"
      + JSON.stringify({ header: EPISODE.header, step: EP_STATE, decision: dec }, null, 2);
  } else {
    text = JSON.stringify({ state: STATE, episode: EPISODE }, null, 2);
  }
  el.textContent = text;
  el.classList.toggle("hidden");
}

function bind() {
  $("#lang-select").addEventListener("change", (e) => setLang(e.target.value));
  $("#settings-lang").addEventListener("change", (e) => setLang(e.target.value));
  $("#btn-mode-live").addEventListener("click", () => {
    if (EPISODE && EPISODE.in_progress) startPoll();
    setMode("live");
  });
  $("#btn-mode-replay").addEventListener("click", () => {
    if (!EPISODE || !EPISODE.timeline || !EPISODE.timeline.length) openDrawer("episodes");
    setMode("replay");
  });
  $("#btn-ep-list").addEventListener("click", () => refreshEpisodeList());
  $("#btn-ep-load").addEventListener("click", loadEpisode);
  const epBtn = $("#ep-select-btn");
  if (epBtn) epBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    openEpisodePicker(!EP_PICKER_OPEN);
  });
  document.addEventListener("click", (e) => {
    if (!EP_PICKER_OPEN) return;
    const picker = $("#ep-picker");
    if (picker && !picker.contains(e.target)) openEpisodePicker(false);
  });
  $("#ep-live").addEventListener("change", (e) => {
    if (e.target.checked) startPoll();
    else stopPoll();
  });
  $("#btn-play").addEventListener("click", togglePlay);
  $("#btn-prev").addEventListener("click", stepPrev);
  $("#btn-next").addEventListener("click", stepNext);
  $("#play-speed").addEventListener("change", (e) => setSpeed(e.target.value));
  $("#ep-scrub").addEventListener("input", (e) => {
    pause();
    queueScrub(e.target.value);
  });
  $$(".dtab").forEach((t) => t.addEventListener("click", () => openDrawer(t.dataset.drawer)));
  $$(".act").forEach((b) => b.addEventListener("click", () => doAction(b.dataset.act)));
  $("#btn-start").addEventListener("click", startEpisode);
  $("#btn-teacher").addEventListener("click", doTeacher);
  $("#btn-view-json").addEventListener("click", showJsonModal);
  ["#ov-trajectory", "#ov-fov", "#ov-target", "#ov-walls", "#ov-objects"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", () => {
      if (GOD3D_READY) {
        God3D.setLayerVisibility({
          walls: !!($("#ov-walls") && $("#ov-walls").checked),
          objects: !!($("#ov-objects") && $("#ov-objects").checked),
        });
      }
      renderGod();
    });
  });
  $$("#god-view-modes button").forEach((b) => {
    b.addEventListener("click", () => setGodMode(b.dataset.godview));
  });
  $$("#god-unity-tools button").forEach((b) => {
    b.addEventListener("click", () => {
      const v = b.dataset.uv;
      if (v === "reset") {
        // Reset restores the current preset (Follow or Overview).
        unityApplyPreset(UNITY_CAM.preset);
      } else {
        unityApplyPreset(v === "overview" ? "overview" : "follow");
      }
      localStorage.setItem("sf.unityview", UNITY_VIEW);
      resetUnity();
      renderGod();
    });
  });
  $$("#god-cam-tools button").forEach((b) => {
    b.addEventListener("click", () => {
      if (GOD3D_READY) {
        God3D.view(b.dataset.cam);
        $$("#god-cam-tools button").forEach((x) => x.classList.toggle("active", x === b));
        renderGodMeta3D(canonicalEpisode());
      }
    });
  });
  const cond = $("#ep-condensed");
  if (cond) {
    cond.checked = CONDENSED;
    cond.addEventListener("change", (e) => {
      CONDENSED = !!e.target.checked;
      localStorage.setItem("sf.condensed", CONDENSED ? "1" : "0");
      rebuildDisplayOrder();
      renderPlayer();
    });
  }
  $("#model-json-modal").addEventListener("click", () => $("#model-json-modal").classList.add("hidden"));
  bindUnityCamera();
  window.addEventListener("resize", () => {
    sizeFpv();
    if (GOD3D_READY) God3D.resize();
    renderGod();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("#model-json-modal").classList.add("hidden");
      if (EP_PICKER_OPEN) openEpisodePicker(false);
      return;
    }
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      togglePlay();
      return;
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      stepNext();
      return;
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepPrev();
    }
  });
}

async function boot() {
  await loadI18n();
  bind();
  sizeFpv();
  await refreshUnityStatus();
  initGod3D();
  try {
    const info = await apiPost("/api/house/load", { backend: "deterministic", seed: 0, house_id: "" });
    HOUSE = info.house || null;
  } catch (err) {
    HOUSE = null;
  }
  try {
    await refreshHouses();
  } catch (err) { /* houses are optional */ }
  try {
    await refreshEpisodeList({ autoload: true });
  } catch (err) {
    EPISODES = [];
  }
  refreshHud();
  if (EP_LIST_TIMER) clearInterval(EP_LIST_TIMER);
  EP_LIST_TIMER = setInterval(() => {
    refreshEpisodeList({ silent: true });
  }, 10000);
  if (HUD_TIMER) clearInterval(HUD_TIMER);
  HUD_TIMER = setInterval(refreshHud, 5000);
  renderAll();
}

function initGod3D() {
  const canvas = $("#god3d-canvas");
  GOD3D_READY = !!(canvas && God3D.isAvailable() && God3D.init(canvas));
  if (GOD3D_READY && God3D.setUserInteractHandler) {
    God3D.setUserInteractHandler(() => {
      // User grabbed the camera: switch the tool UI to Free/Perspective.
      $$("#god-cam-tools button").forEach((b) => {
        b.classList.toggle("active", b.dataset.cam === "perspective");
      });
      renderGodMeta3D(canonicalEpisode());
    });
  }
  setGodMode(GOD_MODE);
}

async function refreshUnityStatus() {
  try {
    UNITY_STATUS = await apiGet("/api/unity/status");
  } catch (err) {
    UNITY_STATUS = { available: false, reason: String(err.message || err) };
  }
}

boot();
