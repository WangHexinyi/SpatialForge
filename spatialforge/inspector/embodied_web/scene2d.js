// SpatialForge God View — canonical top-down 2D projection.
//
// This module is the single source of truth for the 2D diagnostic view. It is
// deliberately pure (no imports, no DOM) so that:
//   * the browser 2D view derives *only* from the same canonical scene payload
//     that the Three.js 3D view consumes, and
//   * the cross-language consistency tests can execute it under node and assert
//     that 2D projected coordinates match the 3D world coordinates.
//
// Coordinate convention (matches Three.js / AI2-THOR): world is Y-up, the
// ground plane is X/Z. The top-down projection is therefore (x, z) with the
// screen Y axis flipped so +Z points down on screen. No independent coordinate
// normalization is allowed here: the viewport is fitted from the scene bounds
// (or the episode points only when no scene geometry exists).

export const SCENE2D_SCHEMA = "spatialforge_scene2d.v1";

function isFiniteNum(n) {
  return typeof n === "number" && isFinite(n);
}

function finiteBounds(b) {
  return !!(b && Array.isArray(b.min) && Array.isArray(b.max)
    && b.min.length >= 3 && b.max.length >= 3
    && isFiniteNum(b.min[0]) && isFiniteNum(b.min[2])
    && isFiniteNum(b.max[0]) && isFiniteNum(b.max[2]));
}

export function boundsFromEpisode(episode) {
  const pts = [];
  const push = (p) => { if (Array.isArray(p) && p.length >= 3 && isFiniteNum(p[0]) && isFiniteNum(p[2])) pts.push(p); };
  (episode && episode.trace ? episode.trace : []).forEach(push);
  const ag = episode && episode.agent;
  if (ag && ag.position) push(ag.position);
  const sp = episode && episode.spawn;
  if (sp && sp.position) push(sp.position);
  (episode && episode.targets ? episode.targets : []).forEach((t) => t && t.position && push(t.position));
  if (!pts.length) return { min: [0, 0, 0], max: [1, 0, 1] };
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  pts.forEach((p) => {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[2] < minZ) minZ = p[2];
    if (p[2] > maxZ) maxZ = p[2];
  });
  return { min: [minX, 0, minZ], max: [maxX, 0, maxZ] };
}

export function makeTransform(bounds, width, height, padFrac) {
  const minX = bounds.min[0];
  const maxX = bounds.max[0];
  const minZ = bounds.min[2];
  const maxZ = bounds.max[2];
  const spanX = Math.max(maxX - minX, 1e-3);
  const spanZ = Math.max(maxZ - minZ, 1e-3);
  const pad = Math.max(0.6, Math.max(spanX, spanZ) * (padFrac === undefined ? 0.06 : padFrac));
  const x0 = minX - pad;
  const x1 = maxX + pad;
  const z0 = minZ - pad;
  const z1 = maxZ + pad;
  const scale = Math.min(width / (x1 - x0), height / (z1 - z0));
  const offX = (width - (x1 - x0) * scale) / 2;
  const offY = (height - (z1 - z0) * scale) / 2;
  return {
    plane: "xz",
    scale,
    offX,
    offY,
    minX: x0,
    maxX: x1,
    minZ: z0,
    maxZ: z1,
    width,
    height,
    project(p) {
      return [offX + (p[0] - x0) * scale, height - offY - (p[2] - z0) * scale];
    },
  };
}

export function projectScenePoint(model, p) {
  const t = model.transform;
  return [t.offX + (p[0] - t.minX) * t.scale, t.height - t.offY - (p[2] - t.minZ) * t.scale];
}

function polygon(points, project) {
  return (points || []).map(project);
}

function objectFootprint(obj, project) {
  const aabb = obj && obj.aabb;
  if (aabb && Array.isArray(aabb.center) && Array.isArray(aabb.size)) {
    const [cx, cy, cz] = aabb.center;
    const [sx, sy, sz] = aabb.size;
    const hx = Math.max(sx, 0.02) / 2;
    const hz = Math.max(sz, 0.02) / 2;
    return [
      project([cx - hx, cy, cz - hz]),
      project([cx + hx, cy, cz - hz]),
      project([cx + hx, cy, cz + hz]),
      project([cx - hx, cy, cz + hz]),
    ];
  }
  return null;
}

function openingSegment(opening, project) {
  if (!opening || !opening.position || !isFiniteNum(opening.width)) return null;
  const [x, y, z] = opening.position;
  const yaw = ((opening.yaw_deg || 0) * Math.PI) / 180;
  const half = Math.max(opening.width, 0.05) / 2;
  const dx = Math.sin(yaw) * half;
  const dz = Math.cos(yaw) * half;
  return [project([x - dx, y, z - dz]), project([x + dx, y, z + dz])];
}

/**
 * Build the complete 2D display model from the canonical scene payload.
 * @param {{scene?:object, episode?:object, source?:string, unavailable?:string[]}} canonical
 * @param {{width:number, height:number}} viewport
 */
export function buildScene2d(canonical, viewport) {
  const scene = (canonical && canonical.scene) || null;
  const episode = (canonical && canonical.episode) || {};
  const width = Math.max(80, (viewport && viewport.width) || 640);
  const height = Math.max(80, (viewport && viewport.height) || 480);

  let bounds = scene && scene.bounds;
  let boundsSource = "scene";
  if (!finiteBounds(bounds)) {
    bounds = boundsFromEpisode(episode);
    boundsSource = "episode";
  }
  const t = makeTransform(bounds, width, height, 0.06);
  const project = (p) => t.project(p);

  const rooms = ((scene && scene.rooms) || []).map((r) => ({
    id: r.id,
    type: r.type,
    floor_y: r.floor_y,
    polygon: polygon(r.floor_polygon, project),
  }));

  const walls = ((scene && scene.walls) || []).map((w) => ({
    id: w.id,
    room_id: w.room_id,
    empty: !!w.empty,
    height: w.height,
    length: w.length,
    base: polygon(w.base, project),
  }));

  const mapOpening = (o) => ({
    id: o.id,
    kind: o.kind,
    position: o.position || null,
    width: o.width,
    yaw_deg: o.yaw_deg,
    segment: openingSegment(o, project),
  });
  const doors = ((scene && scene.doors) || []).map(mapOpening);
  const windows = ((scene && scene.windows) || []).map(mapOpening);

  const objects = ((scene && scene.objects) || []).map((o) => ({
    id: o.id,
    category: o.category,
    position: o.position || null,
    aabb: o.aabb || null,
    footprint: objectFootprint(o, project),
    point: o.position ? project(o.position) : null,
  }));

  const segs = (episode.trace_segments && episode.trace_segments.length)
    ? episode.trace_segments
    : (episode.trace && episode.trace.length ? [episode.trace] : []);
  const trajectory = {
    segments: segs.map((seg) => (seg || []).map(project)),
    points: (episode.trace || []).map(project),
  };

  const agent = (episode.agent && episode.agent.position)
    ? {
        world: episode.agent.position.slice(),
        projected: project(episode.agent.position),
        yaw_deg: Number(episode.agent.rotation_yaw_deg) || 0,
        horizon_deg: Number(episode.agent.camera_horizon_deg) || 0,
      }
    : null;

  const spawn = (episode.spawn && episode.spawn.position)
    ? { world: episode.spawn.position.slice(), projected: project(episode.spawn.position) }
    : null;

  const targets = (episode.targets || [])
    .filter((x) => x && x.position)
    .map((x) => ({
      world: x.position.slice(),
      projected: project(x.position),
      object_id: x.object_id === undefined ? null : x.object_id,
      category: x.category === undefined ? null : x.category,
      visible: episode.visible,
    }));

  const terminal = (episode.terminal && agent)
    ? { world: agent.world.slice(), projected: agent.projected.slice() }
    : null;

  return {
    schema: SCENE2D_SCHEMA,
    bounds,
    bounds_source: boundsSource,
    plane: "xz",
    transform: {
      scale: t.scale,
      offX: t.offX,
      offY: t.offY,
      minX: t.minX,
      maxX: t.maxX,
      minZ: t.minZ,
      maxZ: t.maxZ,
      width,
      height,
    },
    source: (canonical && canonical.source) || (scene && scene.source) || null,
    unavailable: (canonical && canonical.unavailable) || (scene && scene.unavailable) || [],
    counts: {
      rooms: rooms.length,
      walls: walls.length,
      doors: doors.length,
      windows: windows.length,
      objects: objects.length,
      trace_points: (episode.trace || []).length,
      trajectory_segments: trajectory.segments.length,
      targets: targets.length,
    },
    rooms,
    walls,
    doors,
    windows,
    objects,
    trajectory,
    agent,
    spawn,
    targets,
    terminal,
  };
}

// ---------------------------------------------------------------------------
// Drawing (browser only; pure canvas 2D API, no external deps)
// ---------------------------------------------------------------------------

const COLOR = {
  bg: "#060a0f",
  grid: "rgba(74,163,255,0.07)",
  room: ["rgba(43,58,74,0.55)", "rgba(51,69,90,0.55)"],
  roomEdge: "rgba(143,163,184,0.45)",
  wall: "rgba(143,163,184,0.85)",
  wallEmpty: "rgba(185,199,214,0.4)",
  door: "#b98a4b",
  window: "#6fc2e8",
  object: "rgba(92,113,137,0.55)",
  objectEdge: "rgba(127,149,171,0.6)",
  target: "#e0574f",
  targetVisible: "#37c07a",
  agent: "#4aa3ff",
  spawn: "#e8b23a",
  trajectory: "rgba(74,163,255,0.65)",
  terminal: "#d05ad0",
};

function drawPolygon(ctx, pts, fill, stroke, lineWidth) {
  if (!pts || pts.length < 3) return;
  ctx.beginPath();
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth || 1; ctx.stroke(); }
}

function drawSegment(ctx, seg, stroke, lineWidth) {
  if (!seg || seg.length < 2) return;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth || 1;
  ctx.beginPath();
  ctx.moveTo(seg[0][0], seg[0][1]);
  ctx.lineTo(seg[1][0], seg[1][1]);
  ctx.stroke();
}

/**
 * Draw the 2D diagnostic view. Returns the model (for tests / metadata).
 */
export function drawScene2d(ctx, model, flags) {
  const f = Object.assign(
    { rooms: true, walls: true, objects: true, trajectory: true, fov: true, target: true },
    flags || {}
  );
  const W = model.transform.width;
  const H = model.transform.height;
  // The caller owns the device-pixel-ratio transform; draw in CSS pixels.
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = COLOR.bg;
  ctx.fillRect(0, 0, W, H);

  // grid in world coordinates
  const t = model.transform;
  const grid = niceStep(Math.max(t.maxX - t.minX, t.maxZ - t.minZ));
  ctx.strokeStyle = COLOR.grid;
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(t.minX / grid) * grid; gx <= t.maxX; gx += grid) {
    const x = t.offX + (gx - t.minX) * t.scale;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let gz = Math.ceil(t.minZ / grid) * grid; gz <= t.maxZ; gz += grid) {
    const y = H - t.offY - (gz - t.minZ) * t.scale;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  if (f.rooms) {
    model.rooms.forEach((r, i) => {
      drawPolygon(ctx, r.polygon, COLOR.room[i % 2], COLOR.roomEdge, 1);
    });
  }

  if (f.walls) {
    model.walls.forEach((w) => {
      drawSegment(ctx, w.base, w.empty ? COLOR.wallEmpty : COLOR.wall, w.empty ? 1 : 2.4);
    });
  }
  model.doors.forEach((d) => drawSegment(ctx, d.segment, COLOR.door, 3));
  model.windows.forEach((w) => drawSegment(ctx, w.segment, COLOR.window, 2.4));

  if (f.objects) {
    model.objects.forEach((o) => {
      if (o.footprint) drawPolygon(ctx, o.footprint, COLOR.object, COLOR.objectEdge, 1);
      else if (o.point) {
        ctx.fillStyle = COLOR.objectEdge;
        ctx.beginPath(); ctx.arc(o.point[0], o.point[1], 2, 0, Math.PI * 2); ctx.fill();
      }
    });
  }

  if (f.trajectory && model.trajectory.segments.length) {
    ctx.strokeStyle = COLOR.trajectory;
    ctx.lineWidth = 1.6;
    model.trajectory.segments.forEach((seg) => {
      if (seg.length < 2) return;
      ctx.beginPath();
      seg.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
      ctx.stroke();
    });
    ctx.fillStyle = "rgba(74,163,255,0.5)";
    model.trajectory.points.forEach((p) => {
      ctx.beginPath(); ctx.arc(p[0], p[1], 1.8, 0, Math.PI * 2); ctx.fill();
    });
  }

  if (f.target) {
    model.targets.forEach((tg) => {
      const [x, y] = tg.projected;
      ctx.fillStyle = tg.visible === true ? COLOR.targetVisible : COLOR.target;
      ctx.beginPath();
      ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y);
      ctx.closePath(); ctx.fill();
      if (tg.visible === true) {
        ctx.strokeStyle = "rgba(55,192,122,0.9)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.stroke();
      }
    });
  }

  if (model.spawn) {
    const [x, y] = model.spawn.projected;
    ctx.strokeStyle = COLOR.spawn;
    ctx.lineWidth = 1.4;
    ctx.strokeRect(x - 4, y - 4, 8, 8);
  }

  if (model.terminal) {
    const [x, y] = model.terminal.projected;
    ctx.strokeStyle = COLOR.terminal;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
    ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
  }

  if (model.agent) {
    const [x, y] = model.agent.projected;
    const yaw = (model.agent.yaw_deg * Math.PI) / 180;
    const dir = [Math.sin(yaw), Math.cos(yaw)];
    const sx = dir[0];
    const sy = -dir[1];
    const ang = Math.atan2(sy, sx);
    if (f.fov) {
      const r = Math.min(W, H) * 0.16;
      ctx.fillStyle = "rgba(74,163,255,0.14)";
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.arc(x, y, r, ang - Math.PI / 6, ang + Math.PI / 6);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(74,163,255,0.35)";
      ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.fillStyle = COLOR.agent;
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#eaf4ff";
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + sx * 14, y + sy * 14); ctx.stroke();
  }

  return model;
}

function niceStep(span) {
  const raw = span / 8;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6))));
  const norm = raw / mag;
  const m = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
  return m * mag;
}
