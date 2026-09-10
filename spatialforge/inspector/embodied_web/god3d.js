// SpatialForge God View — real 3D semantic research inspector (Three.js/WebGL).
//
// Renders the ProcTHOR/AI2-THOR house geometry (rooms / walls / doors / windows
// / object proxies) plus researcher-only episode truth: agent pose, heading,
// camera frustum, growing trajectory, privileged target position, spawn and
// terminal markers. Missing metadata is rendered as "unavailable", never faked.

import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { WALL_THICKNESS, wallBoxes, openingBox } from "./wall_geometry.js";
import {
  worldToThree,
  threeToWorld,
  worldDirToThree,
  worldBoundsToThree,
  topViewCamera,
  wallRotationY,
} from "./world_space.js";

const FOV_VERTICAL_DEG = 60;
const FOV_ASPECT = 1.0;
const FOV_FAR = 3.0;

let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let canvasEl = null;
let root = null;          // scene root (cleared on scene reload)
let dyn = null;           // dynamic episode overlay group
let gridHelper = null;
let bounds = null;
let followAgent = false;
let followLast = null;
let lastEpisode = null;
let lastScenePayload = null;
let overlayFlags = { trajectory: true, fov: true, target: true };
let visible3d = true;
let userInteracted = false;
let lastBoundsKey = null;
let onUserInteract = null;

/** Register a callback fired when the user grabs the camera (enters Free). */
export function setUserInteractHandler(fn) {
  onUserInteract = fn;
}

/** Current camera orientation for the researcher HUD (reported in world space). */
export function cameraState() {
  if (!camera || !controls) return null;
  const dx = camera.position.x - controls.target.x;
  const dy = camera.position.y - controls.target.y;
  const dz = camera.position.z - controls.target.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const world = threeToWorld([dx, dy, dz]);
  const yaw = (Math.atan2(world[0], world[2]) * 180) / Math.PI;
  const pitch = (Math.asin(dist > 1e-6 ? dy / dist : 0) * 180) / Math.PI;
  const target = threeToWorld([controls.target.x, controls.target.y, controls.target.z]);
  return {
    mode: followAgent ? "follow" : "free",
    yaw: Math.round(yaw * 10) / 10,
    pitch: Math.round(pitch * 10) / 10,
    distance: Math.round(dist * 100) / 100,
    target,
  };
}

const COL = {
  floor: 0x2b3a4a,
  floorAlt: 0x33455a,
  wall: 0x8fa3b8,
  wallEmpty: 0xb9c7d6,
  door: 0xb98a4b,
  window: 0x6fc2e8,
  object: 0x5c7189,
  target: 0xe0574f,
  targetVisible: 0x37c07a,
  agent: 0x4aa3ff,
  spawn: 0xe8b23a,
  trajectory: 0x4aa3ff,
  terminal: 0xd05ad0,
  aabb: 0x7f95ab,
};

function available() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) {
    return false;
  }
}

export function isAvailable() {
  return available();
}

export function init(canvas) {
  if (!available()) return false;
  canvasEl = canvas;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x060a0f, 1);
  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x060a0f, 30, 80);

  camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
  camera.position.set(6, 6, 6);

  controls = new OrbitControls(camera, renderer.domElement);
  configureControls();
  // Grabbing the camera always wins: enter Free so the Follow updater can never
  // snap the user's view back on the next frame.
  controls.addEventListener("start", () => {
    userInteracted = true;
    followAgent = false;
    if (onUserInteract) onUserInteract(cameraState());
  });

  const hemi = new THREE.HemisphereLight(0xdfeaff, 0x1a2028, 1.1);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.4);
  dir.position.set(8, 14, 6);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xaac4ff, 0.5);
  dir2.position.set(-8, 10, -6);
  scene.add(dir2);

  root = new THREE.Group();
  scene.add(root);
  dyn = new THREE.Group();
  scene.add(dyn);
  return true;
}

function configureControls() {
  if (!controls || !canvasEl) return;
  controls.enabled = true;
  controls.enableRotate = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  canvasEl.style.pointerEvents = "auto";
  canvasEl.style.touchAction = "none";
}

export function setVisible(v) {
  visible3d = !!v;
  if (canvasEl) canvasEl.style.display = visible3d ? "" : "none";
  if (visible3d) configureControls();
}

function disposeGroup(g) {
  if (!g) return;
  while (g.children.length) {
    const c = g.children.pop();
    c.traverse?.((n) => {
      if (n.geometry) n.geometry.dispose();
      if (n.material) {
        const mats = Array.isArray(n.material) ? n.material : [n.material];
        mats.forEach((m) => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
    });
  }
}

function makeShape(points) {
  // `points` are Three-space; ShapeGeometry lies in XY and `rotation.x = +PI/2`
  // maps local (x, y) -> world (x, 0, y), so feeding Three (x, z) lands the
  // floor exactly at the canonical worldToThree position.
  const shape = new THREE.Shape();
  points.forEach((p, i) => {
    const x = p[0];
    const y = p[2];
    if (i === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  });
  shape.closePath();
  return shape;
}

function addFloors(sceneData) {
  (sceneData.rooms || []).forEach((room, i) => {
    const poly = (room.floor_polygon || []).map((p) => worldToThree(p));
    if (poly.length < 3) return;
    const geom = new THREE.ShapeGeometry(makeShape(poly));
    const mat = new THREE.MeshStandardMaterial({
      color: i % 2 ? COL.floorAlt : COL.floor,
      roughness: 0.95,
      metalness: 0.0,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.rotation.x = Math.PI / 2;
    mesh.position.y = (room.floor_y || 0) + 0.005;
    mesh.userData.layer = "floors";
    root.add(mesh);
  });
}

function addWalls(sceneData) {
  (sceneData.walls || []).forEach((w) => {
    if (!w.base || w.height <= 0 || w.length <= 0) return;
    const boxes = wallBoxes(w);
    if (!boxes.length) return;
    const mat = new THREE.MeshStandardMaterial({
      color: w.empty ? COL.wallEmpty : COL.wall,
      roughness: 0.9,
      transparent: true,
      opacity: w.empty ? 0.35 : 0.75,
    });
    boxes.forEach((box) => {
      const geom = new THREE.BoxGeometry(box.size[0], box.size[1], box.size[2]);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(box.center[0], box.center[1], box.center[2]);
      mesh.rotation.y = box.rotationY;
      mesh.userData.layer = "walls";
      root.add(mesh);
    });
  });
}

function addOpenings(sceneData) {
  const groups = [["doors", COL.door], ["windows", COL.window]];
  groups.forEach(([key, color]) => {
    (sceneData[key] || []).forEach((o) => {
      const box = openingBox(o);
      if (!box) return;
      const geom = new THREE.PlaneGeometry(box.size[0], box.size[1]);
      const mat = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: key === "windows" ? 0.45 : 0.7,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, mat);
      // box.center is the hole center; windows are raised above the floor.
      mesh.position.set(box.center[0], box.center[1], box.center[2]);
      mesh.rotation.y = box.rotationY;
      mesh.userData.layer = key;
      root.add(mesh);
    });
  });
}

function addObjects(sceneData, targetIds) {
  const tset = new Set((targetIds || []).map(String));
  (sceneData.objects || []).forEach((o) => {
    if (!o.position) return;
    const isTarget = tset.has(String(o.id));
    if (o.aabb && o.aabb.center && o.aabb.size) {
      const [cx, cy, cz] = worldToThree(o.aabb.center);
      const [sx, sy, sz] = o.aabb.size;
      const geom = new THREE.BoxGeometry(Math.max(sx, 0.03), Math.max(sy, 0.03), Math.max(sz, 0.03));
      const mat = new THREE.MeshStandardMaterial({
        color: isTarget ? COL.target : COL.object,
        transparent: true,
        opacity: isTarget ? 0.55 : 0.28,
        roughness: 0.8,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(cx, cy, cz);
      mesh.userData.layer = "objects";
      root.add(mesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color: COL.aabb, transparent: true, opacity: 0.5 })
      );
      edges.position.copy(mesh.position);
      edges.userData.layer = "objects";
      root.add(edges);
    } else {
      const geom = new THREE.BoxGeometry(0.12, 0.12, 0.12);
      const mat = new THREE.MeshStandardMaterial({
        color: isTarget ? COL.target : COL.object,
        transparent: true,
        opacity: 0.85,
      });
      const mesh = new THREE.Mesh(geom, mat);
      const [px, py, pz] = worldToThree(o.position);
      mesh.position.set(px, py + 0.06, pz);
      mesh.userData.layer = "objects";
      root.add(mesh);
    }
  });
}

export function setLayerVisibility(flags) {
  if (!root) return;
  const f = flags || {};
  root.traverse((n) => {
    const layer = n.userData && n.userData.layer;
    if (layer === "walls" && f.walls !== undefined) n.visible = !!f.walls;
    if (layer === "objects" && f.objects !== undefined) n.visible = !!f.objects;
  });
}

function makeLabelSprite(text, color) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const font = "600 28px system-ui,sans-serif";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.width = w;
  canvas.height = 44;
  const c2 = canvas.getContext("2d");
  c2.font = font;
  c2.fillStyle = "rgba(6,10,15,0.72)";
  c2.fillRect(0, 0, w, 44);
  c2.fillStyle = color;
  c2.fillText(text, 12, 31);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set((w / 44) * 0.32, 0.32, 1);
  return sprite;
}

function clearDynamic() {
  disposeGroup(dyn);
}

function updateAgent(agent) {
  if (!agent || !agent.position) return;
  const [x, y, z] = worldToThree(agent.position);
  const yaw = ((agent.rotation_yaw_deg || 0) * Math.PI) / 180;

  const group = new THREE.Group();
  group.position.set(x, y, z);

  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.11, 0.16, 0.62, 14),
    new THREE.MeshStandardMaterial({ color: COL.agent, emissive: 0x123a66, roughness: 0.5 })
  );
  body.position.y = 0.31;
  group.add(body);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 14, 12),
    new THREE.MeshStandardMaterial({ color: 0xdce9f7, roughness: 0.4 })
  );
  head.position.y = 0.72;
  group.add(head);

  // heading arrow (Thor forward = (sin yaw, 0, cos yaw), converted once)
  const forward = new THREE.Vector3(
    ...worldDirToThree([Math.sin(yaw), 0, Math.cos(yaw)])
  );
  const arrow = new THREE.ArrowHelper(forward, new THREE.Vector3(0, 0.86, 0), 0.55, 0x9fd0ff, 0.16, 0.1);
  group.add(arrow);
  dyn.add(group);

  if (overlayFlags.fov) {
    const fovRad = (FOV_VERTICAL_DEG * Math.PI) / 180;
    const far = FOV_FAR;
    const halfH = Math.tan(fovRad / 2) * far;
    const halfW = halfH * FOV_ASPECT;
    // Thor right = (cos yaw, 0, -sin yaw); convert both frame axes once.
    const right = new THREE.Vector3(
      ...worldDirToThree([Math.cos(yaw), 0, -Math.sin(yaw)])
    );
    const origin = new THREE.Vector3(0, 0.72, 0);
    const corner = (sx, sy) =>
      origin
        .clone()
        .addScaledVector(right, sx * halfW)
        .addScaledVector(forward, far)
        .add(new THREE.Vector3(0, sy, 0));
    const corners = [
      corner(-1, -halfH),
      corner(1, -halfH),
      corner(1, halfH),
      corner(-1, halfH),
    ];
    const verts = [];
    corners.forEach((c) => {
      verts.push(origin.x, origin.y, origin.z, c.x, c.y, c.z);
    });
    [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([a, b]) => {
      verts.push(corners[a].x, corners[a].y, corners[a].z, corners[b].x, corners[b].y, corners[b].z);
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    const lines = new THREE.LineSegments(
      geom,
      new THREE.LineBasicMaterial({ color: COL.agent, transparent: true, opacity: 0.5 })
    );
    lines.position.set(x, y, z);
    dyn.add(lines);

    // ground FOV wedge: sector centered on the canonical forward direction.
    // CircleGeometry angle phi maps to Three (cos phi, 0, -sin phi) after the
    // -PI/2 X rotation, so the center angle is the same wallRotationY used to
    // align +X-length geometry with a Thor yaw.
    const centerAngle = wallRotationY(agent.rotation_yaw_deg || 0);
    const wedge = new THREE.Mesh(
      new THREE.CircleGeometry(far * 0.6, 24, centerAngle - fovRad / 2, fovRad),
      new THREE.MeshBasicMaterial({ color: COL.agent, transparent: true, opacity: 0.12, side: THREE.DoubleSide })
    );
    wedge.rotation.x = -Math.PI / 2;
    wedge.position.set(x, y + 0.02, z);
    dyn.add(wedge);
  }
}

function updateTrajectory(trace, segments) {
  if (!overlayFlags.trajectory) return;
  const segs = (segments && segments.length ? segments : [trace]).filter(
    (s) => s && s.length >= 2
  );
  if (!segs.length) return;
  const allPts = [];
  segs.forEach((seg) => {
    const pts = seg.map((p) => {
      const [x, y, z] = worldToThree(p);
      return new THREE.Vector3(x, y + 0.04, z);
    });
    allPts.push(...pts);
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({ color: COL.trajectory, transparent: true, opacity: 0.85 })
    );
    dyn.add(line);
  });
  const dots = new THREE.BufferGeometry().setFromPoints(allPts);
  const cloud = new THREE.Points(
    dots,
    new THREE.PointsMaterial({ color: COL.trajectory, size: 0.07, transparent: true, opacity: 0.8 })
  );
  dyn.add(cloud);
}

function updateTargets(targets, visible) {
  if (!overlayFlags.target) return;
  (targets || []).forEach((t) => {
    if (!t.position) return;
    const [x, y, z] = worldToThree(t.position);
    const color = visible === true ? COL.targetVisible : COL.target;
    const diamond = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.16, 0),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4, transparent: true, opacity: 0.95 })
    );
    diamond.position.set(x, y + 0.25, z);
    dyn.add(diamond);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.26, 0.02, 8, 32),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(x, y + 0.03, z);
    dyn.add(ring);
    const label = makeLabelSprite(t.category || "target", "#ffb3ad");
    label.position.set(x, y + 0.62, z);
    dyn.add(label);
  });
}

function updateSpawn(spawn) {
  if (!spawn || !spawn.position) return;
  const [x, y, z] = worldToThree(spawn.position);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.3, 0.025, 8, 36),
    new THREE.MeshBasicMaterial({ color: COL.spawn, transparent: true, opacity: 0.9 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(x, y + 0.03, z);
  dyn.add(ring);
  const label = makeLabelSprite("spawn (setup)", "#ffd98a");
  label.position.set(x, y + 0.55, z);
  dyn.add(label);
}

export function setScene(payload) {
  lastScenePayload = payload;
  if (!renderer || !payload || !payload.scene) return false;
  disposeGroup(root);
  const rawBounds = payload.scene.bounds || null;
  bounds = rawBounds ? worldBoundsToThree(rawBounds) : null;
  addFloors(payload.scene);
  addWalls(payload.scene);
  addOpenings(payload.scene);
  const targetIds = ((payload.episode || {}).targets || []).map((t) => t.object_id);
  addObjects(payload.scene, targetIds);

  if (gridHelper) {
    root.remove(gridHelper);
    gridHelper.geometry.dispose();
    gridHelper.material.dispose();
    gridHelper = null;
  }
  if (bounds) {
    const size = Math.max(
      bounds.max[0] - bounds.min[0],
      bounds.max[2] - bounds.min[2]
    );
    gridHelper = new THREE.GridHelper(Math.max(size, 4), Math.max(8, Math.round(size * 2)), 0x1f2c3a, 0x16202c);
    gridHelper.position.set(
      (bounds.min[0] + bounds.max[0]) / 2, -0.01, (bounds.min[2] + bounds.max[2]) / 2
    );
    root.add(gridHelper);
  }
  updateEpisode(payload.episode);
  const boundsKey = rawBounds ? JSON.stringify(rawBounds) : "none";
  if (boundsKey !== lastBoundsKey) {
    lastBoundsKey = boundsKey;
    fitBounds();
  }
  return true;
}

export function updateEpisode(episode) {
  lastEpisode = episode || null;
  clearDynamic();
  if (!episode) return;
  updateTrajectory(episode.trace || [], episode.trace_segments);
  updateSpawn(episode.spawn);
  updateTargets(episode.targets || [], episode.visible);
  if (episode.agent) updateAgent(episode.agent);
  if (episode.terminal) {
    const ag = episode.agent;
    if (ag && ag.position) {
      const [x, y, z] = worldToThree(ag.position);
      const flag = new THREE.Mesh(
        new THREE.ConeGeometry(0.12, 0.3, 4),
        new THREE.MeshBasicMaterial({ color: COL.terminal })
      );
      flag.position.set(x, y + 1.15, z);
      dyn.add(flag);
    }
  }
}

export function setOverlayFlags(flags) {
  overlayFlags = Object.assign({}, overlayFlags, flags || {});
  if (lastScenePayload) {
    updateEpisode(lastScenePayload.episode);
  }
}

function fitBounds() {
  if (!bounds || !camera || !controls) return;
  const cx = (bounds.min[0] + bounds.max[0]) / 2;
  const cz = (bounds.min[2] + bounds.max[2]) / 2;
  const size = Math.max(
    bounds.max[0] - bounds.min[0],
    bounds.max[2] - bounds.min[2],
    4
  );
  controls.target.set(cx, 0.6, cz);
  camera.position.set(cx + size * 0.75, size * 0.8, cz + size * 0.75);
  camera.updateProjectionMatrix();
  controls.update();
}

export function view(name) {
  followAgent = name === "follow";
  if (!camera || !controls) return;
  if (name === "reset") {
    fitBounds();
    return;
  }
  if (name === "top") {
    if (!bounds) return;
    const cam = topViewCamera(bounds);
    controls.target.set(cam.target[0], cam.target[1], cam.target[2]);
    camera.position.set(cam.position[0], cam.position[1], cam.position[2]);
    controls.update();
    return;
  }
  if (name === "perspective") {
    fitBounds();
    return;
  }
  if (name === "follow" && lastEpisode && lastEpisode.agent && lastEpisode.agent.position) {
    const [x, y, z] = worldToThree(lastEpisode.agent.position);
    const yaw = ((lastEpisode.agent.rotation_yaw_deg || 0) * Math.PI) / 180;
    const forward = worldDirToThree([Math.sin(yaw), 0, Math.cos(yaw)]);
    controls.target.set(x, y + 0.7, z);
    camera.position.set(
      x - forward[0] * 1.6,
      y + 1.5,
      z - forward[2] * 1.6
    );
    followLast = [x, y, z];
    controls.update();
  }
}

export function resize() {
  if (!renderer || !canvasEl) return;
  const parent = canvasEl.parentElement;
  const W = Math.max(80, parent.clientWidth);
  const H = Math.max(80, parent.clientHeight);
  renderer.setSize(W, H, false);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
}

export function render() {
  if (!renderer || !scene || !camera) return;
  if (followAgent && lastEpisode && lastEpisode.agent && lastEpisode.agent.position) {
    const [x, y, z] = worldToThree(lastEpisode.agent.position);
    if (followLast) {
      const dx = x - followLast[0];
      const dy = y - followLast[1];
      const dz = z - followLast[2];
      if (dx || dy || dz) {
        camera.position.x += dx;
        camera.position.y += dy;
        camera.position.z += dz;
        controls.target.x += dx;
        controls.target.y += dy;
        controls.target.z += dz;
      }
    } else {
      controls.target.set(x, y + 0.7, z);
    }
    followLast = [x, y, z];
  } else {
    followLast = null;
  }
  controls.update();
  renderer.render(scene, camera);
}

export function hasScene() {
  return !!(lastScenePayload && lastScenePayload.scene);
}

export function dispose() {
  if (renderer) {
    renderer.dispose();
    renderer = null;
  }
  scene = null;
  camera = null;
  controls = null;
  canvasEl = null;
  root = null;
  dyn = null;
}
