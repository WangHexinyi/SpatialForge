"""Embodied Inspector runtime manager (backend selection + episode control).

Holds a live :class:`EmbodiedEnvironment` and lets the Inspector frontend drive
an object-search episode: start, step actions, read observation/state, fetch
the first-person RGB frame, inspect model input vs privileged truth, and query
the teacher path.

Backend selection: the deterministic grid backend is the CPU test/UAT harness
(always works headless). The ProcTHOR backend is used when the ProcTHOR / AI2-THOR
environment is importable and a house can be loaded; otherwise a clear error is
surfaced (never silently replaced with a mock for the smoke).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.environment import EmbodiedEnvironment
from spatialforge.embodied.episode_labels import (
    classify_episode,
    condensed_view,
    curated_priority,
)
from spatialforge.embodied.spawn_semantics import check_episode_semantics
from spatialforge.inspector.scene3d import build_scene3d, episode_scene_payload

try:  # real backend is only usable when ai2thor is importable
    import ai2thor  # noqa: F401
    from spatialforge.embodied.backends.procthor import ProcTHORBackend

    _HAS_PROCTHOR = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PROCTHOR = False


class EmbodiedSession:
    """One active episode session exposed over HTTP."""

    def __init__(self, backend_name: str = "deterministic", seed: int = 0):
        self.backend_name = backend_name
        self._seed = int(seed)
        self.backend = self._make_backend(backend_name, seed)
        self.env: Optional[EmbodiedEnvironment] = None
        self._house_loaded = False
        self._house: Optional[Dict[str, Any]] = None
        self._last_model_input: Optional[Dict[str, Any]] = None
        self._last_privileged: Optional[Dict[str, Any]] = None
        # ---- saved teacher-rollout episodes (research inspection) ----
        self.episodes_root: Optional[Path] = None
        self.episodes_roots: List[Path] = []
        self._loaded_episode: Optional[Dict[str, Any]] = None
        # ---- 3D God View scene cache / house resolution ----
        self._scene_cache: Dict[str, Dict[str, Any]] = {}
        self._episode_house_map: Optional[Dict[str, str]] = None
        # ---- real Unity God View renderer (lazy, one engine per house) ----
        self._god_cameras: Dict[str, Any] = {}
        self._god_camera_lock = threading.RLock()
        # ---- display-only telemetry ingested from runners ----
        self.last_telemetry: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Saved teacher-rollout episode inspection (researcher-only)
    # ------------------------------------------------------------------
    def set_episodes_root(self, path) -> None:
        self.set_episodes_roots([path])

    def set_episodes_roots(self, paths) -> None:
        roots = []
        for p in paths or []:
            if p is None:
                continue
            root = Path(p)
            root.mkdir(parents=True, exist_ok=True)
            if root not in roots:
                roots.append(root)
        self.episodes_roots = roots
        self.episodes_root = roots[0] if roots else None
        self._episode_house_map = None

    def _episode_dirs(self) -> List[Path]:
        dirs: List[Path] = []
        for root in self.episodes_roots or ([self.episodes_root] if self.episodes_root else []):
            if root and root.is_dir():
                dirs.extend(d for d in sorted(root.iterdir()) if d.is_dir())
        return dirs

    def _find_episode_dir(self, episode_id: str) -> Optional[Path]:
        if not episode_id:
            return None
        for root in self.episodes_roots or ([self.episodes_root] if self.episodes_root else []):
            if not root:
                continue
            d = (root / episode_id).resolve()
            if str(d).startswith(str(root.resolve())) and d.is_dir():
                return d
        return None

    def _decisions_for(self, d: Path, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if rec.get("decisions"):
            return list(rec["decisions"])
        dec_f = d / "decisions.json"
        if dec_f.is_file():
            try:
                return json.loads(dec_f.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def list_episodes(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in self._episode_dirs():
            priv = d / "privileged_research_record.json"
            if not priv.is_file():
                continue
            try:
                rec = json.loads(priv.read_text(encoding="utf-8"))
            except Exception:
                continue
            task = rec.get("task", {})
            ep = rec.get("episode", {}) or {}
            decisions = self._decisions_for(d, rec)
            quality = rec.get("render_quality")
            cls = classify_episode(rec, decisions=decisions, render_quality=quality)
            out.append({
                "episode_id": d.name,
                "house_id": task.get("house_id") or rec.get("house_id"),
                "category": task.get("target_category") or rec.get("category"),
                "status": ep.get("status") or rec.get("outcome", {}).get("status"),
                "success": ep.get("success") or rec.get("outcome", {}).get("success"),
                "steps": len(rec.get("steps", [])),
                "teacher_plan": rec.get("teacher_plan"),
                "has_student_record": (d / "student_training_record.json").is_file(),
                "frames": rec.get("frames", []),
                "model_controlled": bool(
                    rec.get("model_controlled") or decisions
                    or rec.get("teacher_plan", {}).get("model_controlled")
                ),
                "model_terminal_reason": rec.get("terminal_reason"),
                "decision_count": len(decisions),
                "invalid_decision_count": sum(
                    1 for x in decisions if x.get("invalid")),
                "render_quality": quality,
                "labels": cls["labels"],
                "flags": cls["flags"],
                "priority": curated_priority(cls),
                "no_progress_runs": cls["metrics"]["no_progress_runs"],
                "has_setup": cls["metrics"]["has_setup"],
            })
        # in-progress model episodes (live_state.json without final records)
        for d in self._episode_dirs():
            if (d / "privileged_research_record.json").is_file():
                continue
            live = d / "live_state.json"
            if not live.is_file():
                continue
            try:
                st = json.loads(live.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "episode_id": d.name,
                "category": st.get("category"),
                "status": f"live:{st.get('status', 'running')}",
                "success": None,
                "steps": int(st.get("step", 0) or 0),
                "teacher_plan": None,
                "has_student_record": False,
                "frames": [],
                "model_controlled": True,
                "live": True,
                "decision_count": int(st.get("decision_idx", 0) or 0),
                "latest_model_action": (st.get("decision") or {}).get("model_action"),
                "labels": ["LIVE"],
                "flags": {"valid": False, "legacy": False},
                "priority": 0,
                "render_quality": None,
                "no_progress_runs": [],
            })
        # curated ordering: researcher-useful post-fix episodes first
        out.sort(key=lambda e: (-int(e.get("priority") or 0), str(e.get("episode_id"))))
        return out

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """Load a saved episode + build a pageable timeline (privileged)."""
        d = self._find_episode_dir(episode_id)
        if d is None:
            raise RuntimeError(f"episode {episode_id} not found")
        priv_f = d / "privileged_research_record.json"
        live_f = d / "live_state.json"
        if not priv_f.is_file():
            if not live_f.is_file():
                raise RuntimeError(f"episode {episode_id} not found")
            # in-progress model episode (God View LIVE polling)
            st = json.loads(live_f.read_text(encoding="utf-8"))
            decision = st.get("decision", {}) or {}
            self._loaded_episode = {
                "episode_id": episode_id,
                "header": {
                    "episode_id": episode_id,
                    "category": st.get("category"),
                    "status": st.get("status"),
                    "live": True,
                    "goal": st.get("goal"),
                    "step": st.get("step"),
                    "elapsed_s": st.get("elapsed_s"),
                    "decision_idx": st.get("decision_idx"),
                    "labels": ["LIVE"],
                    "flags": {"valid": False, "legacy": False},
                    "_privileged": True,
                },
                "timeline": [],
                "_dir": str(d),
            }
            return {
                "header": self._loaded_episode["header"],
                "timeline": [],
                "live_decision": decision,
                "live_frame": st.get("frame"),
                "in_progress": True,
            }
        rec = json.loads(priv_f.read_text(encoding="utf-8"))
        student = {}
        stu_f = d / "student_training_record.json"
        if stu_f.is_file():
            student = json.loads(stu_f.read_text(encoding="utf-8"))

        task = rec.get("task", {})
        ep = rec.get("episode", {}) or {}
        decisions = self._decisions_for(d, rec)
        classification = classify_episode(
            rec, decisions=decisions, render_quality=rec.get("render_quality")
        )
        header = {
            "episode_id": episode_id,
            "house_id": task.get("house_id") or rec.get("house_id"),
            "category": task.get("target_category") or rec.get("category"),
            "target_object_ids": task.get("target_object_ids") or rec.get("target_object_id"),
            "target_positions": task.get("target_positions"),
            "teacher_plan": rec.get("teacher_plan"),
            "spawn": rec.get("spawn"),
            "setup": rec.get("setup") or {},
            "status": ep.get("status") or rec.get("outcome", {}).get("status"),
            "success": ep.get("success"),
            "success_reason": ep.get("success_reason") or rec.get("outcome", {}).get("reason"),
            "student_domain": student.get("domain") if student else None,
            "student_leak_checked": student.get("_leak_checked") if student else None,
            "model_controlled": bool(
                rec.get("model_controlled") or decisions
                or rec.get("teacher_plan", {}).get("model_controlled")
            ),
            "model_terminal_reason": rec.get("terminal_reason"),
            "decision_count": len(decisions),
            "invalid_decision_count": sum(1 for x in decisions if x.get("invalid")),
            "render_quality": rec.get("render_quality"),
            "house_path": rec.get("house_path"),
            "labels": classification["labels"],
            "flags": classification["flags"],
            "render_quality_bad": "INVALID_SENSOR" in classification["labels"],
            "_privileged": True,
        }

        spawn_rec = rec.get("spawn") or {}
        steps_rec = rec.get("steps", [])
        # ------------------------------------------------------------------
        # Canonical initial pose (frame_0000 truth).
        #
        # frame_0000 is the observation captured by the environment *before* the
        # first action. Its pose must therefore be the authoritative pre-action-0
        # state, never a post-action state. Deriving it from ``steps[0].agent``
        # (which is the state AFTER action 0) was an off-by-one: for a first
        # action that rotates, the Inspector/God View showed the post-rotation
        # heading beside a pre-rotation RGB frame.
        #
        # Preference order (all are persisted truth, none fabricated):
        #   1. ``setup.initial_pose``  -- post-fix authoritative spawn pose.
        #   2. ``decisions[0].agent``  -- model records: the per-decision snapshot
        #      is taken *before* env.step, so it is the pre-action-0 pose.
        #   3. ``spawn``               -- teacher legacy records: start_episode
        #      teleported to the sampled spawn, so it is the initial pose.
        #   4. ``steps[0].agent``      -- last resort, explicitly marked; a
        #      post-action pose is never silently relabelled as the start.
        # ------------------------------------------------------------------
        start_agent = None
        initial_source = None
        setup_initial = (rec.get("setup") or {}).get("initial_pose") or {}
        if setup_initial.get("position"):
            start_agent = {
                "position": setup_initial["position"],
                "rotation_yaw_deg": setup_initial.get("rotation_yaw_deg"),
                "camera_horizon_deg": setup_initial.get("horizon_deg"),
            }
            initial_source = "setup.initial_pose"
        elif decisions and (decisions[0].get("agent") or {}).get("position"):
            start_agent = dict(decisions[0]["agent"])
            initial_source = "decisions[0].agent"
        elif spawn_rec.get("position"):
            start_agent = {
                "position": spawn_rec["position"],
                "rotation_yaw_deg": spawn_rec.get("yaw"),
                "camera_horizon_deg": spawn_rec.get("horizon"),
            }
            initial_source = "spawn"
        elif steps_rec and (steps_rec[0].get("agent") or {}).get("position"):
            start_agent = dict(steps_rec[0]["agent"])
            initial_source = "steps[0].agent (post-action fallback)"
        timeline = [{
            "idx": 0, "type": "context", "step": 0, "frame": "frame_0000.png",
            "action_origin": "setup",
            "authoritative_target_visible": bool(spawn_rec.get("initially_visible")),
            "agent": start_agent,
            "agent_source": initial_source,
            "note": "episode start (environment setup / spawn; not a model action)",
        }]
        # Canonical entry semantic (one authoritative AI2-THOR state per entry):
        #   entry 0        -> frame_0000 + initial pose (pre-action-0 state)
        #   entry N (>=1)  -> frame_N + steps[N-1] pose/action, i.e. the state
        #                     AFTER executed action N-1. frame_N and
        #                     steps[N-1].agent are serialized from the SAME
        #                     env.step event (``trans.observation`` and
        #                     ``trans.agent_state``), so the FPV beside the God
        #                     View can never represent a different temporal state.
        # ``step`` is the environment step counter after that action (== N).
        for i, st in enumerate(steps_rec, start=1):
            timeline.append({
                "idx": i,
                "type": "action",
                "step": st.get("step"),
                "action": st.get("action_type"),
                "action_origin": st.get("action_origin") or "model",
                "action_success": st.get("action_success"),
                "agent": st.get("agent"),
                "authoritative_target_visible": st.get("authoritative_target_visible"),
                "terminal": st.get("terminal"),
                "frame": f"frame_{i:04d}.png",
            })
        condensed = condensed_view(timeline, classification["metrics"]["no_progress_runs"])

        self._loaded_episode = {
            "episode_id": episode_id,
            "header": header,
            "timeline": timeline,
            "decisions": decisions,
            "classification": classification,
            "condensed": condensed,
            "_dir": str(d),
        }
        return {
            "header": header,
            "timeline": timeline,
            "student_record": student,
            "decisions": decisions,
            "classification": classification,
            "condensed": condensed,
        }

    def episode_state(self, idx: int) -> Dict[str, Any]:
        ep = self._loaded_episode
        if ep is None:
            raise RuntimeError("no episode loaded; POST /api/episodes/load first")
        t = ep["timeline"]
        idx = int(idx)
        if idx < 0 or idx >= len(t):
            raise RuntimeError(f"step index {idx} out of range")
        entry = dict(t[idx])
        frame_name = entry.pop("frame")
        frame_rel = os.path.join(ep["episode_id"], "frames", frame_name)
        return {
            "episode_id": ep["episode_id"],
            "idx": idx,
            **entry,
            "frame": frame_rel,
            "header": ep["header"],
            "_privileged_researcher_only": True,
        }

    def episode_frame_by_name(self, frame_name: str) -> Optional[bytes]:
        ep = self._loaded_episode
        if ep is None:
            return None
        frame = (Path(ep["_dir"]) / "frames" / frame_name).resolve()
        if not str(frame).startswith(str((Path(ep["_dir"]) / "frames").resolve())):
            return None
        if not frame.is_file():
            return None
        return frame.read_bytes()

    def episode_frame_png(self, idx: int) -> Optional[bytes]:
        ep = self._loaded_episode
        if ep is None:
            return None
        t = ep["timeline"]
        idx = int(idx)
        if idx < 0 or idx >= len(t):
            return None
        frame = Path(ep["_dir"]) / "frames" / t[idx]["frame"]
        if not frame.is_file():
            return None
        return frame.read_bytes()


    # ------------------------------------------------------------------
    # 3D God View scene resolution (researcher-only)
    # ------------------------------------------------------------------
    def _load_privileged(self, episode_id: str) -> Optional[Dict[str, Any]]:
        d = self._find_episode_dir(episode_id)
        if d is None:
            return None
        p = d / "privileged_research_record.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def episode_semantics(self, episode_id: str) -> Dict[str, Any]:
        """Spawn/reset invariant check for one saved episode (researcher-only)."""
        rec = self._load_privileged(episode_id)
        if rec is None:
            raise RuntimeError(f"episode {episode_id} has no privileged record")
        result = check_episode_semantics(rec)
        result["episode_id"] = episode_id
        result["_privileged_researcher_only"] = True
        return result

    def _build_episode_house_map(self) -> Dict[str, str]:
        if self._episode_house_map is not None:
            return self._episode_house_map
        mapping: Dict[str, str] = {}
        for root in (self.episodes_roots or ([self.episodes_root] if self.episodes_root else [])):
            if not root or not root.is_dir():
                continue
            for results_file in root.glob("**/results/*.jsonl"):
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            ep, house = row.get("episode_id"), row.get("house")
                            if ep and house:
                                mapping[str(ep)] = str(house)
                except Exception:
                    continue
        self._episode_house_map = mapping
        return mapping

    def _house_search_dirs(self) -> List[Path]:
        dirs = []
        env = os.environ.get("SF_HOUSES_DIRS")
        if env:
            dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
        for root in (self.episodes_roots or ([self.episodes_root] if self.episodes_root else [])):
            if root:
                dirs.append(root / "houses")
        repo = Path(__file__).resolve().parents[2]
        dirs.extend([
            repo / "outputs" / "embodied_bc" / "houses",
            Path("/root/autodl-tmp"),
        ])
        return dirs

    def _scene_snapshot_paths(self, stem: str) -> List[Path]:
        out = []
        for root in (self.episodes_roots or ([self.episodes_root] if self.episodes_root else [])):
            if root:
                out.append(root / "_scenes" / f"{stem}.json")
        return out

    def _upgrade_scene_openings(
        self, scene: Dict[str, Any], house_path: Optional[str]
    ) -> Dict[str, Any]:
        """Backfill derived wall-opening truth onto a persisted scene snapshot.

        Older ``_scenes/<house>.json`` snapshots predate the wall-segment
        derivation and carry wall-local opening positions with no ``segments``.
        Rebuilding the canonical scene from the house JSON and merging only the
        derived fields keeps the snapshot's live AI2-THOR metadata while making
        the openings truthful in both the 2D and 3D views.
        """
        walls = scene.get("walls") or []
        openings = (scene.get("doors") or []) + (scene.get("windows") or [])
        if not walls:
            return scene
        if all("segments" in w for w in walls) and all(
            "wall_id" in o for o in openings
        ):
            return scene
        if not house_path or not Path(house_path).is_file():
            return scene
        try:
            house = json.loads(Path(house_path).read_text(encoding="utf-8"))
            fresh = build_scene3d(house)
        except Exception:  # noqa: BLE001
            return scene
        segments = {str(w.get("id")): w.get("segments") for w in fresh.get("walls", [])}
        for wall in walls:
            if "segments" not in wall:
                wall["segments"] = segments.get(str(wall.get("id")))
        for key in ("doors", "windows"):
            fresh_by_id = {str(o.get("id")): o for o in fresh.get(key, [])}
            for opening in scene.get(key) or []:
                source = fresh_by_id.get(str(opening.get("id")))
                if source is None or "wall_id" in opening:
                    continue
                for field in ("position", "wall_id", "wall_ids", "local_x"):
                    opening[field] = source.get(field)
        return scene

    def _scene_for_house_path(self, house_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not house_path:
            return None
        if house_path in self._scene_cache:
            return self._scene_cache[house_path]
        scene = None
        stem = Path(house_path).stem
        # 1) house-level scene snapshot persisted at rollout time
        for snap in self._scene_snapshot_paths(stem):
            if snap.is_file():
                try:
                    payload = json.loads(snap.read_text(encoding="utf-8"))
                    scene = payload.get("scene")
                    if scene:
                        break
                except Exception:
                    scene = None
        # 2) build from the official ProcTHOR house JSON
        if scene is None and Path(house_path).is_file():
            try:
                house = json.loads(Path(house_path).read_text(encoding="utf-8"))
                scene = build_scene3d(house)
            except Exception:
                scene = None
        if scene is not None:
            scene = self._upgrade_scene_openings(scene, house_path)
            self._scene_cache[house_path] = scene
        return scene

    def _resolve_episode_house_path(
        self, episode_id: str, header: Dict[str, Any]
    ) -> Optional[str]:
        recorded = header.get("house_path")
        if recorded and Path(recorded).is_file():
            return str(recorded)
        mapped = self._build_episode_house_map().get(episode_id)
        if mapped and Path(mapped).is_file():
            return mapped
        house_id = str(header.get("house_id") or "")
        if house_id:
            for base in self._house_search_dirs():
                for candidate in (base / f"{house_id}.json", base / house_id):
                    if candidate.is_file():
                        return str(candidate)
        # fallback: a persisted house-level scene snapshot carries the real path
        snaps = []
        for root in (self.episodes_roots or ([self.episodes_root] if self.episodes_root else [])):
            if root and (root / "_scenes").is_dir():
                snaps.extend(sorted((root / "_scenes").glob("*.json")))
        paths = []
        for snap in snaps:
            try:
                payload = json.loads(snap.read_text(encoding="utf-8"))
            except Exception:
                continue
            hp = payload.get("house_path")
            if hp and Path(hp).is_file():
                paths.append(str(hp))
        unique = list(dict.fromkeys(paths))
        if len(unique) == 1:
            return unique[0]
        return None

    def episode_scene3d(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Full 3D research scene + episode truth for the God View."""
        ep = self._loaded_episode
        if episode_id is not None and (ep is None or ep.get("episode_id") != episode_id):
            self.load_episode(episode_id)
            ep = self._loaded_episode
        if ep is None:
            raise RuntimeError("no episode loaded; POST /api/episodes/load first")
        header = ep.get("header", {})
        rec = self._load_privileged(ep["episode_id"]) or {}
        house_path = self._resolve_episode_house_path(ep["episode_id"], header)
        scene = self._scene_for_house_path(house_path)
        if scene is None:
            return {
                "schema": "spatialforge_scene3d.v1",
                "scene": None,
                "episode": None,
                "unavailable": [
                    "house geometry not found for this episode",
                    f"house_id={header.get('house_id')}",
                    "set SF_HOUSES_DIRS or persist _scenes/<house>.json",
                ],
                "_privileged_researcher_only": True,
            }
        spawn = rec.get("spawn") or header.get("spawn") or {}
        timeline = ep.get("timeline", [])
        terminal = bool(rec.get("episode", {}).get("success")) or (
            str(header.get("status")) not in ("running", "None", "")
        )
        payload = episode_scene_payload(
            scene,
            spawn=spawn,
            timeline=timeline,
            target_positions=header.get("target_positions") or [],
            target_object_ids=header.get("target_object_ids") or [],
            teacher_plan=header.get("teacher_plan"),
            terminal=terminal,
            frames=rec.get("frames") or [],
        )
        scene_snaps = self._scene_snapshot_paths(Path(house_path).stem) if house_path else []
        payload["scene_meta"] = {
            "source": scene.get("source"),
            "house_path": house_path,
            "scene_file": str(scene_snaps[0]) if scene_snaps else None,
            "unavailable": scene.get("unavailable", []),
        }
        payload["setup"] = rec.get("setup") or {}
        payload["spawn_semantics"] = check_episode_semantics(rec) if rec else None
        payload["classification"] = ep.get("classification") or (
            classify_episode(rec, decisions=ep.get("decisions"),
                             render_quality=rec.get("render_quality")) if rec else None
        )
        payload["condensed"] = ep.get("condensed") or {"indices": [], "segments": []}
        payload["_privileged_researcher_only"] = True
        return payload

    def live_scene3d(self) -> Dict[str, Any]:
        """3D scene for the live loaded house (when the backend exposes geometry)."""
        if self.backend is not None and hasattr(self.backend, "scene_geometry"):
            try:
                scene = self.backend.scene_geometry()
                trace = []
                agent = None
                if self.env is not None and self.env.agent_state is not None:
                    st = self.env.agent_state
                    agent = {
                        "position": list(st.position),
                        "rotation_yaw_deg": st.rotation_yaw_deg,
                        "camera_horizon_deg": st.camera_horizon_deg,
                    }
                    trace = [list(st.position)]
                targets = []
                if self.env is not None and self.env.task is not None:
                    targets = [
                        {"position": list(p), "object_id": None}
                        for p in self.env.task.target_positions
                    ]
                return {
                    "schema": "spatialforge_scene3d.v1",
                    "scene": scene,
                    "episode": {
                        "trace": trace,
                        "agent": agent,
                        "targets": targets,
                        "spawn": None,
                        "teacher_plan": None,
                        "terminal": False,
                    },
                    "_privileged_researcher_only": True,
                }
            except Exception as e:  # noqa: BLE001
                return {"scene": None, "error": str(e)}
        return {
            "schema": "spatialforge_scene3d.v1",
            "scene": None,
            "episode": None,
            "unavailable": [
                "live backend exposes no 3D geometry",
                f"backend={self.backend_name}",
            ],
            "_privileged_researcher_only": True,
        }

    # ------------------------------------------------------------------
    # Real Unity God View (primary researcher view; never model input)
    # ------------------------------------------------------------------
    def unity_godview_status(self) -> Dict[str, Any]:
        try:
            from spatialforge.inspector.god_camera import _HAS_PROCTHOR
        except Exception:  # noqa: BLE001
            _HAS_PROCTHOR = False
        display = os.environ.get("SF_GOD_CAMERA_DISPLAY") or os.environ.get("DISPLAY")
        renderer = None
        hardware = None
        if _HAS_PROCTHOR:
            try:
                from spatialforge.embodied.rendering import current_renderer, is_nvidia

                renderer = current_renderer(display)
                hardware = is_nvidia(renderer)
            except Exception:  # noqa: BLE001
                renderer = None
        return {
            "available": bool(_HAS_PROCTHOR),
            "backend": self.backend_name,
            "reason": None if _HAS_PROCTHOR else "ai2thor is not importable in this interpreter",
            "display": display,
            "renderer": renderer,
            "hardware": hardware,
            "_privileged_researcher_only": True,
        }

    def _god_camera_for(self, house_path: str):
        with self._god_camera_lock:
            cam = self._god_cameras.get(house_path)
            if cam is None:
                from spatialforge.inspector.god_camera import GodCameraRenderer

                cam = GodCameraRenderer(
                    house_path,
                    width=int(os.environ.get("SF_GOD_CAMERA_WIDTH", "640")),
                    height=int(os.environ.get("SF_GOD_CAMERA_HEIGHT", "480")),
                    x_display=os.environ.get("SF_GOD_CAMERA_DISPLAY")
                    or os.environ.get("DISPLAY"),
                    quality=os.environ.get("SF_GOD_CAMERA_QUALITY") or "Medium",
                )
                self._god_cameras[house_path] = cam
            return cam

    def _live_god_camera(self):
        """A God camera bound to the *running* ProcTHOR controller (no restart)."""
        with self._god_camera_lock:
            cam = getattr(self, "_live_camera", None)
            backend = self.backend
            if cam is not None and getattr(cam, "_provided_backend", None) is backend:
                return cam
            from spatialforge.inspector.god_camera import GodCameraRenderer

            getter = getattr(backend, "house_path", None)
            house_path = getter() if callable(getter) else None
            if not house_path:
                raise RuntimeError(
                    "live Unity God View needs a ProcTHOR backend started from a house JSON"
                )
            cam = GodCameraRenderer(
                str(house_path),
                width=int(os.environ.get("SF_GOD_CAMERA_WIDTH", "640")),
                height=int(os.environ.get("SF_GOD_CAMERA_HEIGHT", "480")),
                x_display=os.environ.get("SF_GOD_CAMERA_DISPLAY")
                or os.environ.get("DISPLAY"),
                quality=os.environ.get("SF_GOD_CAMERA_QUALITY") or "Medium",
                backend=backend,
            )
            self._live_camera = cam
            return cam

    def unity_godview_png(
        self,
        episode_id: Optional[str] = None,
        step: int = 0,
        view: str = "follow",
        camera: Optional[Dict[str, Any]] = None,
        quality: str = "full",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[bytes]:
        """Render the real Unity scene for one episode step (researcher-only).

        ``quality="preview"`` renders a real low-latency frame at
        ``GOD_VIEW_PREVIEW_SIZE`` during camera drags; ``quality="full"`` renders
        the settle frame at the browser's viewport size (clamped). The frame is
        JPEG because PNG encode dominates the frame budget.
        """
        from spatialforge.inspector.god_camera import (
            GOD_VIEW_FULL_JPEG_QUALITY,
            GOD_VIEW_FULL_SIZE,
            GOD_VIEW_MAX_SIZE,
            GOD_VIEW_PREVIEW_JPEG_QUALITY,
            GOD_VIEW_PREVIEW_SIZE,
        )

        ep = self._loaded_episode
        if episode_id is not None and (ep is None or ep.get("episode_id") != episode_id):
            self.load_episode(episode_id)
            ep = self._loaded_episode
        if ep is None:
            raise RuntimeError("no episode loaded; POST /api/episodes/load first")
        header = ep.get("header", {})
        house_path = self._resolve_episode_house_path(ep["episode_id"], header)
        if not house_path:
            raise RuntimeError("house geometry not found for Unity God View")
        timeline = ep.get("timeline", [])
        pose = None
        try:
            step_i = int(step)
        except (TypeError, ValueError):
            step_i = 0
        if 0 <= step_i < len(timeline):
            pose = timeline[step_i].get("agent")
        if pose is None and timeline:
            pose = timeline[0].get("agent")

        if str(quality).lower() == "preview":
            resolution = GOD_VIEW_PREVIEW_SIZE
            jpeg_quality = GOD_VIEW_PREVIEW_JPEG_QUALITY
        else:
            resolution = GOD_VIEW_FULL_SIZE
            if width and height:
                try:
                    resolution = (
                        max(320, min(int(width), GOD_VIEW_MAX_SIZE[0])),
                        max(240, min(int(height), GOD_VIEW_MAX_SIZE[1])),
                    )
                except (TypeError, ValueError):
                    resolution = GOD_VIEW_FULL_SIZE
            jpeg_quality = GOD_VIEW_FULL_JPEG_QUALITY

        cam = self._god_camera_for(house_path)
        return cam.render(
            agent_pose=pose,
            view=view,
            camera=camera,
            resolution=resolution,
            image_format="jpeg",
            jpeg_quality=jpeg_quality,
        )

    def live_unity_png(
        self, view: str = "follow", camera: Optional[Dict[str, Any]] = None
    ) -> Optional[bytes]:
        """Render the real Unity scene for the *live* manual/ProcTHOR episode.

        Uses the already-running controller (never restarts the engine) and the
        current ``env.agent_state`` pose. No privileged teleport is used to fake
        an action: the pose is whatever the last real environment transition
        produced.

        The live God camera shares the running engine, whose resolution is the
        model observation resolution; it is deliberately never resized here so
        FPV/training observations stay byte-identical. JPEG keeps transfer and
        encode off the critical path.
        """
        from spatialforge.inspector.god_camera import GOD_VIEW_LIVE_JPEG_QUALITY

        if self.env is None or self.env.agent_state is None:
            raise RuntimeError("no active episode; start a live episode first")
        st = self.env.agent_state
        pose = {
            "position": list(st.position),
            "rotation_yaw_deg": st.rotation_yaw_deg,
            "camera_horizon_deg": st.camera_horizon_deg,
        }
        # Reuse the running controller and never teleport: the pose is already
        # whatever the last real environment transition produced.
        with self._god_camera_lock:
            cam = self._live_god_camera()
            return cam.render(
                agent_pose=pose,
                view=view,
                camera=camera,
                teleport=False,
                image_format="jpeg",
                jpeg_quality=GOD_VIEW_LIVE_JPEG_QUALITY,
            )

    def prewarm_unity(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Start the Unity God engine + third-party camera in the background.

        The first real render otherwise pays the ~10s engine/house startup; this
        moves that cost off the interactive path (the controller is then reused).
        """
        ep = self._loaded_episode
        if episode_id is not None and (ep is None or ep.get("episode_id") != episode_id):
            self.load_episode(episode_id)
            ep = self._loaded_episode
        if ep is None:
            raise RuntimeError("no episode loaded; POST /api/episodes/load first")
        house_path = self._resolve_episode_house_path(ep["episode_id"], ep.get("header", {}))
        if not house_path:
            raise RuntimeError("house geometry not found for Unity God View")
        cam = self._god_camera_for(house_path)

        def _warm() -> None:
            try:
                from spatialforge.inspector.god_camera import GOD_VIEW_PREVIEW_SIZE

                cam.render(
                    agent_pose=None,
                    view="overview",
                    resolution=GOD_VIEW_PREVIEW_SIZE,
                    image_format="jpeg",
                    jpeg_quality=80,
                )
            except Exception:  # noqa: BLE001 - prewarm is best-effort
                pass

        threading.Thread(target=_warm, daemon=True).start()
        return {"prewarming": True, "house_path": house_path}

    def list_houses(self) -> List[Dict[str, Any]]:
        """Curated ProcTHOR house JSONs available for live/manual episodes."""
        seen = {}
        for base in self._house_search_dirs():
            try:
                if not base.is_dir():
                    continue
                for f in sorted(base.glob("*.json")):
                    seen.setdefault(f.stem, str(f))
            except Exception:  # noqa: BLE001
                continue
        return [{"house_id": k, "path": v} for k, v in sorted(seen.items())]

    def close_god_cameras(self) -> None:
        with self._god_camera_lock:
            for cam in self._god_cameras.values():
                try:
                    cam.close()
                except Exception:  # noqa: BLE001
                    pass
            self._god_cameras.clear()

    # ------------------------------------------------------------------
    # Runtime telemetry for the God View overlay (display-only)
    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema": "spatialforge_telemetry.v1", "ts": time.time()}
        try:
            from spatialforge.experiment.performance import NVMLClient

            client = NVMLClient(0)
            stats = client.query() if client.available else {}
            try:
                client.close()
            except Exception:
                pass
            out["gpu"] = {
                "utilization_pct": stats.get("gpu_utilization_pct"),
                "memory_used_mb": stats.get("gpu_memory_used_mb"),
                "memory_total_mb": stats.get("gpu_memory_total_mb"),
                "power_w": stats.get("gpu_power_w"),
                "sm_clock_mhz": stats.get("sm_clock_mhz"),
            }
        except Exception as e:  # noqa: BLE001
            out["gpu"] = {"error": str(e)}
        tele = self.last_telemetry
        if tele is None:
            tele = self._read_telemetry_file()
        if tele is not None:
            out["runtime"] = tele.get("runtime")
            out["model"] = tele.get("model")
            out["ingested_ts"] = tele.get("ts")
        else:
            out["runtime"] = None
            out["model"] = None
        return out

    def _read_telemetry_file(self) -> Optional[Dict[str, Any]]:
        """Fallback display telemetry from a runner-written JSON file."""
        candidates = []
        env_file = os.environ.get("SF_TELEMETRY_FILE")
        if env_file:
            candidates.append(Path(env_file))
        if self.episodes_root is not None:
            candidates.append(self.episodes_root / "live_telemetry.json")
            candidates.append(self.episodes_root.parent / "model_rollout" / "live_telemetry.json")
        for p in candidates:
            try:
                if p.is_file():
                    return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def ingest_telemetry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a display-only telemetry/model snapshot from a runner."""
        payload = dict(payload or {})
        payload.setdefault("ts", time.time())
        self.last_telemetry = payload
        return {"ok": True, "ts": payload["ts"]}

    @staticmethod
    def _make_backend(backend_name: str, seed: int):
        if backend_name == "deterministic":
            return DeterministicGridBackend(seed=seed)
        if backend_name == "procthor":
            if not _HAS_PROCTHOR:
                raise RuntimeError(
                    "ProcTHOR backend unavailable: ai2thor/procthor is not importable "
                    "in this interpreter. Install into the Python-3.10 venv and point "
                    "at a real AI2-THOR Unity build before selecting this backend."
                )
            return ProcTHORBackend()
        raise ValueError(f"unknown backend: {backend_name}")

    def available_backends(self) -> Dict[str, Any]:
        return {
            "deterministic": "deterministic-grid (CPU test/UAT harness)",
            "procthor": "ProcTHOR / AI2-THOR (importable)" if _HAS_PROCTHOR
            else "ProcTHOR / AI2-THOR (unavailable in this interpreter)",
        }

    def resolve_house_path(self, house_id: str) -> Optional[str]:
        """Resolve a house id (stem or path) to a real ProcTHOR house JSON."""
        if not house_id:
            return None
        p = Path(house_id)
        if p.is_file():
            return str(p)
        for base in self._house_search_dirs():
            try:
                for cand in (base / f"{house_id}.json", base / house_id):
                    if cand.is_file():
                        return str(cand)
            except Exception:  # noqa: BLE001
                continue
        return None

    def load_house(self, house_id: str = "", **opts) -> Dict[str, Any]:
        if not self._house_loaded:
            if self.backend_name == "procthor" and "house" not in opts:
                resolved = self.resolve_house_path(house_id)
                if resolved:
                    opts["house"] = resolved
            info = self.backend.load_house(house_id or None, **opts)
            self._house_loaded = True
            self._house = info.to_dict()
        return self._house

    def house(self) -> Optional[Dict[str, Any]]:
        return self._house

    def start(self, target_category: str, **opts) -> Dict[str, Any]:
        if not self._house_loaded:
            self.load_house(**opts)
        self.env = EmbodiedEnvironment(self.backend)
        trans = self.env.start_episode(
            self._house["house_id"], target_category, **opts
        )
        return self._transition_payload(trans)

    def step(self, action_type: str) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("no active episode; call /start first")
        trans = self.env.step(AgentActionType(action_type))
        if trans is None:
            return self.state()
        return self._transition_payload(trans)

    def state(self) -> Dict[str, Any]:
        if self.env is None:
            return {"active": False}
        terminal = self.env.episode.status.value != "running"
        return {
            "active": True,
            "terminal": terminal,
            "backend": self.backend_name,
            "task": self.env.task.agent_view(),
            "episode": self.env.episode.to_dict(),
            "agent_state": self.env.agent_state.to_dict(),
            "model_input": self._last_model_input,
            "privileged": self._last_privileged,
            "step": self.env.episode.step,
            "observation_meta": {
                "step": self.env.observation.step,
                "timestamp_ms": self.env.observation.timestamp_ms,
                "frame_id": self.env.observation.frame_id,
                "camera_horizon_deg": self.env.observation.camera_horizon_deg,
                "target_visible": self.env.observation.target_visible,
            },
        }

    def observation_image_png(self) -> Optional[bytes]:
        if self.env is None or self.env.observation is None:
            return None
        rgb = self.env.observation.rgb
        if rgb is None:
            return None
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def teacher_path(self) -> Dict[str, Any]:
        from spatialforge.embodied.teacher import GridTeacher

        if self.env is None:
            raise RuntimeError("start an episode before requesting teacher path")
        task = self.env.task
        if task.target_object_ids:
            p = self.backend.object_position(task.target_object_ids[0])
            target_cell = (int(p[0]), int(p[1])) if p else (0, 0)
        else:
            target_cell = (0, 0)
        start_cell = (int(self.env.agent_state.position[0]), int(self.env.agent_state.position[1]))
        start_heading = int(self.env.agent_state.rotation_yaw_deg // 90) % 4
        path = GridTeacher(self.backend).plan_to_visible(
            target_cell, start_cell, start_heading
        )
        return {
            "planned": path.planned,
            "actions": list(path.actions),
            "detail": path.detail,
            "_privileged_teacher_only": True,
        }

    def _transition_payload(self, trans) -> Dict[str, Any]:
        self._last_model_input = trans.model_input
        self._last_privileged = trans.privileged
        return self.state()
