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
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
from spatialforge.embodied.contracts import AgentActionType
from spatialforge.embodied.environment import EmbodiedEnvironment

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
        self._loaded_episode: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Saved teacher-rollout episode inspection (researcher-only)
    # ------------------------------------------------------------------
    def set_episodes_root(self, path) -> None:
        self.episodes_root = Path(path)
        self.episodes_root.mkdir(parents=True, exist_ok=True)

    def list_episodes(self) -> List[Dict[str, Any]]:
        if self.episodes_root is None or not self.episodes_root.is_dir():
            return []
        out = []
        for d in sorted(self.episodes_root.iterdir()):
            if not d.is_dir():
                continue
            priv = d / "privileged_research_record.json"
            if not priv.is_file():
                continue
            try:
                rec = json.loads(priv.read_text(encoding="utf-8"))
            except Exception:
                continue
            task = rec.get("task", {})
            ep = rec.get("episode", {}) or {}
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
                "model_controlled": bool(rec.get("model_controlled") or rec.get("decisions")),
                "model_terminal_reason": rec.get("terminal_reason"),
                "decision_count": len(rec.get("decisions", [])),
                "invalid_decision_count": sum(
                    1 for x in rec.get("decisions", []) if x.get("invalid")),
            })
        # in-progress model episodes (live_state.json without final records)
        for d in sorted(self.episodes_root.iterdir()):
            if not d.is_dir() or (d / "privileged_research_record.json").is_file():
                continue
            live = d / "live_state.json"
            if live.is_file():
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
                })
        return out

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """Load a saved episode + build a pageable timeline (privileged)."""
        if self.episodes_root is None:
            raise RuntimeError("no episodes root configured")
        d = (self.episodes_root / episode_id).resolve()
        if not str(d).startswith(str(self.episodes_root.resolve())):
            raise RuntimeError("unsafe episode id")
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
        header = {
            "episode_id": episode_id,
            "house_id": task.get("house_id") or rec.get("house_id"),
            "category": task.get("target_category") or rec.get("category"),
            "target_object_ids": task.get("target_object_ids") or rec.get("target_object_id"),
            "target_positions": task.get("target_positions"),
            "teacher_plan": rec.get("teacher_plan"),
            "spawn": rec.get("spawn"),
            "status": ep.get("status") or rec.get("outcome", {}).get("status"),
            "success": ep.get("success"),
            "success_reason": ep.get("success_reason") or rec.get("outcome", {}).get("reason"),
            "student_domain": student.get("domain") if student else None,
            "student_leak_checked": student.get("_leak_checked") if student else None,
            "model_controlled": bool(rec.get("model_controlled") or rec.get("decisions")),
            "model_terminal_reason": rec.get("terminal_reason"),
            "decision_count": len(rec.get("decisions", [])),
            "invalid_decision_count": sum(
                1 for x in rec.get("decisions", []) if x.get("invalid")),
            "_privileged": True,
        }

        timeline = [{
            "idx": 0, "type": "context", "step": 0, "frame": "frame_0000.png",
            "authoritative_target_visible": bool(rec.get("spawn", {}).get("initially_visible")),
            "note": "episode start (context)",
        }]
        for i, st in enumerate(rec.get("steps", []), start=1):
            timeline.append({
                "idx": i,
                "type": "action",
                "step": st.get("step"),
                "action": st.get("action_type"),
                "action_success": st.get("action_success"),
                "agent": st.get("agent"),
                "authoritative_target_visible": st.get("authoritative_target_visible"),
                "terminal": st.get("terminal"),
                "frame": f"frame_{i:04d}.png",
            })

        self._loaded_episode = {
            "episode_id": episode_id,
            "header": header,
            "timeline": timeline,
            "decisions": rec.get("decisions", []),
            "_dir": str(d),
        }
        return {
            "header": header,
            "timeline": timeline,
            "student_record": student,
            "decisions": rec.get("decisions", []),
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

    def load_house(self, house_id: str = "", **opts) -> Dict[str, Any]:
        if not self._house_loaded:
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
