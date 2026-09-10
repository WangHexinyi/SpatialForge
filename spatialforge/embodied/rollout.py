"""Real ProcTHOR object-search episode generation (teacher rollouts).

Ties the pieces together for a genuine, real-runtime episode:

    house sample -> target category/instance -> (privileged view-pose search)
    -> spawn sample with initial-visible *resampling* -> start_episode
    -> ThorObjectSearchTeacher genuinely drives controller.step()
    -> Done -> verifier decides success / failure / timeout
    -> two strictly-separated records (student training / privileged research)
      + per-step first-person frames persisted for the Inspector.

This is researcher orchestration; it never relaxes the model-input allowlist.
The student record is re-validated for leaks before it is returned.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from spatialforge.embodied.contracts import AgentActionType, EpisodeStatus, TaskStatus
from spatialforge.embodied.environment import EmbodiedEnvironment
from spatialforge.embodied.procthor_teacher import (
    GRID_SPACING,
    ThorObjectSearchTeacher,
    bfs_path,
    build_reachable_grid,
    to_grid_key,
)
from spatialforge.embodied.records import (
    build_privileged_research_record,
    build_student_training_record,
)
from spatialforge.embodied.rendering import resolve_render_quality
from spatialforge.embodied.spawn_semantics import setup_block

# structural / non-searchable categories we never pick as targets
_NON_TARGET = {
    "wall", "floor", "ceiling", "doorway", "window", "cabinet", "dresser",
    "shelf", "countertop", "toilet", "bathtub", "fridge", "lightswitch",
    "houseplant", "table", "armchair", "sofa", "bed", "bathtubbasin",
    "chair", "desk", "bookcase", "television", "kitchencounter",
}


def is_searchable_category(category: str) -> bool:
    return category.lower() not in _NON_TARGET


class ProcthorEpisodeGenerator:
    """Runs real object-search teacher rollouts against a loaded house.

    Construct with a house source; the backend stays open so multiple episodes
    reuse the loaded house (house load is the expensive part).
    """

    def __init__(
        self,
        house_source: Any,
        width: int = 192,
        height: int = 192,
        x_display: Optional[str] = None,
        out_dir: Optional[str] = None,
        seed: int = 0,
        require_nvidia: bool = False,
        max_instance_probes: int = 4,
    ):
        self.out_dir = out_dir
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        #: privileged view-pose search cap across candidate target instances
        #: (throughput engineering on real houses; does not alter recorded
        #: student observations/actions, and never weakens the verifier).
        self.max_instance_probes = int(max_instance_probes)

        from spatialforge.embodied.backends.procthor import ProcTHORBackend

        self.backend = ProcTHORBackend(
            house=house_source, width=width, height=height,
            quality=resolve_render_quality(),
            x_display=x_display, require_nvidia=require_nvidia,
        )
        self.house_info = self.backend.load_house(seed=self.seed)
        self.house_id = self.house_info.house_id
        self._closed = False
        self._scene_written = False

    def close(self) -> None:
        if not self._closed:
            self.backend.close()
            self._closed = True

    # ------------------------------------------------------------------
    def list_categories(self) -> Dict[str, int]:
        return dict(self.house_info.object_category_counts)

    def target_instance_ids(self, category: str) -> List[str]:
        return self.backend.target_category_object_ids(category)

    # ------------------------------------------------------------------
    def generate_episode(
        self,
        category: str,
        max_steps: int = 120,
        allow_easy: bool = False,
        min_spawn_m: float = 2.0,
        max_spawn_m: float = 8.0,
        demo_resample: bool = False,
        tag: str = "run",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate one real episode; returns a full summary dict.

        When ``seed`` is given, an episode-local RNG (``random.Random(seed)``) is
        used for spawn sampling so the episode content is independent of the
        worker's prior job sequence (deterministic worker partition sweeps).
        """
        started = time.time()
        episode_id = f"ep-{tag}-{uuid.uuid4().hex[:8]}"
        gen = random.Random(seed) if seed is not None else self.rng

        instance_ids = self.backend.target_category_object_ids(category)
        if not instance_ids:
            return self._fail_record(
                episode_id, category, "invalid_task",
                "category has no real instances in this house", started,
            )

        # pick a target instance that has an observable reachable pose
        target_oid = None
        pose = None
        for oid in instance_ids[: self.max_instance_probes]:
            p = self._plan_pose(oid)
            if p is not None:
                target_oid, pose = oid, p
                break
        if target_oid is None:
            return self._fail_record(
                episode_id, category, "unreachable_task",
                "no reachable/observable view pose for any instance", started,
            )

        # ---- spawn sampling with initial-visible resampling ----------
        spawn = self._sample_spawn(
            target_oid, allow_easy, min_spawn_m, max_spawn_m,
            demo_resample=demo_resample,
        )
        if spawn is None:
            return self._fail_record(
                episode_id, category, "spawn_resample_failed",
                "could not sample a non-trivially-visible reachable spawn", started,
            )

        # ---- genuine episode start ----------------------------------
        agent = {
            "position": list(spawn["position"]),
            "rotation": {"x": 0.0, "y": float(spawn["yaw"]), "z": 0.0},
            "horizon": float(spawn["horizon"]),
            "standing": True,
        }
        env = EmbodiedEnvironment(self.backend, episode_max_steps=max_steps)
        initial = env.start_episode(
            self.house_id, category, max_steps=max_steps, agent=agent
        )
        initially_visible = bool(env.observation.target_visible)
        setup = setup_block(spawn, initial.agent_state)

        # ---- genuine teacher drive -----------------------------------
        teacher = ThorObjectSearchTeacher(env, target_oid)
        plan = {"target_object_id": target_oid, **pose}
        res = teacher.run(pose)

        steps = len(teacher.transitions)
        status = env.episode.status.value
        success = env.episode.success

        # ---- frames + records ---------------------------------------
        frame_names, frame_dir = self._persist_frames(episode_id, initial, teacher.transitions)

        student = build_student_training_record(
            episode_id, env.task.agent_view(), teacher.transitions, initial.observation
        )
        privileged = build_privileged_research_record(
            episode_id, env.task.privileged_view(), env.episode, teacher.transitions,
            teacher_plan={"view_pose": pose, "planned": res.get("planned")},
            spawn={
                "position": list(spawn["position"]),
                "yaw": spawn["yaw"],
                "horizon": spawn["horizon"],
                "initially_visible": initially_visible,
                "resample_attempts": spawn["resample_attempts"],
                "min_dist_m": spawn["min_dist_m"],
            },
            frames=[os.path.relpath(p, self.out_dir) if self.out_dir else p for p in frame_names],
            setup=setup,
            render_quality=self.backend.quality,
            house_path=(
                os.path.abspath(str(self.backend.house_source))
                if self.backend.house_source else None
            ),
        )
        self._persist_records(episode_id, student, privileged)
        self._persist_scene()

        return {
            "episode_id": episode_id,
            "house_id": self.house_id,
            "category": category,
            "target_object_id": target_oid,
            "initial_agent_pose": list(spawn["position"]) + [spawn["yaw"], spawn["horizon"]],
            "initially_visible": initially_visible,
            "spawn_resample_attempts": spawn["resample_attempts"],
            "view_pose": pose,
            "planned": res.get("planned"),
            "reached": res.get("reached"),
            "final_target_visible": res.get("visible_at_done"),
            "status": status,
            "success": success,
            "reason": env.episode.success_reason,
            "steps": steps,
            "max_steps": max_steps,
            "teacher_action_sequence": [t.action.action_type.value for t in teacher.transitions if t.action],
            "env_step_times_ms": [round(t, 1) for t in env.env_step_times_ms],
            "student_record": student,
            "privileged_record": privileged,
            "frame_dir": frame_dir,
            "elapsed_s": round(time.time() - started, 1),
        }

    # ------------------------------------------------------------------
    def _plan_pose(self, oid: str):
        preview = ThorObjectSearchTeacher.__new__(ThorObjectSearchTeacher)
        preview.backend = self.backend
        preview.target_object_id = oid
        preview.env = None
        preview.max_tries_view_pose = 40
        preview.scan_horizons_deg = (0.0, 30.0, 45.0)
        return preview.plan_view_pose()

    def _sample_spawn(
        self, target_oid: str, allow_easy: bool,
        min_spawn_m: float, max_spawn_m: float, demo_resample: bool,
    ) -> Optional[Dict[str, Any]]:
        reachable = self.backend.reachable_positions()
        grid = build_reachable_grid(reachable)
        if not grid:
            return None
        tpos = self.backend.object_position(target_oid)
        tx, tz = float(tpos[0]), float(tpos[2])
        tk = to_grid_key(tx, tz)

        cells = list(grid.keys())
        self.rng.shuffle(cells)
        # rank: prefer moderate distance so the nav path stays within budget
        def dist(c):
            return ((c[0] - tk[0]) ** 2 + (c[1] - tk[1]) ** 2) ** 0.5 * GRID_SPACING

        cells.sort(key=lambda c: dist(c))
        candidates = [c for c in cells if min_spawn_m <= dist(c) <= max_spawn_m]

        probe_attempts = 0
        if demo_resample:
            # intentionally probe an easy (visible) spawn first to demonstrate resampling
            easy = min(cells, key=dist)
            if dist(easy) < min_spawn_m:
                self.backend.set_agent_pose(
                    (easy[0] * GRID_SPACING, grid[easy], easy[1] * GRID_SPACING),
                    0.0, 0.0, render=False,
                )
                probe_attempts += 1

        tried = probe_attempts
        for cell in candidates:
            if tried >= 40:
                break
            y = grid[cell]
            d = dist(cell)
            md = self.backend.set_agent_pose(
                (cell[0] * GRID_SPACING, y, cell[1] * GRID_SPACING),
                0.0, 0.0, render=False,
            )
            tried += 1
            visible = self._visible_in(md, target_oid)
            if allow_easy or (not visible and d >= min_spawn_m):
                return {
                    "position": (cell[0] * GRID_SPACING, y, cell[1] * GRID_SPACING),
                    "yaw": 0.0,
                    "horizon": 0.0,
                    "initially_visible": visible,
                    "resample_attempts": tried,
                    "min_dist_m": d,
                    "allow_easy": allow_easy,
                }
        return None

    @staticmethod
    def _visible_in(md: dict, oid: str) -> bool:
        for obj in md.get("objects", []):
            if obj.get("objectId") == oid:
                return bool(obj.get("visible", False))
        return False

    # ------------------------------------------------------------------
    def _persist_frames(self, episode_id, initial, transitions):
        if not self.out_dir:
            return [], None
        frame_dir = os.path.join(self.out_dir, episode_id, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        names = []

        def save(obs, i):
            if obs is None or obs.rgb is None:
                return
            path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            Image.fromarray(obs.rgb.astype(np.uint8)).save(path)
            obs.image_path = os.path.relpath(path, self.out_dir)
            obs.frame_id = f"{episode_id}-{i}"
            names.append(path)

        save(initial.observation, 0)
        for idx, trans in enumerate(transitions, start=1):
            save(trans.observation, idx)
        return names, frame_dir

    def _persist_records(self, episode_id, student, privileged):
        if not self.out_dir:
            return
        ep_dir = os.path.join(self.out_dir, episode_id)
        os.makedirs(ep_dir, exist_ok=True)
        import json

        with open(os.path.join(ep_dir, "student_training_record.json"), "w") as f:
            json.dump(student, f, indent=2, default=_json_default)
        with open(os.path.join(ep_dir, "privileged_research_record.json"), "w") as f:
            json.dump(privileged, f, indent=2, default=_json_default)

    def _persist_scene(self) -> None:
        """Persist the 3D semantic scene once per loaded house (God View).

        Stored under ``<out_dir>/_scenes/<house_stem>.json`` so episodes can be
        replayed in 3D without re-instantiating AI2-THOR.
        """
        if not self.out_dir or self._scene_written:
            return
        try:
            from spatialforge.embodied.bc_manifest import house_stem
            import json

            src = getattr(self.backend, "house_source", None)
            stem = house_stem(str(src)) if src else "house"
            scene_dir = os.path.join(self.out_dir, "_scenes")
            os.makedirs(scene_dir, exist_ok=True)
            payload = {
                "schema": "spatialforge_scene3d_episode_source.v1",
                "house_stem": stem,
                "house_path": os.path.abspath(str(src)) if src else None,
                "scene": self.backend.scene_geometry(),
            }
            tmp = os.path.join(scene_dir, f".{stem}.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=_json_default)
            os.replace(tmp, os.path.join(scene_dir, f"{stem}.json"))
            self._scene_written = True
        except Exception:
            # scene capture is researcher tooling; never fail a real rollout
            self._scene_written = True

    def _fail_record(self, episode_id, category, code, reason, started):
        record = {
            "schema": "privileged_research_record.v1",
            "domain": "privileged_research",
            "_privileged": True,
            "episode_id": episode_id,
            "house_id": self.house_id,
            "category": category,
            "outcome": {"code": code, "reason": reason, "success": False,
                        "status": "failure", "steps": 0},
            "teacher_plan": {},
            "steps": [],
        }
        return {
            "episode_id": episode_id,
            "house_id": self.house_id,
            "category": category,
            "target_object_id": None,
            "initial_agent_pose": None,
            "status": "failure",
            "success": False,
            "outcome_code": code,
            "reason": reason,
            "steps": 0,
            "elapsed_s": round(time.time() - started, 1),
            "student_record": {"schema": "student_training_record.v1",
                               "domain": "student_training", "episode_id": episode_id,
                               "goal": {"target_category": category}, "steps": []},
            "privileged_record": record,
            "frame_dir": None,
        }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
