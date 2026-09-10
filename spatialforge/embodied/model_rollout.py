"""Model-controlled closed-loop episode generation (real AI2-THOR).

Phase-1 student policy pi(a_t | RGB_t, goal, permitted history) is queried over
HTTP for every decision; only its parsed action is ever executed:

    Observation_t -> Qwen action inference -> strict parse
        -> (valid) env.step(action) / Done -> verifier
        -> (invalid) counted, never converted into an environment action
    -> Observation_{t+1} -> ...

No teacher trajectory replay, no teacher action fallback, no human correction,
no scripted action substitution. After ``max_consecutive_invalid`` invalid
outputs in a row the episode finishes deterministically as
``invalid_exhausted`` (reported policy).

Episodes persist in the same layout as teacher rollouts (init frame + one frame
per executed action + student/privileged records), extended with a per-decision
log (model raw output, parsed action, latency, executed flag) and an atomic
``live_state.json`` for the researcher God View.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatialforge.embodied.bc_dataset import build_bc_question
from spatialforge.embodied.contracts import AgentActionType, EpisodeStatus
from spatialforge.embodied.environment import EmbodiedEnvironment
from spatialforge.embodied.records import (
    build_privileged_research_record,
    build_student_training_record,
)
from spatialforge.embodied.rollout import ProcthorEpisodeGenerator
from spatialforge.embodied.spawn_semantics import setup_block

_EXECUTABLE = frozenset({
    "MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown",
    "Crouch", "Stand",
})

_ACTION_BY_LABEL = {m.value: m for m in AgentActionType}


class ModelActionClient:
    """Minimal HTTP client to the torch-side action server (stdlib only)."""

    def __init__(self, url: str, timeout_s: float = 180.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        import urllib.request

        self._request = urllib.request
        # localhost-only server: never route through ambient HTTP proxies
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    def decide(self, rgb, question: str) -> Dict[str, Any]:
        buf = io.BytesIO()
        Image.fromarray(rgb.astype("uint8")).save(buf, format="PNG")
        payload = {
            "question": question,
            "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
        req = self._request.Request(
            self.url + "/action",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


def permitted_history_list(env: EmbodiedEnvironment) -> List[Dict[str, Any]]:
    """Recent executed actions in the permitted (student) representation."""
    out = []
    if env.history is not None:
        for action in env.history.actions:
            if action is None:
                continue
            d = action.to_dict()
            out.append({
                "action_type": d["action_type"],
                "step": d["step"],
                "success": d.get("success"),
                "collision": d.get("collision"),
                "blocked": d.get("blocked"),
            })
    return out


def no_progress_update(cell, last_cell, count):
    """Track consecutive executed steps that stay in the same 0.5 m xz cell."""
    if cell == last_cell:
        return last_cell, count + 1
    return cell, 0


def build_decision_question(env: EmbodiedEnvironment) -> str:
    """Question text identical in format to the BC training prompts."""
    obs = env.observation
    task = env.task
    return build_bc_question(
        task.instruction,
        task.target_category,
        obs.camera_horizon_deg,
        permitted_history_list(env),
        max_steps=task.max_steps,
        current_step=env.episode.step if env.episode else 0,
    )


class ProcthorModelEpisodeGenerator(ProcthorEpisodeGenerator):
    """Real object-search episodes where the model chooses every action."""

    def generate_episode(  # type: ignore[override]
        self,
        category: str,
        client: ModelActionClient,
        max_steps: int = 120,
        min_spawn_m: float = 2.0,
        max_spawn_m: float = 8.0,
        tag: str = "model",
        seed: Optional[int] = None,
        max_consecutive_invalid: int = 5,
        setup_timeout_s: float = 150.0,
        max_no_progress_steps: int = 0,
    ) -> Dict[str, Any]:
        import random

        started = time.time()
        episode_id = f"ep-{tag}-{random.Random(seed or 0).randint(0, 10**9):08x}-{os.getpid()}"
        rng = random.Random(seed) if seed is not None else self.rng

        instance_ids = self.backend.target_category_object_ids(category)
        if not instance_ids:
            return self._fail_record(
                episode_id, category, "invalid_task",
                "category has no real instances in this house", started,
            )
        # privileged target/view-pose/spawn selection (same policy as teacher)
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
        spawn = self._sample_spawn(
            target_oid, False, min_spawn_m, max_spawn_m, demo_resample=False,
        )
        if spawn is None:
            return self._fail_record(
                episode_id, category, "spawn_resample_failed",
                "could not sample a non-trivially-visible reachable spawn", started,
            )
        if time.time() - started > setup_timeout_s:
            # pathological engine load: privileged setup exceeded the budget;
            # fail loudly instead of burning the worker (reported outcome).
            return self._fail_record(
                episode_id, category, "setup_timeout",
                f"privileged view-pose/spawn setup exceeded {setup_timeout_s:.0f}s", started,
            )

        env = EmbodiedEnvironment(self.backend, episode_max_steps=max_steps)
        agent = {
            "position": list(spawn["position"]),
            "rotation": {"x": 0.0, "y": float(spawn["yaw"]), "z": 0.0},
            "horizon": float(spawn["horizon"]),
            "standing": True,
        }
        initial = env.start_episode(
            self.house_id, category, max_steps=max_steps, agent=agent
        )
        initial_visible = bool(env.observation.target_visible)
        setup = setup_block(spawn, initial.agent_state)

        decisions: List[Dict[str, Any]] = []
        executed: List[Any] = []  # genuine EmbodiedTransitions from env.step
        consec_invalid = 0
        terminal_reason = None
        #: opt-in no-progress early stop (raw decisions/frames are still
        #: persisted); disabled by default so existing experiments are unchanged.
        _last_cell = None
        _no_progress = 0

        while env.episode.status == EpisodeStatus.RUNNING:
            frame_ref = len(executed)
            q = build_decision_question(env)
            decision = {
                "decision_idx": len(decisions),
                "frame": f"frame_{frame_ref:04d}.png",
                "step": env.episode.step,
                "question": q,
            }
            t0 = time.time()
            try:
                resp = client.decide(env.observation.rgb, q)
            except Exception as exc:  # model server failure -> real blocker
                return self._fail_record(
                    episode_id, category, "inference_failure",
                    f"model server error: {str(exc)[-300:]}", started,
                )
            decision["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
            parsed = resp.get("parsed", {})
            decision["model_raw"] = str(resp.get("raw", ""))
            decision["model_action"] = parsed.get("action")
            decision["parse_reason"] = parsed.get("reason")
            # privileged snapshot of what the model saw / where it was
            if env.observation is not None:
                decision["authoritative_target_visible"] = bool(env.observation.target_visible)
            if env.agent_state is not None:
                decision["agent"] = {
                    "position": list(env.agent_state.position),
                    "rotation_yaw_deg": env.agent_state.rotation_yaw_deg,
                    "camera_horizon_deg": env.agent_state.camera_horizon_deg,
                    "is_crouching": env.agent_state.is_crouching,
                }
            decisions.append(decision)
            self._write_live(episode_id, env, decision, started, len(decisions))

            action_name = decision["model_action"]
            if (action_name is None
                    or (action_name not in _EXECUTABLE and action_name != "Done")):
                decision["executed"] = False
                decision["invalid"] = True
                consec_invalid += 1
                if consec_invalid >= max_consecutive_invalid:
                    env._finish(
                        EpisodeStatus.FAILURE, False,
                        f"exhausted {max_consecutive_invalid} consecutive invalid "
                        "model outputs (deterministic failure policy).",
                    )
                    terminal_reason = "invalid_exhausted"
                continue

            consec_invalid = 0
            decision["executed"] = True
            decision["invalid"] = False
            if action_name == "Done":
                trans = env.step(AgentActionType.DONE)  # verifier owns success
                if trans is not None:
                    executed.append(trans)
                break
            trans = env.step(_ACTION_BY_LABEL[action_name])
            if trans is not None:
                executed.append(trans)
                if max_no_progress_steps > 0:
                    st = trans.agent_state
                    cell = (round(st.position[0] * 2) / 2, round(st.position[2] * 2) / 2)
                    _last_cell, _no_progress = no_progress_update(
                        cell, _last_cell, _no_progress
                    )
                    if _no_progress >= max_no_progress_steps:
                        env._finish(
                            EpisodeStatus.FAILURE, False,
                            f"no_progress_early_stop: {_no_progress} consecutive "
                            "steps without spatial progress.",
                        )
                        terminal_reason = "no_progress_early_stop"
                        break

        # persist genuine frames: init + one per executed transition (same layout
        # as teacher rollouts; the Done transition duplicates the last frame)
        frame_names, frame_dir = self._persist_episode_frames(
            episode_id, initial.observation, executed
        )

        # student-shape record: strictly the executed *student* behavior
        # (model decisions, not teacher truth); never enters BC training data.
        student = build_student_training_record(
            episode_id, env.task.agent_view(), executed, initial.observation
        )
        privileged = build_privileged_research_record(
            episode_id, env.task.privileged_view(), env.episode, executed,
            teacher_plan={
                "view_pose": pose,
                "target_object_id": target_oid,
                "mode": "model_controlled",
                "model_controlled": True,
            },
            spawn={
                "position": list(spawn["position"]),
                "yaw": spawn["yaw"],
                "horizon": spawn["horizon"],
                "initially_visible": initial_visible,
                "resample_attempts": spawn["resample_attempts"],
            },
            frames=[os.path.relpath(p, self.out_dir) if self.out_dir else p for p in frame_names],
            setup=setup,
            render_quality=self.backend.quality,
            house_path=(
                os.path.abspath(str(self.backend.house_source))
                if self.backend.house_source else None
            ),
        )
        privileged["decisions"] = decisions
        privileged["model_controlled"] = True
        privileged["terminal_reason"] = terminal_reason or env.episode.success_reason
        self._persist_episode_records(episode_id, student, privileged, decisions)
        self._persist_scene()
        self._clear_live(episode_id)

        invalid = [d for d in decisions if d.get("invalid")]
        status = env.episode.status.value
        return {
            "episode_id": episode_id,
            "house_id": self.house_id,
            "category": category,
            "target_object_id": target_oid,
            "initial_agent_pose": list(spawn["position"]) + [spawn["yaw"], spawn["horizon"]],
            "initially_visible": initial_visible,
            "status": status,
            "success": bool(env.episode.success),
            "reason": terminal_reason or env.episode.success_reason,
            "steps": len(executed),
            "executed_actions": [t.action.action_type.value for t in executed if t and t.action],
            "decision_count": len(decisions),
            "invalid_decision_count": len(invalid),
            "model_action_sequence": [d.get("model_action") for d in decisions],
            "max_steps": max_steps,
            "elapsed_s": round(time.time() - started, 1),
            "student_record": student,
            "privileged_record": privileged,
            "frame_dir": frame_dir,
        }

    # ------------------------------------------------------------------
    def _persist_episode_frames(self, episode_id, initial_obs, executed):
        if not self.out_dir:
            return [], None
        frame_dir = os.path.join(self.out_dir, episode_id, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        names = []
        entries = [(initial_obs, 0)] + [(t.observation, i + 1) for i, t in enumerate(executed)]
        for obs, index in entries:
            if obs is None or obs.rgb is None:
                continue
            path = os.path.join(frame_dir, f"frame_{index:04d}.png")
            if not os.path.exists(path):
                Image.fromarray(obs.rgb.astype(np.uint8)).save(path)
            obs.image_path = os.path.relpath(path, self.out_dir) if self.out_dir else path
            names.append(path)
        return names, frame_dir

    def _persist_episode_records(self, episode_id, student, privileged, decisions):
        if not self.out_dir:
            return
        ep_dir = os.path.join(self.out_dir, episode_id)
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, "student_training_record.json"), "w") as f:
            json.dump(student, f, indent=2, default=str)
        with open(os.path.join(ep_dir, "privileged_research_record.json"), "w") as f:
            json.dump(privileged, f, indent=2, default=str)
        with open(os.path.join(ep_dir, "decisions.json"), "w") as f:
            json.dump(decisions, f, indent=2, default=str)

    def _persist_live_frame(self, episode_id, frame_ref, rgb) -> None:
        if not self.out_dir:
            return
        frame_dir = os.path.join(self.out_dir, episode_id, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        path = os.path.join(frame_dir, f"frame_{frame_ref:04d}.png")
        if not os.path.exists(path):
            Image.fromarray(rgb.astype(np.uint8)).save(path)

    def _write_live(self, episode_id, env, decision, started, idx) -> None:
        if not self.out_dir:
            return
        ep_dir = os.path.join(self.out_dir, episode_id)
        os.makedirs(ep_dir, exist_ok=True)
        state = {
            "episode_id": episode_id,
            "live": True,
            "decision_idx": idx,
            "step": env.episode.step,
            "status": env.episode.status.value,
            "elapsed_s": round(time.time() - started, 1),
            "decision": {k: v for k, v in decision.items() if k != "question"},
            "goal": env.task.instruction if env.task else None,
            "category": env.task.target_category if env.task else None,
            "frame": decision.get("frame"),
        }
        tmp = os.path.join(ep_dir, ".live_state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(ep_dir, "live_state.json"))

    def _clear_live(self, episode_id) -> None:
        if not self.out_dir:
            return
        try:
            os.remove(os.path.join(self.out_dir, episode_id, "live_state.json"))
        except OSError:
            pass


