"""Embodied vertical-slice smoke.

Runs the full closed loop required by the vertical slice:

    house load -> start episode -> observation -> MoveAhead/Rotate
    -> new observation/state -> teacher path -> replay -> Done -> verifier

The default backend is the deterministic CPU harness (headless, always runs and
always proves the runtime loop + verifier + isolation). Passing ``--procthor``
attempts a real ProcTHOR / AI2-THOR house; if the interpreter lacks ai2thor or no
Unity build is configured it reports the blocker explicitly (it is never faked).
"""

from __future__ import annotations

import json
import sys
import time

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check(model_input: dict, label: str) -> None:
    s = json.dumps(model_input)
    forbidden = (
        "target_positions", "target_object_ids", "reachable_positions",
        "teacher_path", "position", "segmentation", "privileged", "shortest",
    )
    hits = [f for f in forbidden if f in s]
    if hits:
        raise AssertionError(f"{label}: isolation leak -> {hits}")


def run_deterministic(target="mug") -> dict:
    from spatialforge.embodied.backends.deterministic import DeterministicGridBackend
    from spatialforge.embodied.contracts import AgentActionType
    from spatialforge.embodied.environment import EmbodiedEnvironment
    from spatialforge.embodied.teacher import GridTeacher

    backend = DeterministicGridBackend()
    house = backend.load_house(None, seed=0)
    print(f"[house] loaded {house.house_id} rooms={house.room_count} "
          f"cats={house.object_category_counts} backend={backend.LABEL}")

    env = EmbodiedEnvironment(backend)
    tr = env.start_episode(house.house_id, target, max_steps=100)
    print(f"[episode] task={tr.task.task_id} target={target} step0 frame="
          f"{tr.observation.frame_id} rgb={tr.observation.rgb.shape}")
    _check(tr.model_input, "reset")

    # agent actually moves and the first-person RGB actually changes
    f0 = tr.observation.rgb.copy()
    tr = env.step(AgentActionType.MOVE_AHEAD)
    rgb_changed = not (f0 == tr.observation.rgb).all()
    print(f"[transition] MoveAhead -> step {env.episode.step} pos="
          f"{env.agent_state.position[:2]} yaw={env.agent_state.rotation_yaw_deg} "
          f"rgb_changed={rgb_changed}")
    assert rgb_changed, "first-person RGB did not change after MoveAhead"
    _check(tr.model_input, "after MoveAhead")

    # rotation changes heading
    tr = env.step(AgentActionType.ROTATE_LEFT)
    print(f"[transition] RotateLeft -> yaw={env.agent_state.rotation_yaw_deg}")

    # teacher path
    tp = backend.object_position(tr.task.target_object_ids[0])
    start = (int(env.agent_state.position[0]), int(env.agent_state.position[1]))
    h = int(env.agent_state.rotation_yaw_deg // 90) % 4
    path = GridTeacher(backend).plan_to_visible((int(tp[0]), int(tp[1])), start, h)
    print(f"[teacher] planned={path.planned} actions={path.actions}")

    # reset a fresh deterministic env, spawn facing away, replay teacher -> Done
    env2 = EmbodiedEnvironment(DeterministicGridBackend())
    env2.backend.load_house(None, seed=0)
    tr2 = env2.start_episode(house.house_id, target, start_heading=2)
    tp2 = env2.backend.object_position(tr2.task.target_object_ids[0])
    s2 = (int(env2.agent_state.position[0]), int(env2.agent_state.position[1]))
    h2 = int(env2.agent_state.rotation_yaw_deg // 90) % 4
    path2 = GridTeacher(env2.backend).plan_to_visible((int(tp2[0]), int(tp2[1])), s2, h2)
    print(f"[teacher-away] planned={path2.planned} actions={path2.actions}")
    for a in path2.actions:
        env2.step(AgentActionType(a))
    fin = env2.step(AgentActionType.DONE)
    print(f"[verifier] Done -> terminal={fin.terminal} success={fin.episode.success} "
          f"reason={fin.episode.success_reason}")
    assert fin.episode.success, "vertical slice failed: verifier rejected Done"
    _check(fin.model_input, "final")
    return {"ok": True, "backend": backend.LABEL}


def run_procthor(house: str) -> dict:
    from spatialforge.embodied.backends.procthor import ProcTHORBackend
    from spatialforge.embodied.contracts import AgentActionType
    from spatialforge.embodied.environment import EmbodiedEnvironment

    backend = ProcTHORBackend()
    backend.load_house(house)
    env = EmbodiedEnvironment(backend)
    tr = env.start_episode(house, "Mug", max_steps=100)
    print(f"[procthor] loaded {backend.house().to_dict()}")
    tr = env.step(AgentActionType.MOVE_AHEAD)
    print(f"[procthor] MoveAhead success={tr.action.success} pos="
          f"{env.agent_state.position}")
    tr = env.step(AgentActionType.DONE)
    print(f"[procthor] verifier success={env.episode.success} reason="
          f"{env.episode.success_reason}")
    backend.close()
    return {"ok": True, "backend": backend.LABEL}


def main() -> int:
    if "--procthor" in sys.argv:
        house = None
        for a in sys.argv:
            if a.startswith("--house="):
                house = a.split("=", 1)[1]
        if not house:
            print("ERROR: --procthor requires --house=<procthor-scene>")
            return 2
        try:
            result = run_procthor(house)
        except Exception as e:  # noqa: BLE001
            print(f"[procthor] BLOCKER: {type(e).__name__}: {e}")
            return 3
    else:
        result = run_deterministic()
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
