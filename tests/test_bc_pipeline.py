"""CPU tests: BC sample building, action parser, worker-pool partitioning,
manifest aggregation and house-level split integrity (no ai2thor required).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from spatialforge.embodied import bc_manifest as bm
from spatialforge.embodied.action_parser import parse_action_output
from spatialforge.embodied.bc_dataset import (
    BC_ACTION_VOCAB,
    action_distribution,
    build_bc_question,
    scan_rows_leakage,
    scan_text_leakage,
    student_record_to_rows,
)
from spatialforge.embodied.worker_pool import aggregate_results, partition_jobs


# ----------------------------------------------------------------------
# action parser
# ----------------------------------------------------------------------
class TestActionParser:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("MoveAhead", "MoveAhead"),
            ("RotateLeft", "RotateLeft"),
            ("RotateRight.", "RotateRight"),
            ("\nLookUp", "LookUp"),
            ("  LookDown  ", "LookDown"),
            ("Crouch", "Crouch"),
            ("Stand", "Stand"),
            ("Done", "Done"),
            ("moveahead", "MoveAhead"),
            ("`RotateLeft`", "RotateLeft"),
            ("MoveAhead\n\n", "MoveAhead"),
        ],
    )
    def test_valid(self, raw, expected):
        r = parse_action_output(raw)
        assert r["valid"] is True
        assert r["action"] == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "I see a mug but not sure what to do",
         "RotateRight and then LookDown", "MoveAhead Done", "apple pie",
         "Understand the scene", "WalkAhead", "I will MoveAhead now.",
         "The action is: RotateLeft because I need to turn"],
    )
    def test_invalid(self, raw):
        r = parse_action_output(raw)
        assert r["valid"] is False
        assert r["action"] is None

    def test_no_silent_substitution(self):
        assert parse_action_output("zzz")["reason"] != "exact"


# ----------------------------------------------------------------------
# bc prompt + records expansion
# ----------------------------------------------------------------------
def _hist(types, blocked=None):
    return [
        {"action_type": t, "step": i + 1, "success": False if blocked and t in blocked else True,
         "collision": False, "blocked": t in (blocked or [])}
        for i, t in enumerate(types)
    ]


def _sample_record():
    hist = _hist(["MoveAhead", "RotateLeft"])
    goal = {"instruction": "Find a mug.", "target_category": "mug",
            "house_id": "house_x", "max_steps": 120}
    steps = [
        {"role": "context", "step": 0, "observation_ref": {
            "step": 0, "image_path": "ep-1/frames/frame_0000.png",
            "camera_horizon_deg": 0.0, "rgb_shape": [256, 256, 3]},
         "action_history": [], "teacher_action": None},
        {"role": "action", "step": 1,
         "observation_ref": {"step": 1, "image_path": "ep-1/frames/frame_0001.png",
                             "camera_horizon_deg": 0.0, "rgb_shape": [256, 256, 3]},
         "observation": {"step": 1, "camera_horizon_deg": 0.0,
                         "action_history": [], "rgb_shape": [256, 256, 3]},
         "action_history": hist, "teacher_action": "MoveAhead",
         "teacher_action_success": True, "terminal": False},
        {"role": "action", "step": 2,
         "observation_ref": {"step": 2, "image_path": "ep-1/frames/frame_0002.png",
                             "camera_horizon_deg": 0.0, "rgb_shape": [256, 256, 3]},
         "observation": {"step": 2, "camera_horizon_deg": 0.0,
                         "rgb_shape": [256, 256, 3]},
         "action_history": hist + [{"action_type": "MoveAhead", "step": 2,
                                    "success": True, "collision": False,
                                    "blocked": False}],
         "teacher_action": "RotateLeft", "teacher_action_success": True,
         "verifier_target_visible": True,
         "terminal": False},
        {"role": "action", "step": 3,
         "observation_ref": {"step": 3, "image_path": "ep-1/frames/frame_0003.png",
                             "camera_horizon_deg": 0.0, "rgb_shape": [256, 256, 3]},
         "observation": {"step": 3, "camera_horizon_deg": 0.0,
                         "rgb_shape": [256, 256, 3]},
         "action_history": hist + [
             {"action_type": "MoveAhead", "step": 2, "success": True,
              "collision": False, "blocked": False},
             {"action_type": "RotateLeft", "step": 3, "success": True,
              "collision": False, "blocked": False}],
         "teacher_action": "Done", "teacher_action_success": True,
         "verifier_target_visible": True,
         "terminal": True},
    ]
    return {"schema": "student_training_record.v1", "domain": "student_training",
            "episode_id": "ep-1", "goal": goal, "steps": steps}


class TestBcSampleRows:
    def test_rows_mapping_and_done_target(self):
        rows = student_record_to_rows(_sample_record())
        # init obs -> MoveAhead; obs step1 -> RotateLeft; obs step2(visible) -> Done
        assert len(rows) == 3
        assert [r["answer"] for r in rows] == ["MoveAhead", "RotateLeft", "Done"]
        assert rows[2]["terminal_target"] is True
        assert rows[0]["image_path"].endswith("frame_0000.png")  # initial obs

    def test_question_format(self):
        q = build_bc_question("Find a mug.", "mug", 0.0,
                              _hist(["MoveAhead", "MoveAhead(blocked)".split()[0]]),
                              max_steps=150, current_step=3)
        assert "Goal: Find a mug." in q
        assert "MoveAhead" in q
        for a in BC_ACTION_VOCAB:
            assert a in q
        assert "Step: 3 of 150." in q

    def test_invisible_done_target_rows_filtered(self):
        rec = _sample_record()
        # invisible Done decision -> row must be dropped (keeps Done truthful)
        for st in rec["steps"]:
            st.pop("verifier_target_visible", None)
        rows = student_record_to_rows(rec)
        assert len(rows) == 2
        assert all(r["answer"] != "Done" for r in rows)

    def test_question_leak_free(self):
        rows = student_record_to_rows(_sample_record())
        for r in rows:
            assert scan_text_leakage(r["question"]) is None
        assert scan_rows_leakage(rows) == []

    def test_scan_text_leakage_catches_terms(self):
        assert scan_text_leakage("go to target_positions entry 0") is not None
        assert scan_text_leakage("the world position is x y z") is not None
        assert scan_text_leakage("go to reachable cell") is not None
        assert scan_text_leakage("target_position is (1.0, 2.0)") is not None
        assert scan_text_leakage("nice mug on the table") is None
        assert scan_text_leakage("turn left toward the window") is None


class TestManifest:
    def _row(self, house, ep, step, ans="MoveAhead", cat="mug"):
        return {"id": f"{ep}-s{step}", "question": "Q", "answer": ans,
                "image_path": "x.png", "scene_id": house, "episode_id": ep,
                "category": cat, "split": "all"}

    def test_split_disjoint_enforced(self):
        rows = [self._row("h1", "e1", 1), self._row("h2", "e2", 1)]
        tr, vr = bm.split_rows_by_house(rows, ["h1"], ["h2"])
        assert len(tr) == 1 and len(vr) == 1
        with pytest.raises(ValueError):
            bm.split_rows_by_house(rows, ["h1", "h2"], ["h2"])

    def test_split_unknown_house_raises(self):
        rows = [self._row("h3", "e3", 1)]
        with pytest.raises(ValueError):
            bm.split_rows_by_house(rows, ["h1"], ["h2"])

    def test_deterministic_cap_stratified(self):
        rows = [self._row(f"h{i % 3}", f"e{j}", k) for i in range(6)
                for j in range(3) for k in range(4)]
        capped = bm.cap_rows_deterministic(rows, 20)
        assert len(capped) == 20
        again = bm.cap_rows_deterministic(rows, 20)
        assert [r["id"] for r in capped] == [r["id"] for r in again]
        # every house keeps >0 rows while total <= cap
        houses = {r["scene_id"] for r in capped}
        assert houses == {f"h{i % 3}" for i in range(6)}

    def test_report_dataset_counts(self):
        rows = [self._row("h1", "e1", 1, ans="MoveAhead"),
                self._row("h1", "e2", 1, ans="RotateLeft"),
                self._row("h2", "e3", 1, ans="MoveAhead")]
        rep = bm.report_dataset(rows)
        assert rep["total_samples"] == 3
        assert rep["action_counts"]["MoveAhead"] == 2
        assert rep["episodes"] == 3 and rep["houses"] == 2

    def test_student_loaders_skip_meta_dirs(self, tmp_path):
        ep = tmp_path / "ep-abc"
        ep.mkdir()
        (ep / "student_training_record.json").write_text(
            json.dumps({"schema": "student_training_record.v1"}) + "\n")
        (tmp_path / "runs").mkdir()
        assert len(bm.load_student_records(str(tmp_path))) == 1


class TestWorkerPool:
    def test_partition_round_robin(self):
        jobs = [{"job_id": f"j{i}"} for i in range(7)]
        shards = partition_jobs(jobs, 3)
        assert [len(s) for s in shards] == [3, 2, 2]
        flat = [j["job_id"] for s in shards for j in s]
        assert sorted(flat) == ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]
        # round-robin order within shards preserved
        assert shards[0] == [jobs[0], jobs[3], jobs[6]]

    def test_aggregate_metrics(self):
        rows = [
            {"steps": 10, "success": True, "outcome_code": "success",
             "elapsed_s": 20.0, "env_step_times": [100.0, 200.0, 300.0]},
            {"steps": 4, "success": True, "outcome_code": "success",
             "elapsed_s": 12.0, "env_step_times": [250.0, 250.0]},
            {"steps": 0, "success": False, "outcome_code": "invalid_task",
             "elapsed_s": 1.0, "env_step_times": []},
        ]
        agg = aggregate_results(rows, wall_s=60.0)
        assert agg["total_env_steps"] == 14
        assert agg["successful_episodes"] == 2
        assert agg["episodes_per_min"] == pytest.approx(3.0)
        assert agg["env_step_latency_ms"]["count"] == 5
        assert agg["env_step_latency_ms"]["p50_ms"] == 250.0
        assert agg["aggregate_steps_per_sec"] == pytest.approx(14 / 60.0, abs=1e-3)

    def test_action_distribution(self):
        rows = [{"answer": "MoveAhead"}, {"answer": "MoveAhead"}, {"answer": "Done"}]
        d = action_distribution(rows)
        assert d["counts"]["MoveAhead"] == 2 and d["counts"]["Done"] == 1
        assert d["total"] == 3
