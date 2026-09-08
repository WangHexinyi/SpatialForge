"""Read-only artifact loading, indexation, and path-traversal guarded access.

Scope:
- Resolves and verifies scene states, rendered observations, dataset samples,
  evaluation predictions, and live progress telemetry.
- Enforces strict read-only guarantees and path traversal prevention.
- Provides fallback and explicit unavailable states for missing artifacts.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from spatialforge.environment.challenge import analyze_scene_challenge
from spatialforge.environment.camera import CameraPose
from spatialforge.environment.scene import SceneState, load_scene_state
from spatialforge.environment.views import generate_cardinal_views, VIEW_IDS
from spatialforge.environment.sampling import (
    DIRECTIONS_26,
    canonical_camera,
    canonical_directions,
    jitter_pose,
)
from spatialforge.experiment.evaluation import EvaluationPredictionRecord
from spatialforge.inspector.contracts import (
    PredictionChannel,
    RuntimeChannel,
)


def resolve_safe_path(base_dir: Union[str, Path], relative_or_absolute: Union[str, Path]) -> Path:
    """Resolve a target path ensuring it strictly resides within allowed root directories.

    Prevents directory traversal (e.g. '../', '/etc/passwd').
    Raises:
        PermissionError: If the target path resolves outside base_dir.
        FileNotFoundError: If the target path does not exist.
    """
    base = Path(base_dir).resolve()
    target = Path(relative_or_absolute)
    if not target.is_absolute():
        resolved = (base / target).resolve()
    else:
        resolved = target.resolve()

    # Verify that resolved path begins with base
    try:
        resolved.relative_to(base)
    except ValueError:
        raise PermissionError(f"Access denied: Path {relative_or_absolute!r} escapes sandbox {base}")

    return resolved


class ArtifactRepository:
    """Central read-only indexer for SpatialForge experimental assets."""

    def __init__(self, repo_root: Optional[Union[str, Path]] = None):
        if repo_root is None:
            self.repo_root = Path(__file__).resolve().parents[2]
        else:
            self.repo_root = Path(repo_root).resolve()

        self.scenes_dir = self.repo_root / "outputs" / "scenes"
        self.experiments_dir = self.repo_root / "outputs" / "experiments"
        self.rendered_dir = self.experiments_dir / "g2.0-e" / "rendered"
        self.e1_dir = self.experiments_dir / "g2.0-e"

        # In-memory indexes
        self._samples_by_scene_view: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self._samples_by_id: Dict[str, Dict[str, Any]] = {}
        self._samples_indexed: bool = False

        self._predictions_by_run: Dict[str, Dict[str, EvaluationPredictionRecord]] = {}
        self._prediction_runs_scanned: bool = False

    def list_scenes(self) -> List[str]:
        """Return sorted list of available scene IDs."""
        if not self.scenes_dir.exists():
            return []
        scenes = []
        for p in self.scenes_dir.glob("scene_*.json"):
            scenes.append(p.stem)
        return sorted(scenes)

    def load_scene(self, scene_id: str) -> SceneState:
        """Load SceneState for a given scene_id."""
        scene_file = self.scenes_dir / f"{scene_id}.json"
        safe_path = resolve_safe_path(self.scenes_dir, scene_file)
        if not safe_path.exists():
            raise FileNotFoundError(f"Scene {scene_id} not found at {safe_path}")
        return load_scene_state(safe_path, scene_id=scene_id)

    def load_challenge_metadata(self, scene_id: str) -> Dict[str, Any]:
        """Retrieve or compute SceneChallengeMetadata dict for a given scene."""
        scene_file = self.scenes_dir / f"{scene_id}.json"
        safe_path = resolve_safe_path(self.scenes_dir, scene_file)
        if safe_path.exists():
            try:
                data = json.loads(safe_path.read_text(encoding="utf-8"))
                if "challenge_metadata" in data and isinstance(data["challenge_metadata"], dict):
                    return data["challenge_metadata"]
            except Exception:
                pass

        scene_state = self.load_scene(scene_id)
        return analyze_scene_challenge(scene_state).to_dict()

    def load_manifest(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Load observation manifest for a given scene if rendered."""
        manifest_file = self.rendered_dir / scene_id / "manifest.json"
        if not manifest_file.exists():
            return None
        try:
            safe_path = resolve_safe_path(self.rendered_dir, manifest_file)
            return json.loads(safe_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_views_for_scene(self, scene_id: str) -> List[str]:
        """Return available view IDs for a scene."""
        manifest = self.load_manifest(scene_id)
        if manifest and "observations" in manifest:
            views = [obs["view_id"] for obs in manifest["observations"] if "view_id" in obs]
            if views:
                return sorted(list(dict.fromkeys(views)))

        scene_dir = self.rendered_dir / scene_id
        if scene_dir.exists():
            views = []
            for png in scene_dir.glob("*.png"):
                views.append(png.stem)
            if views:
                return sorted(views)

        # Fallback to standard cardinal views
        return list(VIEW_IDS)

    def get_observation_camera(self, scene_id: str, view_id: str) -> CameraPose:
        """Retrieve CameraPose for an observation view, with fallback to geometry generator."""
        manifest = self.load_manifest(scene_id)
        if manifest and "observations" in manifest:
            for obs in manifest["observations"]:
                if obs.get("view_id") == view_id and "camera" in obs:
                    cam_dict = obs["camera"]
                    return CameraPose(
                        position=tuple(cam_dict["position"]),
                        look_at=tuple(cam_dict["look_at"]),
                        up=tuple(cam_dict.get("up", (0.0, 0.0, 1.0))),
                        fov_deg=float(cam_dict.get("fov_deg", 60.0)),
                    )

        # Deterministic generation fallback
        cardinal_cams = dict(zip(VIEW_IDS, generate_cardinal_views()))
        if view_id in cardinal_cams:
            return cardinal_cams[view_id]

        if view_id.endswith("_jitter"):
            base_view = view_id[:-7]
            if base_view in cardinal_cams:
                scene = self.load_scene(scene_id)
                base_cam = cardinal_cams[base_view]
                v_idx = VIEW_IDS.index(base_view)
                jitter_seed = scene.seed * 1000 + v_idx * 17 + 42
                return jitter_pose(
                    base_cam,
                    seed=jitter_seed,
                    angular_deg=15.0,
                    radius_frac=0.10,
                    target_radius=0.0,
                )

        # Canonical directions (6, 14, 26) from Camera Sampling Foundation
        canonical_map = dict(DIRECTIONS_26)
        if view_id in canonical_map:
            dir_vec = canonical_map[view_id]
            return canonical_camera(
                target=(0.0, 0.0, 0.4),
                radius=6.0,
                direction=dir_vec,
                fov_deg=60.0,
            )

        # Allow index-based canonical view lookup: e.g. "view_15", "canonical_15"
        if view_id.startswith("view_") or view_id.startswith("canonical_"):
            try:
                idx = int(view_id.split("_", 1)[1])
                dirs = canonical_directions(26)
                if 0 <= idx < len(dirs):
                    _, dir_vec = dirs[idx]
                    return canonical_camera(
                        target=(0.0, 0.0, 0.4),
                        radius=6.0,
                        direction=dir_vec,
                        fov_deg=60.0,
                    )
            except (ValueError, IndexError):
                pass

        raise KeyError(f"Unknown view_id {view_id!r} for scene {scene_id!r}")

    def list_view_sets(self, scene_id: str) -> Dict[str, Any]:
        """Return view sets: historical 8, canonical 6, 14, 26, and their camera poses."""
        historical_views = self.list_views_for_scene(scene_id)
        dirs_6 = [vid for vid, _ in canonical_directions(6)]
        dirs_14 = [vid for vid, _ in canonical_directions(14)]
        dirs_26 = [vid for vid, _ in canonical_directions(26)]

        canonical_poses = {}
        for vid, d in canonical_directions(26):
            pose = canonical_camera(target=(0.0, 0.0, 0.4), radius=6.0, direction=d, fov_deg=60.0)
            has_rendered = self.get_image_path(scene_id, vid) is not None
            canonical_poses[vid] = {
                "view_id": vid,
                "direction_id": vid,
                "name": vid.replace("_", " ").title(),
                "direction": list(d),
                "position": [round(x, 4) for x in pose.position],
                "look_at": [round(x, 4) for x in pose.look_at],
                "up": [round(x, 4) for x in pose.up],
                "fov_deg": pose.fov_deg,
                "is_rendered": has_rendered,
            }

        view_sets = [
            {
                "id": "historical_8",
                "name": "Historical 8",
                "description": "Historical 8 rendered viewpoints from E1 benchmark",
                "count": len(historical_views),
                "view_ids": historical_views,
            },
            {
                "id": "canonical_6",
                "name": "Canonical 6",
                "description": "6 Axis-aligned orthogonal views (+X, -X, +Y, -Y, +Z, -Z)",
                "count": len(dirs_6),
                "view_ids": dirs_6,
            },
            {
                "id": "canonical_14",
                "name": "Canonical 14",
                "description": "6 Orthogonal + 8 Corner views",
                "count": len(dirs_14),
                "view_ids": dirs_14,
            },
            {
                "id": "canonical_26",
                "name": "Canonical 26",
                "description": "Complete 26-direction 3D neighborhood camera lattice",
                "count": len(dirs_26),
                "view_ids": dirs_26,
            },
        ]

        return {
            "scene_id": scene_id,
            "view_sets": view_sets,
            "historical_8": historical_views,
            "canonical_6": dirs_6,
            "canonical_14": dirs_14,
            "canonical_26": dirs_26,
            "canonical_poses": canonical_poses,
        }

    def get_image_path(self, scene_id: str, view_id: str) -> Optional[str]:
        """Return repository-relative image path if rendered image exists."""
        candidate = self.rendered_dir / scene_id / f"{view_id}.png"
        if candidate.exists():
            return candidate.relative_to(self.repo_root).as_posix()
        return None

    def _index_dataset_samples(self) -> None:
        """Scan E1 JSONL datasets and index multimodal samples."""
        if self._samples_indexed:
            return

        dataset_files = [
            self.e1_dir / "train_group_a.jsonl",
            self.e1_dir / "train_group_c.jsonl",
            self.e1_dir / "train_group_d.jsonl",
            self.e1_dir / "holdout_s1_cardinal.jsonl",
            self.e1_dir / "holdout_s2_jitter.jsonl",
        ]

        for df in dataset_files:
            if not df.exists():
                continue
            try:
                with open(df, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        sample_id = item.get("id") or item.get("sample_id")
                        scene_id = item.get("scene_id")
                        view_id = item.get("view_id")
                        if sample_id and scene_id and view_id:
                            item["_dataset_source"] = df.name
                            key = (scene_id, view_id)
                            if sample_id not in self._samples_by_id:
                                self._samples_by_id[sample_id] = item
                                self._samples_by_scene_view[key].append(item)
            except Exception:
                continue

        self._samples_indexed = True

    def list_samples(
        self,
        scene_id: Optional[str] = None,
        view_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return list of indexed samples, optionally filtered by scene_id and view_id."""
        self._index_dataset_samples()
        if scene_id and view_id:
            return list(self._samples_by_scene_view.get((scene_id, view_id), []))
        elif scene_id:
            results = []
            for (sc, _), samples in self._samples_by_scene_view.items():
                if sc == scene_id:
                    results.extend(samples)
            return results
        return list(self._samples_by_id.values())

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific sample by sample_id."""
        self._index_dataset_samples()
        return self._samples_by_id.get(sample_id)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Scan outputs/experiments for evaluation or training run directories."""
        runs = []
        if not self.experiments_dir.exists():
            return runs

        candidates = list(self.experiments_dir.glob("g2.0-e3*/baseline_b"))
        candidates.extend(self.experiments_dir.glob("g2.0-e3*/group_*/seed_*"))
        candidates.extend(self.experiments_dir.glob("g2.0-e3-performance/*"))

        for run_path in sorted(candidates):
            if not run_path.is_dir():
                continue
            run_id = run_path.relative_to(self.experiments_dir).as_posix()
            manifest_file = run_path / "run_manifest.json"
            progress_file = run_path / "progress.json"
            config_file = run_path / "run_config.json"

            run_info: Dict[str, Any] = {
                "run_id": run_id,
                "path": str(run_path),
                "has_predictions": (run_path / "predictions").exists(),
                "has_progress": progress_file.exists(),
                "has_manifest": manifest_file.exists(),
            }

            if manifest_file.exists():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    run_info["status"] = data.get("status", "unknown")
                    run_info["group"] = data.get("group")
                    run_info["seed"] = data.get("seed")
                    run_info["training_profile_id"] = data.get("training_profile_id")
                except Exception:
                    pass

            if "training_profile_id" not in run_info and config_file.exists():
                try:
                    cdata = json.loads(config_file.read_text(encoding="utf-8"))
                    tcfg = cdata.get("training_config", {})
                    if tcfg.get("per_device_train_batch_size") == 1 and tcfg.get("gradient_accumulation_steps") == 8:
                        run_info["training_profile_id"] = "reference"
                    elif tcfg.get("per_device_train_batch_size") == 4 and tcfg.get("gradient_accumulation_steps") == 2:
                        run_info["training_profile_id"] = "max_performance"
                    elif tcfg.get("per_device_train_batch_size") == 2 and tcfg.get("gradient_accumulation_steps") == 4:
                        run_info["training_profile_id"] = "balanced"
                except Exception:
                    pass

            runs.append(run_info)
        return runs

    def load_predictions_for_run(self, run_id: str) -> Dict[str, EvaluationPredictionRecord]:
        """Load and index prediction records for a specific run directory."""
        if run_id in self._predictions_by_run:
            return self._predictions_by_run[run_id]

        run_path = self.experiments_dir / run_id
        pred_dir = run_path / "predictions"
        records: Dict[str, EvaluationPredictionRecord] = {}

        if pred_dir.exists():
            for jsonl_path in pred_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            rec = EvaluationPredictionRecord.from_dict(data)
                            records[rec.sample_id] = rec
                except Exception:
                    continue

        self._predictions_by_run[run_id] = records
        return records

    def match_prediction(
        self,
        sample_id: str,
        expected_scene_id: str,
        expected_view_id: str,
        expected_ground_truth: str,
        run_id: Optional[str] = None,
    ) -> PredictionChannel:
        """Find prediction record for sample_id and verify metadata compatibility."""
        if not run_id:
            # Check all available runs if none specified
            for r in self.list_runs():
                if r.get("has_predictions"):
                    preds = self.load_predictions_for_run(r["run_id"])
                    if sample_id in preds:
                        run_id = r["run_id"]
                        break

        if not run_id:
            return PredictionChannel(status="unavailable", provenance="no_prediction_artifact")

        preds = self.load_predictions_for_run(run_id)
        if sample_id not in preds:
            return PredictionChannel(
                status="unavailable",
                provenance=f"not_in_run:{run_id}",
            )

        rec = preds[sample_id]

        # Metadata compatibility check
        conflicts = []
        if rec.scene_id and rec.scene_id != expected_scene_id:
            conflicts.append(f"scene_id mismatch: pred={rec.scene_id!r} vs sample={expected_scene_id!r}")
        if rec.view_id and rec.view_id != expected_view_id:
            conflicts.append(f"view_id mismatch: pred={rec.view_id!r} vs sample={expected_view_id!r}")
        if rec.ground_truth != expected_ground_truth:
            conflicts.append(f"ground_truth mismatch: pred={rec.ground_truth!r} vs sample={expected_ground_truth!r}")

        if conflicts:
            return PredictionChannel(
                status="alignment_error",
                raw_prediction=rec.raw_prediction,
                parsed_prediction=rec.parsed_prediction,
                is_valid_prediction=rec.is_valid_prediction,
                is_correct=False,
                provenance=f"run:{run_id}",
                error_message="; ".join(conflicts),
            )

        return PredictionChannel(
            status="available",
            raw_prediction=rec.raw_prediction,
            parsed_prediction=rec.parsed_prediction,
            is_valid_prediction=rec.is_valid_prediction,
            is_correct=rec.is_correct,
            provenance=f"run:{run_id}:{rec.source}",
        )

    def load_runtime_telemetry(self, run_id: Optional[str] = None) -> RuntimeChannel:
        """Read live or historical runtime telemetry without affecting training."""
        progress_data: Optional[Dict[str, Any]] = None
        freshness: str = "missing"
        source: str = "none"

        run_path = None
        if run_id:
            run_path = self.experiments_dir / run_id
        else:
            # Look for active running soak or formal run
            runs = self.list_runs()
            for r in runs:
                if r.get("has_progress"):
                    run_path = Path(r["path"])
                    run_id = r["run_id"]
                    break

        if run_path and (run_path / "progress.json").exists():
            p_file = run_path / "progress.json"
            source = str(p_file.relative_to(self.repo_root))
            try:
                raw = p_file.read_text(encoding="utf-8")
                progress_data = json.loads(raw)
                # Check freshness
                mtime = p_file.stat().st_mtime
                age_sec = time.time() - mtime
                status = progress_data.get("status", "running")
                if status == "completed":
                    freshness = "completed"
                elif age_sec > 60:
                    freshness = "stale"
                else:
                    freshness = "fresh"
            except Exception:
                freshness = "malformed"

        # Hardware stats via NVML
        telemetry: Dict[str, Any] = {}
        try:
            from spatialforge.experiment.performance import NVMLClient
            nvml = NVMLClient(0)
            stats = nvml.query()
            nvml.close()
            telemetry["nvml"] = stats
        except Exception:
            telemetry["nvml"] = {"error": "NVML query unavailable"}

        # Add profile info if known
        training_profile = None
        if progress_data:
            training_profile = {
                "group": progress_data.get("group"),
                "seed": progress_data.get("seed"),
            }

        return RuntimeChannel(
            training_profile=training_profile,
            progress=progress_data,
            telemetry=telemetry,
            source=source,
            freshness=freshness,
        )
