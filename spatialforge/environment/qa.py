"""Deterministic multi-view QA curriculum for SpatialForge G2.0-D.

Pure CPU Python. No bpy import and no third-party dependencies.

Scope
-----
This module converts ``SpatialTruthRecord`` values (G2.0-C) into
deterministic natural-language QA training samples. It is deliberately NOT a
rendering, model-training, LLM, projection, FOV, occlusion, visibility,
bbox, or embodied-Agent layer.

Scientific purpose
------------------
Same ``SceneState`` + same objects + different ``CameraPose`` must change the
QA answers of viewpoint-dependent relations correctly, while
viewpoint-invariant controls stay invariant.

Truth is consumed, never recomputed
-----------------------------------
Every spatial relation read here comes verbatim from G2.0-C
``SpatialTruthRecord`` fields (``PairTruth.left_right`` / ``above_below`` /
``front_behind`` / ``near_far`` and ``PairTruth.metric_distance``). This layer
never re-derives camera geometry. The only geometry call in the whole module
is ``compute_spatial_truth`` inside ``generate_multiview_curriculum``, which is
the documented way to obtain per-view truth from a ``SceneState``.

Front-halfspace eligibility (NOT visibility)
--------------------------------------------
G2.0-C deliberately retains behind-camera objects. Normal visual QA is only
generated for a pair when BOTH objects have ``in_front_of_camera == True`` in
every relevant view. This is only a front-halfspace filter.

    eligible_for_visual_qa != guaranteed_visible

No FOV / occlusion / projection logic belongs here.

Neutral policy
--------------
The neutral truth states (``aligned`` / ``same_depth`` / ``equidistant``) are
never collapsed into an arbitrary binary answer. By default they are excluded
from single-view training QA. With ``include_neutral=True`` they are emitted
verbatim with their exact G2.0-C labels. Paired-view transformation QA always
compares only directional (non-neutral) states.

Identity
--------
Object references always use ``object_index`` plus the (possibly duplicated)
name, e.g. ``object 0 ("red_cube")``. ``SceneObject.name`` uniqueness is never
assumed. Sample ids and ordering depend only on indices / view ids / families,
never on ``name`` and never on ``hash()``.

Determinism / ordering
----------------------
Single-view samples iterate ``SpatialTruthRecord.pairs`` (index-combination
order) and, within a pair, the fixed family order ``horizontal``, ``vertical``,
``depth``, ``near_far``.

``generate_multiview_curriculum`` emits, in one flat deterministic tuple:
  1. single-view blocks in input ``CurriculumView`` order (each block in pair
     order then family order above);
  2. paired-view blocks for deterministic view combinations ``(i, j)`` with
     ``i < j`` in input order. Each paired block emits, per eligible pair and
     in the fixed family order, samples only when the relation *changes*
     (unchanged relations are dropped unless ``include_unchanged=True``), and
     then that pair's world-invariant metric-distance control. So invariant
     controls immediately follow the transformation samples of the same pair.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from spatialforge.environment.camera import CameraPose
from spatialforge.environment.scene import SceneState
from spatialforge.environment.relation import (
    REL_ABOVE,
    REL_ALIGNED,
    REL_BEHIND,
    REL_BELOW,
    REL_EQUIDISTANT,
    REL_FARTHER,
    REL_FRONT,
    REL_LEFT,
    REL_NEARER,
    REL_RIGHT,
    REL_SAME_DEPTH,
    SpatialTruthRecord,
    compute_spatial_truth,
)

# Fixed family traversal order shared by single-view and paired-view QA.
_FAMILY_ORDER = (
    ("horizontal", "left_right"),
    ("vertical", "above_below"),
    ("depth", "front_behind"),
    ("near_far", "near_far"),
)

# Relative-phrase used inside single-view question text, per family.
_SINGLE_OPTION_PHRASE = {
    "horizontal": "left of or right of",
    "vertical": "above or below",
    "depth": "in front of or behind",
    "near_far": "nearer to or farther from the camera than",
}

# Truth token -> answer word for each family. Neutral tokens map to their
# exact G2.0-C labels.
_ANSWER_BY_TOKEN = {
    "left_right": {
        REL_LEFT: "left",
        REL_RIGHT: "right",
        REL_ALIGNED: REL_ALIGNED,
    },
    "above_below": {
        REL_ABOVE: "above",
        REL_BELOW: "below",
        REL_ALIGNED: REL_ALIGNED,
    },
    "front_behind": {
        REL_FRONT: "front",
        REL_BEHIND: "behind",
        REL_SAME_DEPTH: REL_SAME_DEPTH,
    },
    "near_far": {
        REL_NEARER: "nearer",
        REL_FARTHER: "farther",
        REL_EQUIDISTANT: REL_EQUIDISTANT,
    },
}

_NEUTRAL_TOKENS = frozenset({REL_ALIGNED, REL_SAME_DEPTH, REL_EQUIDISTANT})

# Exact inverse transitions that receive the ``symmetry_flip`` curriculum tag.
_INVERSE_TRANSITIONS = frozenset(
    {
        (REL_LEFT, REL_RIGHT),
        (REL_RIGHT, REL_LEFT),
        (REL_ABOVE, REL_BELOW),
        (REL_BELOW, REL_ABOVE),
        (REL_FRONT, REL_BEHIND),
        (REL_BEHIND, REL_FRONT),
        (REL_NEARER, REL_FARTHER),
        (REL_FARTHER, REL_NEARER),
    }
)

# Strict tolerance for the world-invariant metric-distance control.
_INVARIANT_TOL = 1e-9

# Family token used for world-invariant metric-distance controls.
FAMILY_METRIC_INVARIANCE = "metric_invariance"

# Curriculum tag for world-invariant metric-distance controls.
TAG_WORLD_INVARIANT = "world_invariant"

# Curriculum tag for exact-inverse paired transitions.
TAG_SYMMETRY_FLIP = "symmetry_flip"

# Curriculum tag for size-vs-camera-distance conflicts.
TAG_SIZE_DISTANCE_CONFLICT = "size_distance_conflict"

# Curriculum tags attached to paired controls whose relations never change.
TAG_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class CurriculumView:
    """A named camera plus caller-provided tags inside a curriculum."""

    view_id: str
    camera: CameraPose
    tags: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError("view_id must be a non-empty string")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class QASample:
    """One deterministic natural-language spatial QA sample."""

    sample_id: str
    scene_id: str
    mode: str
    family: str
    question: str
    answer: str
    object_indices: Tuple[int, ...]
    view_ids: Tuple[str, ...]
    tags: Tuple[str, ...]
    is_view_dependent: bool

    def to_dict(self) -> dict:
        """Deterministic JSON-compatible serialization."""
        return {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "mode": self.mode,
            "family": self.family,
            "question": self.question,
            "answer": self.answer,
            "object_indices": list(self.object_indices),
            "view_ids": list(self.view_ids),
            "tags": list(self.tags),
            "is_view_dependent": self.is_view_dependent,
        }


def _merge_tags(*groups: Tuple[str, ...]) -> Tuple[str, ...]:
    """Deterministically concatenate tag groups, preserving first occurrence."""
    out = []
    seen = set()
    for group in groups:
        for tag in group:
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return tuple(out)


def _objects_by_index(record: SpatialTruthRecord) -> dict:
    return {obj.object_index: obj for obj in record.objects}


def _pairs_by_index(record: SpatialTruthRecord) -> dict:
    return {(p.object_index_a, p.object_index_b): p for p in record.pairs}


def _object_ref(index: int, name: str) -> str:
    """Unambiguous, index-based object reference used in question text."""
    return f'object {index} ("{name}")'


def _front_eligible(record: SpatialTruthRecord, index_a: int, index_b: int) -> bool:
    """Both objects lie in the front halfspace (depth > eps) of this view."""
    objects = _objects_by_index(record)
    return objects[index_a].in_front_of_camera and objects[index_b].in_front_of_camera


def _size_distance_conflict(record: SpatialTruthRecord, pair, index_a, index_b) -> bool:
    """True when physical size ordering opposes camera-distance ordering.

    Uses ``ObjectTruth.size`` metadata only. ``near_far`` is the G2.0-C
    ordering of A vs B by Euclidean camera distance, so:
      A larger but farther  -> REL_FARTHER with size_a > size_b
      A smaller but nearer  -> REL_NEARER  with size_a < size_b
    """
    objects = _objects_by_index(record)
    size_a = objects[index_a].size
    size_b = objects[index_b].size
    if pair.near_far == REL_FARTHER:
        return size_a > size_b
    if pair.near_far == REL_NEARER:
        return size_a < size_b
    return False


def _single_question(family: str, ref_a: str, ref_b: str) -> str:
    return f"From this view, is {ref_a} {_SINGLE_OPTION_PHRASE[family]} {ref_b}?"


def _paired_question(family: str, ref_a: str, ref_b: str, view_id_a: str, view_id_b: str) -> str:
    return (
        f"How does the {family} relation of {ref_a} to {ref_b} change "
        f'from view "{view_id_a}" to view "{view_id_b}"?'
    )


def _invariant_question(ref_a: str, ref_b: str, view_id_a: str, view_id_b: str) -> str:
    return (
        f"Does the metric distance between {ref_a} and {ref_b} change "
        f'between view "{view_id_a}" and view "{view_id_b}"?'
    )


def _check_nonempty_view_id(view_id: str) -> None:
    if not isinstance(view_id, str) or not view_id:
        raise ValueError("view_id must be a non-empty string")


def _validate_same_scene(truth_a: SpatialTruthRecord, truth_b: SpatialTruthRecord) -> None:
    """Reject paired inputs that cannot be the same SceneState under two cameras.

    Identity is authoritative by ``object_index`` (G2.0-C), so world facts are
    compared per index. Duplicate names are safe because identity never relies
    on names.
    """
    if truth_a.scene_id != truth_b.scene_id:
        raise ValueError(
            f"paired QA requires matching scene_id: {truth_a.scene_id!r} vs "
            f"{truth_b.scene_id!r}"
        )
    if len(truth_a.objects) != len(truth_b.objects):
        raise ValueError(
            f"paired QA requires matching object count: {len(truth_a.objects)} "
            f"vs {len(truth_b.objects)}"
        )

    objects_a = _objects_by_index(truth_a)
    objects_b = _objects_by_index(truth_b)

    if list(objects_a) != list(objects_b):
        raise ValueError("paired QA requires identical object indices across records")

    for index in objects_a:
        obj_a = objects_a[index]
        obj_b = objects_b[index]
        for attr in ("name", "shape", "color", "size", "world_location"):
            value_a = getattr(obj_a, attr)
            value_b = getattr(obj_b, attr)
            if value_a != value_b:
                raise ValueError(
                    f"paired QA requires matching stable world fact "
                    f"{attr!r} at object_index {index}: {value_a!r} vs {value_b!r}"
                )


def generate_single_view_qa(
    truth: SpatialTruthRecord,
    view_id: str,
    *,
    view_tags: Tuple[str, ...] = (),
    include_neutral: bool = False,
) -> Tuple[QASample, ...]:
    """Deterministic pairwise QA for one view from one SpatialTruthRecord.

    Pairs iterate in ``truth.pairs`` (index-combination) order; within a pair
    families follow the fixed order horizontal, vertical, depth, near_far.
    Neutral states are excluded by default; ``include_neutral=True`` emits
    them with their exact G2.0-C labels. Pairs containing a behind-camera
    object are always excluded from normal visual QA.
    """
    _check_nonempty_view_id(view_id)
    samples = []
    for pair in truth.pairs:
        index_a = pair.object_index_a
        index_b = pair.object_index_b
        if not _front_eligible(truth, index_a, index_b):
            continue

        derived_tags = ()
        if _size_distance_conflict(truth, pair, index_a, index_b):
            derived_tags += (TAG_SIZE_DISTANCE_CONFLICT,)
        tags = _merge_tags(tuple(view_tags), derived_tags)

        ref_a = _object_ref(index_a, pair.name_a)
        ref_b = _object_ref(index_b, pair.name_b)

        for family, attr in _FAMILY_ORDER:
            token = getattr(pair, attr)
            if token in _NEUTRAL_TOKENS and not include_neutral:
                continue
            answer = _ANSWER_BY_TOKEN[attr][token]
            sample_id = f"{truth.scene_id}:single:{view_id}:{index_a}-{index_b}:{family}"
            samples.append(
                QASample(
                    sample_id=sample_id,
                    scene_id=truth.scene_id,
                    mode="single",
                    family=family,
                    question=_single_question(family, ref_a, ref_b),
                    answer=answer,
                    object_indices=(index_a, index_b),
                    view_ids=(view_id,),
                    tags=tags,
                    is_view_dependent=True,
                )
            )
    return tuple(samples)


def generate_paired_view_qa(
    truth_a: SpatialTruthRecord,
    view_id_a: str,
    truth_b: SpatialTruthRecord,
    view_id_b: str,
    *,
    tags: Tuple[str, ...] = (),
    include_unchanged: bool = False,
) -> Tuple[QASample, ...]:
    """Paired-view transformation QA plus world-invariant controls.

    Both records must describe the same SceneState (same scene_id, object
    count and per-index stable world facts); otherwise ``ValueError``.

    Per eligible pair (both objects in the front halfspace of BOTH views) and
    per fixed family, a paired sample is emitted only when the directional
    relation changes (or when ``include_unchanged=True`` for unchanged
    directional relations). Exact-inverse transitions receive
    ``symmetry_flip``. Each eligible pair is then closed by one
    world-invariant metric-distance control ("unchanged"). If the two records'
    object-to-object ``metric_distance`` disagree beyond a strict tolerance, a
    broken world invariant is reported as ``ValueError``.
    """
    _check_nonempty_view_id(view_id_a)
    _check_nonempty_view_id(view_id_b)
    _validate_same_scene(truth_a, truth_b)

    objects_b = _objects_by_index(truth_b)
    pairs_b = _pairs_by_index(truth_b)
    scene_id = truth_a.scene_id
    base_tags = tuple(tags)

    samples = []
    for pair_a in truth_a.pairs:
        index_a = pair_a.object_index_a
        index_b = pair_a.object_index_b
        if not _front_eligible(truth_a, index_a, index_b):
            continue
        if not _front_eligible(truth_b, index_a, index_b):
            continue
        pair_b = pairs_b[(index_a, index_b)]

        ref_a = _object_ref(index_a, pair_a.name_a)
        ref_b = _object_ref(index_b, pair_a.name_b)

        for family, attr in _FAMILY_ORDER:
            token_a = getattr(pair_a, attr)
            token_b = getattr(pair_b, attr)
            if token_a in _NEUTRAL_TOKENS or token_b in _NEUTRAL_TOKENS:
                continue
            if token_a == token_b and not include_unchanged:
                continue

            answer_a = _ANSWER_BY_TOKEN[attr][token_a]
            answer_b = _ANSWER_BY_TOKEN[attr][token_b]
            derived_tags = ()
            if (token_a, token_b) in _INVERSE_TRANSITIONS:
                derived_tags += (TAG_SYMMETRY_FLIP,)
            if token_a == token_b:
                derived_tags += (TAG_UNCHANGED,)
            sample_tags = _merge_tags(base_tags, derived_tags)

            sample_id = (
                f"{scene_id}:paired:{view_id_a}->{view_id_b}:"
                f"{index_a}-{index_b}:{family}"
            )
            samples.append(
                QASample(
                    sample_id=sample_id,
                    scene_id=scene_id,
                    mode="paired",
                    family=family,
                    question=_paired_question(
                        family, ref_a, ref_b, view_id_a, view_id_b
                    ),
                    answer=f"{answer_a} -> {answer_b}",
                    object_indices=(index_a, index_b),
                    view_ids=(view_id_a, view_id_b),
                    tags=sample_tags,
                    is_view_dependent=True,
                )
            )

        # World-invariant metric-distance control closes this pair.
        if not math.isclose(
            pair_a.metric_distance,
            pair_b.metric_distance,
            rel_tol=_INVARIANT_TOL,
            abs_tol=_INVARIANT_TOL,
        ):
            raise ValueError(
                "broken world invariant: object-to-object metric_distance "
                f"for pair ({index_a}, {index_b}) differs between views: "
                f"{pair_a.metric_distance!r} vs {pair_b.metric_distance!r}"
            )
        sample_id = (
            f"{scene_id}:paired:{view_id_a}->{view_id_b}:"
            f"{index_a}-{index_b}:{FAMILY_METRIC_INVARIANCE}"
        )
        sample_tags = _merge_tags(base_tags, (TAG_WORLD_INVARIANT,))
        samples.append(
            QASample(
                sample_id=sample_id,
                scene_id=scene_id,
                mode="paired",
                family=FAMILY_METRIC_INVARIANCE,
                question=_invariant_question(ref_a, ref_b, view_id_a, view_id_b),
                answer="unchanged",
                object_indices=(index_a, index_b),
                view_ids=(view_id_a, view_id_b),
                tags=sample_tags,
                is_view_dependent=False,
            )
        )
    return tuple(samples)


def generate_multiview_curriculum(
    state: SceneState,
    views: Tuple[CurriculumView, ...],
    *,
    include_neutral: bool = False,
) -> Tuple[QASample, ...]:
    """Deterministic flat curriculum for one scene over many views.

    Emits in order:
      1. single-view QA blocks in input view order;
      2. paired-view blocks for each deterministic combination ``(i, j)`` with
         ``i < j`` in input order (transformation samples plus invariant
         controls, see ``generate_paired_view_qa``).

    ``view_id`` values must be unique and non-empty. Truth is computed once per
    view via ``compute_spatial_truth``; no QA-layer geometry exists.
    """
    seen = set()
    records = []
    for view in views:
        if view.view_id in seen:
            raise ValueError(f"duplicate view_id in multiview input: {view.view_id!r}")
        seen.add(view.view_id)
        records.append((view, compute_spatial_truth(state, view.camera)))

    samples = []
    for view, record in records:
        samples.extend(
            generate_single_view_qa(
                record,
                view.view_id,
                view_tags=view.tags,
                include_neutral=include_neutral,
            )
        )

    for i in range(len(records)):
        view_a, record_a = records[i]
        for j in range(i + 1, len(records)):
            view_b, record_b = records[j]
            combo_tags = _merge_tags(tuple(view_a.tags), tuple(view_b.tags))
            samples.extend(
                generate_paired_view_qa(
                    record_a,
                    view_a.view_id,
                    record_b,
                    view_b.view_id,
                    tags=combo_tags,
                )
            )
    return tuple(samples)
