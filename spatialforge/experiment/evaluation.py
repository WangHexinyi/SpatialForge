"""Evaluation hardening, strict answer parsers, historical VSR taxonomy, and metric aggregation.

G2.0-E3.1 Evaluator Hardening:
- Strict directional parser with word boundary checks, uncertainty/negation detection,
  and multi-label conflict rejection.
- Strict yes/no parser for VSR evaluation preventing substring collisions.
- Preserved complete historical VSR relation taxonomy with disjoint direct vs broader orientation.
- Deterministic per-sample evaluation record schema preserving identity for bootstrap/McNemar.
- Reusable metric aggregation reporting overall and per-dimension accuracy with explicit invalid accounting.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# Supported canonical directional answer vocabulary for G2.0-E
DIRECTIONAL_VOCABULARY: Tuple[str, ...] = (
    "left",
    "right",
    "above",
    "below",
    "front",
    "behind",
    "nearer",
    "farther",
)

# Uncertainty and hedging indicators for directional parsing
DIRECTIONAL_UNCERTAINTY_PATTERNS = (
    r"\b(maybe|may be|perhaps|possibly|probable|probably|likely|unclear|unsure|uncertain|unknown|undetermined|ambiguous|guess)\b",
    r"\b(cannot tell|can't tell|cannot say|can't say|hard to tell|hard to say|hard to determine|not sure|not certain|cannot determine|can't determine)\b",
    r"\bno (idea|clue|way to know|way to tell)\b",
    r"\b(might|could)\b",
    r"\bwhether\b",
)

# Negation and disjunction patterns for directional parsing
DIRECTIONAL_NEGATION_PATTERNS = (
    r"\b(not|n't|never|neither|nor|no|none)\b",
)
DIRECTIONAL_DISJUNCTION_PATTERNS = (
    r"\b(or|either)\b",
)


def parse_directional_answer(text: str) -> Optional[str]:
    """Strictly parse model text generation into one canonical directional answer.

    Semantics:
    - Accepts unambiguous single directional answers ("left", "Left.", full sentence with exactly one direction).
    - Rejects explicit negation ("not left", "is not left").
    - Rejects uncertainty/hedging ("I cannot tell whether it is left", "maybe behind", "unclear").
    - Rejects disjunctions and conflicts ("left or right", "either left or right", "front or behind").
    - Uses exact word boundaries (avoids naive substring collisions like 'upright', 'bright', 'confront').
    - Returns exactly one canonical label or None (fail-safe).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip().lower()
    if not cleaned:
        return None

    # Detect disjunction / conflict markers
    for pat in DIRECTIONAL_DISJUNCTION_PATTERNS:
        if re.search(pat, cleaned):
            return None

    # Detect explicit negation
    for pat in DIRECTIONAL_NEGATION_PATTERNS:
        if re.search(pat, cleaned):
            return None

    # Detect uncertainty or hedging
    for pat in DIRECTIONAL_UNCERTAINTY_PATTERNS:
        if re.search(pat, cleaned):
            return None

    # Extract matching directional terms using word boundaries
    matched_terms: List[str] = []
    for term in DIRECTIONAL_VOCABULARY:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, cleaned):
            matched_terms.append(term)

    # Exactly one unambiguous directional term must be present
    if len(matched_terms) == 1:
        return matched_terms[0]

    return None


# Uncertainty markers for yes/no parsing
YESNO_UNCERTAINTY_PATTERNS = (
    r"\b(not sure|unsure|uncertain|unclear|cannot determine|can't determine|cannot tell|can't tell|cannot say|can't say|hard to tell|hard to say|hard to determine|maybe|may be|perhaps|possibly|potential|undetermined|unknown|ambiguous|guess)\b",
    r"\b(probably|probable|likely)\b",
    r"\bno (idea|clue|way to know|way to tell|opinion)\b",
    r"\b(might|could)\b",
    r"\bwhether\b",
)


def parse_yesno_answer(text: str) -> Optional[str]:
    """Strictly parse model text generation for VSR evaluation (yes/no).

    Semantics:
    - Accepts unambiguous "yes", "Yes.", "Yes, true.", "no", "No.", "No, false.", "No, it is not."
    - Rejects uncertainty ("I am not sure", "Cannot determine", "maybe").
    - Rejects disjunction/both ("yes or no", "both yes and no").
    - Uses standalone token/word boundaries to prevent substring collisions
      with words like "not", "cannot", "unknown", "notice", "nothing".
    - Returns "yes", "no", or None (fail-safe).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip().lower()
    if not cleaned:
        return None

    # Reject uncertainty / hedging phrases
    for pat in YESNO_UNCERTAINTY_PATTERNS:
        if re.search(pat, cleaned):
            return None

    # Reject explicit disjunction or conjunction of alternatives
    if re.search(r"\b(or|either|both)\b", cleaned):
        return None

    has_yes = bool(re.search(r"\byes\b", cleaned))
    has_no = bool(re.search(r"\bno\b", cleaned))

    # Reject conflicting responses containing both yes and no
    if has_yes and has_no:
        return None

    if has_yes:
        # Check if yes is negated (e.g. "not yes", "cannot say yes")
        if re.search(r"\b(not|n't|never|cannot|can't)\s+(say\s+)?yes\b", cleaned):
            return None
        # Check for contradictory "false" without negation
        if re.search(r"\bfalse\b", cleaned) and not re.search(r"\b(not|n't)\s+false\b", cleaned):
            return None
        return "yes"

    if has_no:
        # Check if no is negated (e.g. "not no")
        if re.search(r"\b(not|n't|never)\s+no\b", cleaned):
            return None
        # Check for contradictory "true" without negation (e.g. "No, true" vs "No, not true")
        if re.search(r"\btrue\b", cleaned) and not re.search(r"\b(not|n't)\s+true\b", cleaned):
            return None
        return "no"

    return None


# =====================================================================
# Historical VSR Taxonomy (Recovered from Git history: 15ad78f / 019a60d)
# =====================================================================

VSR_DIRECT_DIMS: frozenset = frozenset({"left_right", "front_back"})
VSR_BROADER_DIMS: frozenset = frozenset({"orientation"})

VSR_ALL_DIMS: Tuple[str, ...] = (
    "left_right",
    "front_back",
    "vertical",
    "near_far",
    "topology",
    "contact",
    "orientation",
    "part_whole",
)

# Exact historical relation-to-dimension mapping recovered from historical vsr_adapter.py
VSR_RELATION_TO_DIM: Dict[str, str] = {
    # Direct horizontal
    "left of": "left_right",
    "right of": "left_right",
    "at the left side of": "left_right",
    "at the right side of": "left_right",
    "at the side of": "left_right",
    # Direct depth
    "in front of": "front_back",
    "behind": "front_back",
    "ahead of": "front_back",
    "at the back of": "front_back",
    # Vertical
    "above": "vertical",
    "below": "vertical",
    "on top of": "vertical",
    "under": "vertical",
    "beneath": "vertical",
    "over": "vertical",
    "down from": "vertical",
    # Near / Far
    "near": "near_far",
    "close to": "near_far",
    "next to": "near_far",
    "beside": "near_far",
    "adjacent to": "near_far",
    "alongside": "near_far",
    "by": "near_far",
    "far away from": "near_far",
    "far from": "near_far",
    "away from": "near_far",
    "beyond": "near_far",
    # Topology
    "in": "topology",
    "inside": "topology",
    "contains": "topology",
    "within": "topology",
    "enclosed by": "topology",
    "outside": "topology",
    "out of": "topology",
    "into": "topology",
    "surrounding": "topology",
    "around": "topology",
    "in the middle of": "topology",
    "at the edge of": "topology",
    "through": "topology",
    # Contact
    "on": "contact",
    "touching": "contact",
    "attached to": "contact",
    "connected to": "contact",
    "against": "contact",
    "attached": "contact",
    "detached from": "contact",
    "off": "contact",
    # Broader historical orientation (NOT direct left/right/front/back)
    "facing": "orientation",
    "facing away from": "orientation",
    "toward": "orientation",
    "opposite to": "orientation",
    "parallel to": "orientation",
    "perpendicular to": "orientation",
    "across from": "orientation",
    "across": "orientation",
    "along": "orientation",
    # Part / Whole
    "part of": "part_whole",
    "has as a part": "part_whole",
    "consists of": "part_whole",
}


def classify_vsr_relation(relation: str) -> str:
    """Classify a VSR relation into dimension; distinguishes direct vs broader orientation.

    Unmapped relations return 'other'.
    """
    rel = relation.strip().lower()
    return VSR_RELATION_TO_DIM.get(rel, "other")


# =====================================================================
# Deterministic Per-Sample Evaluation Record Schema
# =====================================================================

@dataclass(frozen=True)
class EvaluationPredictionRecord:
    """Deterministic per-example evaluation prediction record.

    Retains complete sample identity and raw/parsed predictions for:
    - paired bootstrap resampling
    - McNemar's test
    - discordant-pair analysis
    """

    sample_id: str
    source: str  # "synthetic_s1", "synthetic_s2", "vsr"
    scene_id: Optional[str]  # e.g. "scene_080" for synthetic, None for VSR
    group: str  # "B", "A", "C", "D"
    seed: Optional[int]  # 42, 123, 456, or None for zero-shot baseline B
    family: str  # dimension name (e.g. "horizontal", "vertical", "left_right", etc.)
    ground_truth: str
    raw_prediction: str
    parsed_prediction: Optional[str]
    is_valid_prediction: bool
    is_correct: bool
    model_id: str
    adapter_path: Optional[str] = None  # None for baseline B
    view_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic dictionary representation."""
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "source": self.source,
            "scene_id": self.scene_id,
            "group": self.group,
            "seed": self.seed,
            "family": self.family,
            "ground_truth": self.ground_truth,
            "raw_prediction": self.raw_prediction,
            "parsed_prediction": self.parsed_prediction,
            "is_valid_prediction": self.is_valid_prediction,
            "is_correct": self.is_correct,
            "model_id": self.model_id,
            "adapter_path": self.adapter_path,
        }
        if self.view_id is not None:
            d["view_id"] = self.view_id
        if self.meta:
            d["meta"] = self.meta
        return d

    def to_json(self) -> str:
        """Deterministic JSON string serialization with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationPredictionRecord":
        """Reconstruct record from dictionary."""
        return cls(
            sample_id=str(data["sample_id"]),
            source=str(data["source"]),
            scene_id=str(data["scene_id"]) if data.get("scene_id") is not None else None,
            group=str(data["group"]),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            family=str(data["family"]),
            ground_truth=str(data["ground_truth"]),
            raw_prediction=str(data["raw_prediction"]),
            parsed_prediction=str(data["parsed_prediction"]) if data.get("parsed_prediction") is not None else None,
            is_valid_prediction=bool(data["is_valid_prediction"]),
            is_correct=bool(data["is_correct"]),
            model_id=str(data["model_id"]),
            adapter_path=str(data["adapter_path"]) if data.get("adapter_path") is not None else None,
            view_id=str(data["view_id"]) if data.get("view_id") is not None else None,
            meta=data.get("meta"),
        )


# =====================================================================
# Reusable Deterministic Metric Aggregation
# =====================================================================

def compute_slice_metrics(records: Sequence[EvaluationPredictionRecord]) -> Dict[str, Any]:
    """Compute deterministic aggregation for a single record slice.

    Policy:
    - Primary accuracy = correct / total (invalid parsed predictions count as incorrect).
    - Invalid predictions and invalid rate are explicitly tracked and reported separately.
    - valid_accuracy = correct / valid_predictions is reported as a diagnostic.
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "valid_predictions": 0,
            "invalid_predictions": 0,
            "invalid_rate": 0.0,
            "correct": 0,
            "accuracy": 0.0,
            "valid_accuracy": 0.0,
        }

    valid = sum(1 for r in records if r.is_valid_prediction)
    invalid = total - valid
    correct = sum(1 for r in records if r.is_correct)

    return {
        "total": total,
        "valid_predictions": valid,
        "invalid_predictions": invalid,
        "invalid_rate": round(invalid / total, 6),
        "correct": correct,
        "accuracy": round(correct / total, 6),
        "valid_accuracy": round(correct / valid, 6) if valid > 0 else 0.0,
    }


def aggregate_metrics(records: Sequence[EvaluationPredictionRecord]) -> Dict[str, Any]:
    """Aggregate evaluation records into overall and per-dimension metrics.

    Returns deterministic dictionary with:
    - 'overall': slice metrics over all records
    - 'by_dimension': slice metrics partitioned by record.family
    """
    overall = compute_slice_metrics(records)
    by_dim_records: Dict[str, List[EvaluationPredictionRecord]] = defaultdict(list)
    for r in records:
        by_dim_records[r.family].append(r)

    by_dimension = {
        dim: compute_slice_metrics(recs)
        for dim, recs in sorted(by_dim_records.items())
    }

    return {
        "overall": overall,
        "by_dimension": by_dimension,
    }
