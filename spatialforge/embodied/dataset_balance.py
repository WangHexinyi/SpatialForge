"""Dataset V2 policy: success-first core set + controlled imbalance handling.

Scientific policy:

* CORE rows come from *successful* teacher episodes only.
* AUX rows are selected *informative* failures (real executed trajectories with
  at least one action), included at a bounded ratio so failures never dominate.
* Uninformative failures (invalid/unreachable task, spawn resample failure, no
  executed steps) are excluded from training entirely.

Balance modes are deterministic and never fabricate samples by duplication:

* ``natural``          -- keep the distribution as recorded.
* ``episode``          -- cap rows per episode (episode-level balance).
* ``action_balanced``  -- deterministically *downsample* majority action
                          classes (no synthetic copies).
* ``action_weighted``  -- keep rows, emit per-class weights for weighted CE.

The "natural vs balanced" comparison is a first-class experiment output.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from spatialforge.embodied.bc_dataset import BC_ACTION_VOCAB

BALANCE_MODES = ("natural", "episode", "action_balanced", "action_weighted")

UNINFORMATIVE_CODES = (
    "invalid_task", "unreachable_task", "spawn_resample_failed",
    "worker_crash", "setup_timeout",
)


def episode_outcomes(result_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map episode_id -> outcome row (from worker result JSONL)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in result_rows:
        ep = r.get("episode_id")
        if ep:
            out[str(ep)] = r
    return out


def classify_rows(
    rows: Sequence[Dict[str, Any]],
    outcomes: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split manifest rows into core (successful) / informative-failure / dropped."""
    core, aux, dropped = [], [], []
    for row in rows:
        ep = str(row.get("episode_id", ""))
        outcome = outcomes.get(ep)
        success = bool(outcome.get("success")) if outcome else None
        code = str((outcome or {}).get("outcome_code") or "")
        row = dict(row)
        row["episode_success"] = success
        if success is True:
            row["row_class"] = "core"
            core.append(row)
        elif outcome is None:
            row["row_class"] = "unknown_episode"
            dropped.append(row)
        elif code in UNINFORMATIVE_CODES:
            row["row_class"] = "uninformative_failure"
            dropped.append(row)
        else:
            row["row_class"] = "aux_failure"
            aux.append(row)
    return {"core": core, "aux": aux, "dropped": dropped}


def select_success_core(
    classified: Dict[str, List[Dict[str, Any]]],
    aux_failure_ratio: float = 0.25,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Core = successful trajectories; AUX = capped informative failures."""
    core = list(classified["core"])
    aux = list(classified["aux"])
    rng = random.Random(seed)
    if aux_failure_ratio <= 0:
        aux_selected: List[Dict[str, Any]] = []
    else:
        cap = int(len(core) * aux_failure_ratio)
        aux_sorted = sorted(aux, key=lambda r: str(r.get("id")))
        rng.shuffle(aux_sorted)
        aux_selected = aux_sorted[:cap] if cap > 0 else []
        aux_selected.sort(key=lambda r: str(r.get("id")))
    selected = core + aux_selected
    selected.sort(key=lambda r: str(r.get("id")))
    info = {
        "core_rows": len(core),
        "aux_failure_rows_available": len(aux),
        "aux_failure_rows_selected": len(aux_selected),
        "aux_failure_ratio_target": aux_failure_ratio,
        "dropped_uninformative_rows": len(classified["dropped"]),
    }
    return selected, info


def _action_of(row: Dict[str, Any]) -> str:
    return str(row.get("answer", ""))


def apply_balance(
    rows: Sequence[Dict[str, Any]],
    mode: str = "natural",
    seed: int = 0,
    episode_cap: int = 40,
    min_class_count: int = 200,
    max_weight: float = 3.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply a deterministic balance policy (no sample duplication)."""
    mode = str(mode)
    if mode not in BALANCE_MODES:
        raise ValueError(f"unknown balance mode {mode!r}; valid: {BALANCE_MODES}")
    rows = [dict(r) for r in rows]
    before = _counts(rows)

    if mode == "natural":
        info = {"mode": mode, "before": before, "after": before}
        return rows, info

    if mode == "episode":
        by_ep: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_ep.setdefault(str(r.get("episode_id")), []).append(r)
        kept: List[Dict[str, Any]] = []
        for ep in sorted(by_ep):
            group = sorted(by_ep[ep], key=lambda r: str(r.get("id")))
            kept.extend(group[:episode_cap])
        kept.sort(key=lambda r: str(r.get("id")))
        after = _counts(kept)
        return kept, {
            "mode": mode, "episode_cap": episode_cap,
            "episodes": len(by_ep), "before": before, "after": after,
        }

    # action_balanced / action_weighted
    by_action: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_action.setdefault(_action_of(r), []).append(r)
    counts = {a: len(v) for a, v in by_action.items()}
    present = [c for c in counts.values() if c >= min_class_count] or list(counts.values())
    target = max(min_class_count, int(sum(present) / max(len(present), 1)))

    if mode == "action_weighted":
        import math

        total = sum(counts.values()) or 1
        n_classes = max(len(counts), 1)
        weights = {
            a: round(min(max_weight, math.sqrt(total / (n_classes * max(c, 1)))), 6)
            for a, c in counts.items()
        }
        for r in rows:
            r["sample_weight"] = weights.get(_action_of(r), 1.0)
        info = {
            "mode": mode, "class_weights": weights,
            "weight_power": 0.5, "max_weight": max_weight,
            "before": before, "after": before, "target_per_class": None,
        }
        return rows, info

    rng = random.Random(seed)
    kept = []
    for a in sorted(by_action):
        group = sorted(by_action[a], key=lambda r: str(r.get("id")))
        if len(group) > target:
            rng.shuffle(group)
            group = sorted(group[:target], key=lambda r: str(r.get("id")))
        kept.extend(group)
    kept.sort(key=lambda r: str(r.get("id")))
    after = _counts(kept)
    return kept, {
        "mode": mode, "target_per_class": target,
        "before": before, "after": after,
    }


def _counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {a: 0 for a in BC_ACTION_VOCAB}
    for r in rows:
        a = _action_of(r)
        counts[a] = counts.get(a, 0) + 1
    return counts


def action_distribution_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = _counts(rows)
    total = sum(counts.values())
    pct = {a: round(c * 100.0 / total, 2) for a, c in counts.items()} if total else {}
    return {"counts": counts, "total": total, "pct": pct}
