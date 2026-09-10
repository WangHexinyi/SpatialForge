"""Deterministic strict action parser for model-generated text.

Maps free-form VLM output to exactly one legal embodied action. The policy is
explicit and reportable: an empty/unparseable answer yields ``invalid`` with the
raw text retained -- never a silent substitute action.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from spatialforge.embodied.bc_dataset import BC_ACTION_VOCAB

_ACTION_RE = re.compile(
    r"^(?P<a>(?:MoveAhead|RotateLeft|RotateRight|LookUp|LookDown|Crouch|Stand|Done))(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# canonical label map (case-insensitive, tolerant to common trailing punctuation)
_CANON: Dict[str, str] = {}
for a in BC_ACTION_VOCAB:
    _CANON[a.lower()] = a

_NORMALIZE_RE = re.compile(r"[^A-Za-z]")


def normalize_token(text: str) -> str:
    """Remove whitespace/punctuation; used only for exact matching."""
    return _NORMALIZE_RE.sub("", text)


def parse_action_output(raw: str) -> Dict[str, Any]:
    """Parse raw model text into a decision dict.

    Returns ``{"action": <label|None>, "valid": bool, "raw": raw, "reason": str}``.
    Strict by design: only (a) a canonical action on the first line, or (b)
    output whose *whole normalized text* equals exactly one canonical action
    label, is accepted. Any surrounding prose makes the output invalid -- it is
    reported, never silently substituted.
    """
    raw = (raw or "").strip()
    if not raw:
        return {"action": None, "valid": False, "raw": raw, "reason": "empty_output"}

    stripped = raw.strip().strip("`\"").strip()
    first_line = stripped.splitlines()[0].strip()

    # exact canonical label first
    m = _ACTION_RE.match(first_line)
    if m:
        act = _CANON[m.group("a").lower()]
        rest = m.group("rest").strip()
        if rest and not re.fullmatch(r"[.,;:!?\s\)\]\}'\"]*", rest):
            return {
                "action": None, "valid": False, "raw": raw,
                "reason": f"trailing_text:{rest[:40]!r}",
            }
        return {"action": act, "valid": True, "raw": raw, "reason": "exact"}

    # whole-output canonical equality (tolerant only to whitespace/punctuation)
    flat = normalize_token(stripped).lower()
    found = [a for a in BC_ACTION_VOCAB if normalize_token(a).lower() == flat]
    if len(found) == 1:
        return {"action": found[0], "valid": True, "raw": raw, "reason": "canonical_flat"}
    if len(found) > 1:
        return {"action": None, "valid": False, "raw": raw, "reason": "ambiguous_multiple"}
    return {"action": None, "valid": False, "raw": raw, "reason": "no_action_token"}
