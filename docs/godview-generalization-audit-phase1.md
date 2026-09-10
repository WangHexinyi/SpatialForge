# God View — Multi-House Generalization Audit (Phase 1)

Status: **AUDIT COMPLETE (PARTIAL)** — Generalization Gate **NOT CLOSED**.

Branch: `wip/g2.1-research-snapshot-20260909`
HEAD baseline: `66e825b`
Harness: `scripts/audit_house_generalization.py` (read-only; no production changes)
Artifacts (gitignored, generated): `outputs/diagnostics/generalization/audit.json`,
`outputs/diagnostics/generalization/matrix.md`

## 1. Scope

Audit the God View pipeline across a structurally diverse ProcTHOR house set:

```
house JSON -> canonical scene3d -> shared truth -> Unity / 3D semantic / 2D
```

Checks performed (no engine): house/schema load, `build_scene3d`, scene bounds,
wall orientation/length, wall fragmentation, door/window opening truth (host
resolution, real cut, sealed-doorway detection), object mapping, cross-view
coordinate truth (`world_space.js` / `scene2d.js` via node probes), and
FPV/pose/trajectory temporal consistency on mapped episodes.

Check **not** performed: the Unity render pass. It was aborted on a hard time
cutoff. All Unity columns are `NOT_RUN`; no Unity result is inferred or
fabricated.

## 2. Benchmark set (17 houses)

| # | house | complexity | rooms | walls | doors | windows | objects | area m² |
|---|---|---|---|---|---|---|---|---|
| 1 | house_0008 | small | 1 | 8 | 1 | 0 | 8 | 8.2 |
| 2 | house_0024 | small | 1 | 12 | 1 | 0 | 8 | 14.9 |
| 3 | house_0006 | small | 1 | 8 | 1 | 0 | 7 | 36.5 |
| 4 | house_0004 | medium | 4 | 34 | 3 | 2 | 27 | 81.2 |
| 5 | house_0005 | medium | 4 | 32 | 3 | 3 | 19 | 59.6 |
| 6 | house_0026 | medium | 3 | 30 | 3 | 2 | 23 | 49.8 |
| 7 | house_0027 | medium | 4 | 28 | 4 | 2 | 22 | 79.7 |
| 8 | house_0043 | medium | 4 | 34 | 4 | 3 | 26 | 71.9 |
| 9 | house_0009 | large | 10 | 72 | 10 | 8 | 77 | 284.3 |
| 10 | house_0015 | large | 10 | 74 | 10 | 9 | 72 | 269.4 |
| 11 | house_0034 | large | 7 | 48 | 7 | 5 | 52 | 246.3 |
| 12 | val_house_2 | large | 10 | 70 | 10 | 6 | 54 | 189.3 |
| 13 | house_0011 | complex | 4 | 34 | 4 | 3 | 37 | 166.1 |
| 14 | house_0017 | complex | 3 | 26 | 2 | 4 | 33 | 93.8 |
| 15 | house_0018 | complex | 4 | 34 | 4 | 4 | 33 | 134.2 |
| 16 | house_0040 | complex | 5 | 42 | 5 | 5 | 41 | 96.3 |
| 17 | val_house_0 | control | 7 | 46 | 6 | 4 | 48 | 123.2 |

## 3. Structural diversity

- Rooms 1–10 (Σ 82); walls 8–74 (Σ 632); doors 1–10 (Σ 78); windows 0–9 (Σ 60);
  objects 7–77 (Σ 587); floor area 8.2–284.3 m² (Σ 2004.7).
- Wall heights 2.52–5.69 m (high-ceiling houses: 0006, 0018, 0024, 0034).
- `empty` walls 0–2 per house.
- Room floor polygons are non-rectangular (4–11 vertices).
- All houses in the corpus use schema `1.0.0`; no schema variant observed.

## 4. PASS / DEGRADED / FAIL matrix

| house | complexity | 3D | 2D | openings | objects | pose | trajectory | Unity | overall | reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| house_0008 | small | PASS | PASS | PASS | PASS | n/a | n/a | NOT_RUN | PASS | — |
| house_0024 | small | PASS | PASS | PASS | PASS | n/a | n/a | NOT_RUN | PASS | — |
| house_0006 | small | PASS | PASS | PASS | PASS | n/a | n/a | NOT_RUN | PASS | — |
| house_0004 | medium | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0005 | medium | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0026 | medium | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0027 | medium | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| house_0043 | medium | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| house_0009 | large | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0015 | large | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| house_0034 | large | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| val_house_2 | large | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| house_0011 | complex | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0017 | complex | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | LEGACY_SENSOR_ONLY |
| house_0018 | complex | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| house_0040 | complex | DEGRADED | PASS | PASS | PASS | PASS | PASS | NOT_RUN | DEGRADED | SCENE3D_FRAGMENTED, LEGACY_SENSOR_ONLY |
| val_house_0 | control | PASS | PASS | PASS | PASS | PASS | PASS | NOT_RUN | PASS | — |

## 5. Failure reason frequency

| code | count | note |
|---|---|---|
| LEGACY_SENSOR_ONLY | 13/17 | Episode data, not geometry: only legacy `ep-bc-*` episodes map to these houses (`render_quality=None`, no `setup`, no `house_path`). |
| SCENE3D_FRAGMENTED | 6/17 | 2 sliver wall segments (< 2 cm) each. |
| OPENING_ERROR | 0/17 | All openings resolved to a host wall with a real cut; no sealed doorways. |
| OBJECT_MAPPING_ERROR | 0/17 | Object count preserved 1:1 through normalization. |
| BOUNDS_ERROR | 0/17 | All walls/objects inside room-derived bounds; 2D did not fall back to episode-only bounds. |
| WALL_ORIENTATION_ERROR | 0/17 | All walls axis-aligned with finite positive length. |
| VIEW_MIRROR | 0/17 | `worldToThree` involutive; +X→+X, +Z→−Z; yaw→forward checks pass; 2D `+Z` up. |
| UNITY_* / HOUSE_LOAD_FAIL | NOT_RUN | Unity pass aborted on hard time cutoff. |

## 6. Root-cause clusters

1. **Sliver wall fragmentation** (`_wall_segments`, `spatialforge/inspector/scene3d.py`):
   `_WALL_EPS = 1e-4` retains 1–4 cm wall slivers when a door/window is offset a
   few cm from the wall end. Generic (6/17); each affected house shows exactly
   two slivers.
2. **Legacy episode data quality**: 13/17 mapped episodes are pre-fix `ep-bc-*`
   records (`render_quality=None`), so pose/trajectory truth degrades to legacy
   semantics. This is data, not a scene-pipeline regression.
3. **Stale scene snapshots**: all 59/59 persisted `_scenes/*.json` predate the
   door/opening fix (no `segments`, wall-local openings). Mitigated at load by
   `_upgrade_scene_openings` only when `house_path` resolves (verified for
   house_0009 replay: 72/72 walls upgraded, 18/18 openings rehosted).
4. **Object AABB source**: from house JSON alone objects are 0.12 m proxies;
   real AABBs come only from live AI2-THOR metadata / snapshots.
5. **Unity camera framing**: untested — the single largest evidence gap and the
   most likely locus of the reported Human-UAT degradation.

## 7. Data-quality vs implementation

- **Data quality:** `LEGACY_SENSOR_ONLY` (13/17) — legacy episodes, not a code regression.
- **Implementation (generic, minor):** `SCENE3D_FRAGMENTED` (6/17) — epsilon handling in `_wall_segments`.
- **Unresolved / untested:** Unity render and camera framing for all 17 houses (`NOT_RUN`).

## 8. Recommended next implementation order

1. **Unity camera-fit normalization** (highest impact; unblocks the UAT complaint).
   One generic `fitBounds` / `_camera_for` deriving camera height/distance/FOV
   from scene bounds **including height**, with aspect-ratio clamp, replacing
   the `max(6.0, span*0.6)` / `size*0.8` heuristics.
2. **Sliver wall merge** in `_wall_segments`: raise the discard/merge epsilon to
   ~2–3 cm (or merge slivers into the adjacent segment), generically.
3. **Stale snapshot upgrade**: when `house_path` is unresolvable, rebuild the
   scene from the room/wall JSON already embedded in the snapshot instead of
   serving stale openings.
4. **Episode preference / legacy handling**: prefer post-fix episodes per house
   and label legacy ones explicitly (already surfaced by `classify_episode`).

Only item (1) requires the GPU/Unity harness to validate.

## 9. Gate status

- **Generalization Gate: NOT CLOSED.**
- Unity multi-house audit: **NOT_RUN**.
- No Unity result is claimed or fabricated.
- Focused Fixes #1–#5 are preserved; this audit introduced no production change.
