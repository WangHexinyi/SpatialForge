// Cross-language consistency probe for the canonical 2D projection.
//
// Reads a canonical scene payload JSON path from argv[2], runs the exact
// browser module `spatialforge/inspector/embodied_web/scene2d.js` under node,
// and prints the resulting 2D model (bounds, transform, projected agent/targets
// and independently recomputed projections) as JSON for the pytest harness.
//
// Usage: node scene2d_probe.mjs <payload.json> [extra_points.json]
//
// `extra_points.json` is optional: {"points": [[x,y,z], ...]} and the probe
// then also emits `projected_points` so tests can compare arbitrary canonical
// world points (walls / openings / objects) against the 3D semantic space.

import { readFileSync } from "node:fs";
import {
  buildScene2d,
  projectScenePoint,
} from "../../spatialforge/inspector/embodied_web/scene2d.js";

const payloadPath = process.argv[2];
if (!payloadPath) {
  process.stderr.write("usage: node scene2d_probe.mjs <payload.json>\n");
  process.exit(2);
}

const payload = JSON.parse(readFileSync(payloadPath, "utf8"));
const viewport = { width: 800, height: 600 };
const model = buildScene2d(payload, viewport);

let extraPoints = [];
if (process.argv[3]) {
  extraPoints = JSON.parse(readFileSync(process.argv[3], "utf8")).points || [];
}

const out = {
  schema: model.schema,
  bounds: model.bounds,
  bounds_source: model.bounds_source,
  transform: model.transform,
  counts: model.counts,
  source: model.source,
  projected: {
    agent_world: model.agent ? model.agent.world : null,
    agent: model.agent ? model.agent.projected : null,
    targets: model.targets.map((t) => ({ world: t.world, projected: t.projected })),
    spawn: model.spawn,
  },
  recomputed: {
    agent: model.agent ? projectScenePoint(model, model.agent.world) : null,
    targets: model.targets.map((t) => projectScenePoint(model, t.world)),
  },
  projected_points: extraPoints.map((p) => projectScenePoint(model, p)),
};

process.stdout.write(JSON.stringify(out));
