// Cross-language probe for the God View wall-segment placement math.
//
// Reads a canonical scene payload JSON path from argv[2], runs the exact
// browser module `spatialforge/inspector/embodied_web/wall_geometry.js` under
// node, and prints the world-space wall boxes / opening planes as JSON for the
// pytest harness. This is what lets the regression tests assert that a real
// agent doorway transition never intersects a rendered solid wall segment.
//
// Usage: node tests/tools/wall_geometry_probe.mjs <payload.json>

import { readFileSync } from "node:fs";
import {
  WALL_THICKNESS,
  wallBoxes,
  openingBox,
} from "../../spatialforge/inspector/embodied_web/wall_geometry.js";

const payloadPath = process.argv[2];
if (!payloadPath) {
  process.stderr.write("usage: node wall_geometry_probe.mjs <payload.json>\n");
  process.exit(2);
}

const payload = JSON.parse(readFileSync(payloadPath, "utf8"));
const scene = (payload && payload.scene) || payload || {};

const out = {
  wall_thickness: WALL_THICKNESS,
  walls: (scene.walls || []).map((w) => ({
    id: w.id,
    boxes: wallBoxes(w),
  })),
  doors: (scene.doors || []).map((o) => ({ id: o.id, box: openingBox(o) })),
  windows: (scene.windows || []).map((o) => ({ id: o.id, box: openingBox(o) })),
};

process.stdout.write(JSON.stringify(out));
