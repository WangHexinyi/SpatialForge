// Cross-language probe for the canonical world -> Three.js conversion.
//
// Reads a JSON document from argv[2]:
//   { points: [[x,y,z], ...], dirs: [[x,y,z], ...],
//     bounds: {min:[...], max:[...]}, yaws: [deg, ...] }
// and prints the shipped `world_space.js` results as JSON for the pytest
// harness. This proves the 3D semantic renderer uses one conversion and lets
// the tests verify the top-down camera orientation numerically without WebGL.
//
// Usage: node tests/tools/world_space_probe.mjs <input.json>

import { readFileSync } from "node:fs";
import {
  worldToThree,
  threeToWorld,
  worldDirToThree,
  worldYawToThree,
  wallRotationY,
  worldBoundsToThree,
  topViewCamera,
} from "../../spatialforge/inspector/embodied_web/world_space.js";

const inputPath = process.argv[2];
if (!inputPath) {
  process.stderr.write("usage: node world_space_probe.mjs <input.json>\n");
  process.exit(2);
}

const input = JSON.parse(readFileSync(inputPath, "utf8"));
const threeBounds = input.bounds ? worldBoundsToThree(input.bounds) : null;

const out = {
  points: (input.points || []).map(worldToThree),
  roundtrip: (input.points || []).map((p) => threeToWorld(worldToThree(p))),
  dirs: (input.dirs || []).map(worldDirToThree),
  yaw_dirs: (input.yaws || []).map(worldYawToThree),
  yaw_rotations: (input.yaws || []).map(wallRotationY),
  bounds: threeBounds,
  top: threeBounds ? topViewCamera(threeBounds) : null,
};

process.stdout.write(JSON.stringify(out));
