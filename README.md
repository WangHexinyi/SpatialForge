# SpatialForge

SpatialForge is a 3D research workbench for VLM spatial reasoning and cognitive world-model research, progressing from controlled multi-view geometry toward embodied first-person agents, action-conditioned world prediction, spatial memory, active observation, and interactive object search.

## Current stage: v2.0

- **G2.0-A — Camera Geometry Core:** complete
- **G2.0-B — Multi-view Scene Renderer:** complete
- **Next — G2.0-B.1:** Camera Sampling System

SpatialForge keeps world state separate from observation. Its current diagnostic cameras support calibration, canonical multi-view evaluation, and controlled viewpoint generation. A future embodied camera will instead be attached to an agent and change through actions rather than arbitrary viewpoint teleportation.

See [the v3 project plan](docs/PROJECT_PLAN_v3.md) for the research direction, architecture, completed gates, and roadmap.

The closed v1 static synthetic VQA and LoRA work remains recoverable from Git history; it is no longer the active software architecture.

Licensed under the [MIT License](LICENSE).
