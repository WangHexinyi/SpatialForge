"""SpatialForge Embodied Agent runtime (ProcTHOR / AI2-THOR mainline).

This package implements the *new* primary training paradigm: an embodied,
first-person agent acting in a real ProcTHOR house. Camera / God View are
research/verification infrastructure; the Agent is the subject.

Modules
-------
contracts
    Pure data contracts: AgentObservation, AgentAction, AgentState,
    EpisodeState, ObjectSearchTask, EpisodeHistory + serialization.
actions
    AgentAction space definition and helpers.
model_input
    Strict allowlist builder for model input and privileged truth isolation.
geometry_util
    Light, dependency-free math helpers used by deterministic verifier.
verify
    Object-search success verifier interface.
teacher
    Teacher / shortest-path reference trajectory generation (privileged).
environment
    EmbodiedEnvironment + closed-loop EpisodeRunner over a pluggable backend.
backends
    Deterministic scene-graph backend (CPU test path) and real ProcTHOR /
    AI2-THOR backend.
i18n
    Central zh-CN / en / bilingual dictionary + translation helper.
"""
