"""Embodied closed-loop runtime.

``EmbodiedEnvironment`` is the researcher-side controller that runs one episode
of an object-search task against a pluggable backend. It owns episode/task
bookkeeping, step budget, the Done verifier, and the strict split between
model input and privileged truth.

Backends (in :mod:`spatialforge.embodied.backends`) provide the real or
deterministic environment physics + first-person rendering. A backend exposes
two low-level methods that both return a **raw result dict**::

    {"observation": AgentObservation,
     "agent_state":  AgentState,
     "action":       AgentAction}

The runner is the only component allowed to wrap these raw results into a full
``EmbodiedTransition`` (model input + privileged payload + verifier decisions),
so the leakage boundary has a single owner.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from spatialforge.embodied.contracts import (
    EPISODE_MAX_STEPS_DEFAULT,
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentState,
    EpisodeHistory,
    EpisodeState,
    EpisodeStatus,
    ObjectSearchTask,
    TaskStatus,
)
from spatialforge.embodied.model_input import (
    build_agent_model_input,
    build_privileged_debug_payload,
    validate_model_input_isolation,
)


@dataclass
class EmbodiedTransition:
    """Full outcome of one environment step."""

    observation: AgentObservation
    agent_state: AgentState
    action: AgentAction
    task: ObjectSearchTask
    episode: EpisodeState
    step: int
    terminal: bool
    model_input: Dict[str, Any]
    privileged: Dict[str, Any]


class EmbodiedBackend:
    """Pluggable environment backend protocol (implement in backends/)."""

    # -- lifecycle -----------------------------------------------------
    def load_house(self, house_id: str, **opts) -> Any:
        raise NotImplementedError

    def reset(self, task: ObjectSearchTask, **opts) -> Dict[str, Any]:
        """Return raw result dict with keys observation/agent_state/action."""
        raise NotImplementedError

    def apply_action(
        self, action_type: AgentActionType, parameters: Dict[str, Any], step: int
    ) -> Dict[str, Any]:
        """Execute one action; return raw result dict."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # -- authoritative metadata (verifier / teacher / debug) -----------
    def target_category_object_ids(self, category: str) -> List[str]:
        raise NotImplementedError

    def target_is_visible(self, object_id: str) -> bool:
        raise NotImplementedError

    def object_position(self, object_id: str) -> Optional[Any]:
        return None

    def current_room(self) -> Optional[str]:
        return None

    def reachable_positions(self) -> List[Any]:
        return []

    def house(self) -> Any:
        return None


class EmbodiedEnvironment:
    """Researcher-facing controller for one object-search episode."""

    def __init__(
        self,
        backend: EmbodiedBackend,
        episode_max_steps: int = EPISODE_MAX_STEPS_DEFAULT,
    ):
        self.backend = backend
        self.episode_max_steps = episode_max_steps
        self.episode: Optional[EpisodeState] = None
        self.task: Optional[ObjectSearchTask] = None
        self.history: Optional[EpisodeHistory] = None
        self.observation: Optional[AgentObservation] = None
        self.agent_state: Optional[AgentState] = None
        self.last_raw_action: Optional[AgentAction] = None
        #: wall-clock duration of each genuine env.step in ms (empty until
        #: start_episode; used by throughput instrumentation only).
        self.env_step_times_ms: List[float] = []

    # ------------------------------------------------------------------
    def start_episode(self, house_id: str, target_category: str, **opts):
        """Reset the backend, build a task + episode, return initial transition."""
        task_id = opts.pop("task_id", None) or f"task-{uuid.uuid4().hex[:12]}"
        episode_id = opts.pop("episode_id", None) or f"ep-{uuid.uuid4().hex[:12]}"

        house = self.backend.house()
        effective_house_id = house.house_id if house is not None else house_id

        instruction = opts.pop("instruction", None)
        if instruction is None:
            instruction = f"Find a {target_category}."

        object_ids = self.backend.target_category_object_ids(target_category)

        task = ObjectSearchTask(
            task_id=task_id,
            house_id=effective_house_id,
            target_category=target_category,
            instruction=instruction,
            max_steps=int(opts.pop("max_steps", self.episode_max_steps)),
            target_object_ids=list(object_ids),
            reachable_positions=list(self.backend.reachable_positions()),
        )
        task.target_positions = self._target_positions(object_ids)

        self.task = task
        self.episode = EpisodeState(
            episode_id=episode_id,
            house_id=effective_house_id,
            task_id=task_id,
            max_steps=task.max_steps,
            created_ms=int(time.time() * 1000),
        )
        self.history = EpisodeHistory(episode=self.episode)
        self.env_step_times_ms = []
        raw = self.backend.reset(task, **opts)
        self._adopt_raw(raw)
        self.task.status = TaskStatus.RUNNING
        self.episode.status = EpisodeStatus.RUNNING
        self._update_observation_context()
        return self._transition()

    # ------------------------------------------------------------------
    def step(
        self, action_type: AgentActionType, parameters: Optional[Dict] = None
    ) -> Optional[EmbodiedTransition]:
        """Run one action. Returns None once the episode has terminated."""
        parameters = dict(parameters or {})
        if self.episode is None or self.episode.status != EpisodeStatus.RUNNING:
            return None

        t0 = time.perf_counter()
        if action_type == AgentActionType.DONE:
            trans = self._handle_done()
            self.env_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
            return trans

        raw = self.backend.apply_action(action_type, parameters, self.episode.step)
        self.episode.step += 1
        self.task.step = self.episode.step
        self._adopt_raw(raw)
        self._update_observation_context()

        if self.episode.step >= self.task.max_steps:
            self._finish(
                EpisodeStatus.FAILURE,
                success=False,
                reason="step budget exhausted before Done.",
            )
        trans = self._transition()
        self.env_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return trans

    # ------------------------------------------------------------------
    def _handle_done(self) -> EmbodiedTransition:
        visible = self._any_target_visible()
        reason = (
            "Done emitted while target category object is visible."
            if visible
            else "Done emitted but no target-category object is visible."
        )
        self._finish(
            EpisodeStatus.SUCCESS if visible else EpisodeStatus.FAILURE,
            success=visible,
            reason=reason,
        )
        done = AgentAction(
            action_type=AgentActionType.DONE,
            step=self.episode.step,
            success=visible,
            timestamp_ms=int(time.time() * 1000),
        )
        self.history.record_action(done)
        self.last_raw_action = done
        self.observation.cause_action = done
        self._update_observation_context()
        return self._transition()

    # ------------------------------------------------------------------
    def _adopt_raw(self, raw: Dict[str, Any]) -> None:
        self.observation = raw["observation"]
        self.agent_state = raw["agent_state"]
        action = raw.get("action")
        if action is not None:
            self.last_raw_action = action
            self.history.record_action(action)
        self.history.record_observation(self.observation)
        if self.agent_state is not None:
            self.agent_state = self.agent_state.__class__(
                **{
                    **self.agent_state.__dict__,
                    "step_count": self.episode.step,
                    "last_action": action.describe() if action else None,
                    "last_action_success": action.success if action else None,
                }
            )

    def _update_observation_context(self) -> None:
        self.observation.cause_action = self.last_raw_action
        self.observation.instruction = self.task.instruction
        self.observation.step = self.episode.step
        # verifier-supplied target-visible flag surfaces here (authoritative):
        if self.task is not None:
            self.observation.target_visible = self._any_target_visible()

    def _any_target_visible(self) -> bool:
        return any(
            self.backend.target_is_visible(oid)
            for oid in (self.task.target_object_ids if self.task else [])
        )

    def _target_positions(self, object_ids: List[str]) -> List[Any]:
        positions = []
        for oid in object_ids:
            p = self.backend.object_position(oid)
            if p is not None:
                positions.append(p)
        return positions

    def _finish(self, status: EpisodeStatus, success: bool, reason: str) -> None:
        self.episode.status = status
        self.episode.success = success
        self.episode.success_reason = reason
        self.episode.completed_ms = int(time.time() * 1000)
        self.task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
        self.task.success = success
        self.task.success_reason = reason

    def _transition(self) -> EmbodiedTransition:
        model_input = build_agent_model_input(self.observation, self.task, self.history)
        validate_model_input_isolation(model_input)

        privileged = build_privileged_debug_payload(
            self.observation, self.task, self.agent_state, self.backend.house()
        )

        terminal = self.episode.status != EpisodeStatus.RUNNING
        return EmbodiedTransition(
            observation=self.observation,
            agent_state=self.agent_state,
            action=self.last_raw_action,
            task=self.task,
            episode=self.episode,
            step=self.episode.step,
            terminal=terminal,
            model_input=model_input,
            privileged=privileged,
        )
