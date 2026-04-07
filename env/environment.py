"""Stateful content moderation environment."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..data import generate_episode, load_dataset_split, load_golden_eval_posts
    from .grader import grade_action
    from .models import (
        DecisionRecord,
        EpisodeState,
        ModerationAction,
        ModerationObservation,
        ModerationPost,
        TaskConfig,
        make_policy,
    )
    from .tasks import get_task
except ImportError:
    from data import generate_episode, load_dataset_split, load_golden_eval_posts  # type: ignore[no-redef]
    from env.grader import grade_action  # type: ignore[no-redef]
    from env.models import (  # type: ignore[no-redef]
        DecisionRecord,
        EpisodeState,
        ModerationAction,
        ModerationObservation,
        ModerationPost,
        TaskConfig,
        make_policy,
    )
    from env.tasks import get_task  # type: ignore[no-redef]


class ContentModEnvironment(Environment[ModerationAction, ModerationObservation, EpisodeState]):
    """OpenEnv-compatible moderation queue environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._policy = make_policy()
        self._task: TaskConfig = get_task("easy")
        self._queue: list[dict[str, Any]] = []
        self._state = EpisodeState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        task_name = str(kwargs.get("task", "easy"))
        self._task = get_task(task_name)
        self._state = EpisodeState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task.name,
            queue_size=self._task.episode_length,
            current_index=0,
            total_reward=0.0,
            completed=False,
            decisions=[],
        )

        if self._task.name == "baseline":
            self._queue = load_dataset_split("baseline")
        elif self._task.eval_mode:
            self._queue = load_golden_eval_posts()
        else:
            self._queue = generate_episode(
                seed=0 if seed is None else seed,
                episode_length=self._task.episode_length,
                category_pool=self._task.category_pool,
                perturbation_rate=self._task.perturbation_rate,
            )

        self._state.queue_size = len(self._queue)
        return self._build_observation()

    def step(
        self,
        action: ModerationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        del timeout_s, kwargs
        if self._state.completed or self._state.current_index >= len(self._queue):
            return self._terminal_observation(reward=0.0)

        current = self._queue[self._state.current_index]
        result = grade_action(action, current, self._task)

        self._state.step_count += 1
        self._state.total_reward += result.reward
        self._state.decisions.append(
            DecisionRecord(
                post_id=current["post_id"],
                decision=action.decision,
                category=action.category,
                confidence=action.confidence,
                reward=result.reward,
            )
        )
        self._state.current_index += 1
        self._state.completed = self._state.current_index >= len(self._queue)

        if self._state.completed:
            return self._terminal_observation(reward=result.reward, feedback=result.feedback)

        return self._build_observation(reward=result.reward, feedback=result.feedback)

    @property
    def state(self) -> EpisodeState:
        if self._queue and self._state.current_index < len(self._queue):
            self._state.visible_post_id = self._queue[self._state.current_index]["post_id"]
        else:
            self._state.visible_post_id = None
        return self._state

    def _terminal_observation(
        self,
        *,
        reward: float,
        feedback=None,
    ) -> ModerationObservation:
        placeholder_post = ModerationPost(
            post_id="episode-complete",
            platform="system",
            content="Episode complete.",
            author_history=[],
            severity=0.0,
        )
        return ModerationObservation(
            task_name=self._task.name,
            instructions=self._task.instructions,
            post=placeholder_post,
            policy=self._policy,
            queue_position=max(1, len(self._queue)),
            queue_remaining=0,
            done=True,
            reward=reward,
            feedback=feedback,
            metadata={
                "episode_id": self._state.episode_id,
                "total_reward": round(self._state.total_reward, 4),
                "episode_score": round(
                    self._state.total_reward / max(1, len(self._queue)),
                    4,
                ),
                "task_requirements": self._task.model_dump(),
            },
        )

    def _build_observation(
        self,
        *,
        reward: float | None = None,
        feedback=None,
    ) -> ModerationObservation:
        current = self._queue[self._state.current_index]
        post = ModerationPost(
            post_id=current["post_id"],
            platform=current["platform"],
            content=current["content"],
            author_history=current["author_history"],
            severity=current["severity"],
            ground_truth_label=None,
            ground_truth_category=None,
            required_keywords=[],
            perturbation=current.get("perturbation"),
            seed=current.get("seed"),
        )
        return ModerationObservation(
            task_name=self._task.name,
            instructions=self._task.instructions,
            post=post,
            policy=self._policy,
            queue_position=self._state.current_index + 1,
            queue_remaining=len(self._queue) - self._state.current_index - 1,
            done=False,
            reward=reward,
            feedback=feedback,
            metadata={
                "episode_id": self._state.episode_id,
                "task_requirements": self._task.model_dump(),
            },
        )
