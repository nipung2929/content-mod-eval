# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Content moderation environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ContentModAction,
        ContentModObservation,
        EpisodeState,
        ModerationFeedback,
        ModerationPost,
        Policy,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        ContentModAction,
        ContentModObservation,
        EpisodeState,
        ModerationFeedback,
        ModerationPost,
        Policy,
    )


class ContentModEnv(
    EnvClient[ContentModAction, ContentModObservation, EpisodeState]
):
    """WebSocket client for the content moderation environment."""

    def _step_payload(self, action: ContentModAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ContentModObservation]:
        obs_data = payload.get("observation", {})
        feedback_data = obs_data.get("feedback")
        post_data = obs_data.get("post", {})
        policy_data = obs_data.get("policy", {})
        observation = ContentModObservation(
            task_name=obs_data.get("task_name", "easy"),
            instructions=obs_data.get("instructions", ""),
            post=ModerationPost(**post_data),
            policy=Policy(**policy_data),
            queue_position=obs_data.get("queue_position", 1),
            queue_remaining=obs_data.get("queue_remaining", 0),
            allowed_decisions=obs_data.get("allowed_decisions", ["allow", "warn", "remove"]),
            feedback=ModerationFeedback(**feedback_data) if feedback_data else None,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EpisodeState:
        return EpisodeState(**payload)
