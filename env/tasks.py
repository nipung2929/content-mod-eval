"""Task configuration for the moderation environment."""

from __future__ import annotations

from .models import TaskConfig


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        episode_length=4,
        category_pool=["clean", "spam", "threat", "satire"],
        perturbation_rate=0.1,
        instructions="Decide whether each post should be allowed, warned, or removed.",
        require_category=False,
        require_justification=False,
        require_rule_id=False,
    ),
    "medium": TaskConfig(
        name="medium",
        episode_length=6,
        category_pool=["clean", "spam", "threat", "hate_speech", "misinformation", "satire"],
        perturbation_rate=0.25,
        instructions="Decide the moderation action and identify the best matching policy category.",
        require_category=True,
        require_justification=False,
        require_rule_id=False,
    ),
    "hard": TaskConfig(
        name="hard",
        episode_length=8,
        category_pool=["clean", "spam", "threat", "hate_speech", "misinformation", "self_harm", "satire"],
        perturbation_rate=0.4,
        instructions="Decide the action, identify the category, cite the rule, and justify the decision using policy-aware reasoning.",
        require_category=True,
        require_justification=True,
        require_rule_id=True,
    ),
    "baseline": TaskConfig(
        name="baseline",
        episode_length=16,
        category_pool=[],
        perturbation_rate=0.0,
        instructions="Run the fixed baseline evaluation split with full policy reasoning.",
        require_category=True,
        require_justification=True,
        require_rule_id=True,
        eval_mode=True,
    ),
    "eval": TaskConfig(
        name="eval",
        episode_length=84,
        category_pool=[],
        perturbation_rate=0.0,
        instructions="Run the fixed golden evaluation split with full policy reasoning.",
        require_category=True,
        require_justification=True,
        require_rule_id=True,
        eval_mode=True,
    ),
}


def get_task(task_name: str) -> TaskConfig:
    try:
        return TASKS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {', '.join(TASKS)}") from exc
