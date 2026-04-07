"""Deterministic grading for moderation actions."""

from __future__ import annotations

from .models import GradeResult, ModerationAction, ModerationFeedback, TaskConfig


CATEGORY_TO_RULE = {
    "threat": "P1",
    "hate_speech": "P2",
    "spam": "P3",
    "self_harm": "P4",
    "misinformation": "P5",
    "clean": "P6",
    "satire": "P6",
    "reporting_context": "P6",
}


def _keyword_hits(justification: str | None, required_keywords: list[str]) -> list[str]:
    if not justification:
        return []
    lowered = justification.lower()
    return [keyword for keyword in required_keywords if keyword.lower() in lowered]


def grade_action(
    action: ModerationAction,
    ground_truth: dict[str, object],
    task: TaskConfig,
) -> GradeResult:
    expected_label = str(ground_truth["label"])
    expected_category = str(ground_truth["category"])
    required_keywords = list(ground_truth.get("required_keywords", []))
    expected_rule_id = CATEGORY_TO_RULE.get(expected_category)

    label_correct = action.decision == expected_label
    category_correct = action.category == expected_category if task.require_category else False
    rule_correct = action.cited_rule_id == expected_rule_id if task.require_rule_id else False
    keyword_hits = _keyword_hits(action.justification, required_keywords)

    # Keep rewards strictly inside (0, 1) for valid actions so task-level
    # aggregate scores never saturate at the boundaries.
    reward = 0.05
    penalties: list[str] = []

    if label_correct:
        reward += 0.9 if not task.require_category and not task.require_justification else 0.35
    elif action.confidence > 0.85:
        penalties.append("overconfident_wrong_label")

    if task.require_category and label_correct:
        if category_correct:
            reward += 0.45 if not task.require_justification else 0.25
        elif action.category is None:
            penalties.append("missing_category")

    if task.require_rule_id and label_correct:
        if rule_correct:
            reward += 0.15
        else:
            penalties.append("incorrect_rule")

    if task.require_justification and label_correct:
        if required_keywords:
            reward += 0.2 * (len(keyword_hits) / len(required_keywords))
        if not action.justification:
            penalties.append("missing_justification")

    if action.confidence > 0.95 and not label_correct:
        reward = max(0.01, reward - 0.04)

    reward = max(0.01, min(0.95, reward))

    return GradeResult(
        reward=reward,
        feedback=ModerationFeedback(
            score=reward,
            label_correct=label_correct,
            category_correct=category_correct,
            rule_cited_correctly=rule_correct,
            justification_keywords_hit=keyword_hits,
            penalty_reasons=penalties,
            expected_label=expected_label,
            expected_category=expected_category,
        ),
    )
