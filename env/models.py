"""Typed models for the content moderation environment."""

from __future__ import annotations

from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


DecisionLabel = Literal["allow", "warn", "remove"]
TaskName = Literal["easy", "medium", "hard", "baseline", "eval"]


class PolicyRule(BaseModel):
    rule_id: str
    title: str
    description: str
    applies_to: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    policy_name: str
    version: str
    rules: list[PolicyRule]


class ModerationPost(BaseModel):
    post_id: str
    platform: str
    content: str
    author_history: list[str] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0)
    ground_truth_label: Optional[DecisionLabel] = None
    ground_truth_category: Optional[str] = None
    required_keywords: list[str] = Field(default_factory=list)
    perturbation: Optional[str] = None
    seed: Optional[int] = None


class ModerationAction(Action):
    decision: DecisionLabel
    category: Optional[str] = None
    justification: Optional[str] = None
    cited_rule_id: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModerationFeedback(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label_correct: bool
    category_correct: bool
    rule_cited_correctly: bool = False
    justification_keywords_hit: list[str] = Field(default_factory=list)
    penalty_reasons: list[str] = Field(default_factory=list)
    expected_label: DecisionLabel
    expected_category: str


class ModerationObservation(Observation):
    task_name: TaskName
    instructions: str
    post: ModerationPost
    policy: Policy
    queue_position: int = Field(ge=1)
    queue_remaining: int = Field(ge=0)
    allowed_decisions: list[DecisionLabel] = Field(default_factory=lambda: ["allow", "warn", "remove"])
    feedback: Optional[ModerationFeedback] = None


class DecisionRecord(BaseModel):
    post_id: str
    decision: DecisionLabel
    category: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reward: float = Field(ge=0.0, le=1.0)


class EpisodeState(State):
    task_name: Optional[TaskName] = None
    queue_size: int = 0
    current_index: int = 0
    total_reward: float = 0.0
    completed: bool = False
    visible_post_id: Optional[str] = None
    decisions: list[DecisionRecord] = Field(default_factory=list)


class TaskConfig(BaseModel):
    name: TaskName
    episode_length: int
    category_pool: list[str]
    perturbation_rate: float = Field(ge=0.0, le=1.0)
    instructions: str
    require_category: bool = False
    require_justification: bool = False
    require_rule_id: bool = False
    eval_mode: bool = False


class GradeResult(BaseModel):
    reward: float = Field(ge=0.0, le=1.0)
    feedback: ModerationFeedback


def make_policy() -> Policy:
    return Policy(
        policy_name="Platform Trust & Safety Policy",
        version="2026.04",
        rules=[
            PolicyRule(
                rule_id="P1",
                title="Violence and threats",
                description="Remove credible threats, targeted intimidation, and violent encouragement.",
                applies_to=["threat"],
            ),
            PolicyRule(
                rule_id="P2",
                title="Hate and harassment",
                description="Remove hateful abuse targeting protected characteristics.",
                applies_to=["hate_speech"],
            ),
            PolicyRule(
                rule_id="P3",
                title="Spam and scams",
                description="Remove deceptive promotions, scams, and repeated spammy solicitations.",
                applies_to=["spam"],
            ),
            PolicyRule(
                rule_id="P4",
                title="Crisis support",
                description="Escalate self-harm disclosures with supportive intervention instead of removal when appropriate.",
                applies_to=["self_harm"],
            ),
            PolicyRule(
                rule_id="P5",
                title="Unverified harmful claims",
                description="Warn on harmful misinformation when context is ambiguous or evidence is absent.",
                applies_to=["misinformation"],
            ),
            PolicyRule(
                rule_id="P6",
                title="Contextual allowances",
                description="Allow satire, counterspeech, and reporting context when the post does not endorse the harmful content.",
                applies_to=["clean", "satire", "reporting_context"],
            ),
        ],
    )
