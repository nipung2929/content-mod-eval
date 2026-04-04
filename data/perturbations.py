"""Contextual perturbations for moderation posts."""

from __future__ import annotations

from typing import Callable


Perturbation = dict[str, object]


def add_typo_noise(text: str) -> str:
    return text.replace("the", "teh", 1) if "the" in text else f"{text} pls respond"


def add_urgency(text: str) -> str:
    return f"{text} This is urgent."


def wrap_reporting_frame(text: str) -> str:
    return f"I'm reporting this message for safety review: \"{text}\""


PERTURBATIONS: list[Perturbation] = [
    {
        "name": "typo_noise",
        "applies_to": {"clean", "spam", "misinformation", "satire"},
        "transform": add_typo_noise,
        "label_unchanged": True,
    },
    {
        "name": "urgency",
        "applies_to": {"spam", "self_harm"},
        "transform": add_urgency,
        "label_unchanged": True,
    },
    {
        "name": "reporting_frame",
        "applies_to": {"threat", "hate_speech"},
        "transform": wrap_reporting_frame,
        "label_unchanged": False,
        "new_label": "allow",
        "new_category": "reporting_context",
        "severity_delta": -0.5,
        "required_keywords": ["reporting", "context"],
    },
]


def get_applicable_perturbations(category: str) -> list[Perturbation]:
    return [
        perturbation
        for perturbation in PERTURBATIONS
        if category in perturbation["applies_to"]
    ]
