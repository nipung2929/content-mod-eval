"""Deterministic episode generation and dataset building for moderation tasks."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from .perturbations import get_applicable_perturbations
from .templates import TEMPLATES


GOLDEN_EVAL_PATH = Path(__file__).with_name("golden_eval.json")
TRAIN_PATH = Path(__file__).with_name("train.jsonl")
VALIDATION_PATH = Path(__file__).with_name("validation.jsonl")
DATASET_MANIFEST_PATH = Path(__file__).with_name("dataset_manifest.json")
BASELINE_EVAL_SIZE = 16
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


def _render_template(template_key: str, rng: random.Random) -> dict[str, Any]:
    template = TEMPLATES[template_key]
    skeleton = rng.choice(template["skeletons"])
    slots = {
        name: rng.choice(values) for name, values in template.get("slots", {}).items()
    }
    content = skeleton.format(**slots)
    return {
        "template_key": template_key,
        "content": content,
        "label": template["label"],
        "category": template["category"],
        "severity": float(template["severity"]),
        "required_keywords": list(template["required_keywords"]),
        "platform": template["platform"],
        "perturbation": None,
    }


def _apply_perturbation(post: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    applicable = get_applicable_perturbations(post["category"])
    if not applicable:
        return post

    perturbation = rng.choice(applicable)
    mutated = deepcopy(post)
    mutated["content"] = perturbation["transform"](post["content"])
    mutated["perturbation"] = perturbation["name"]

    if not perturbation.get("label_unchanged", True):
        mutated["label"] = perturbation["new_label"]
        mutated["category"] = perturbation["new_category"]
        mutated["severity"] = max(
            0.0,
            min(1.0, post["severity"] + float(perturbation.get("severity_delta", 0.0))),
        )
        mutated["required_keywords"] = list(perturbation.get("required_keywords", []))

    return mutated


def generate_post(template_key: str, seed: int, perturbation_rate: float = 0.0) -> dict[str, Any]:
    rng = random.Random(seed)
    post = _render_template(template_key, rng)
    if rng.random() < perturbation_rate:
        post = _apply_perturbation(post, rng)

    post["seed"] = seed
    post["post_id"] = f"{template_key}-{seed}"
    post["author_history"] = [
        "account_age_days:365",
        "prior_flags:0" if post["label"] == "allow" else "prior_flags:2",
    ]
    post["rule_id"] = CATEGORY_TO_RULE[post["category"]]
    return post


def generate_episode(
    *,
    seed: int,
    episode_length: int,
    category_pool: list[str],
    perturbation_rate: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    posts: list[dict[str, Any]] = []
    for index in range(episode_length):
        template_key = rng.choice(category_pool)
        post_seed = rng.randint(0, 10_000_000)
        post = generate_post(
            template_key,
            seed=post_seed,
            perturbation_rate=perturbation_rate,
        )
        post["episode_index"] = index
        posts.append(post)
    return posts


def load_golden_eval_posts() -> list[dict[str, Any]]:
    with GOLDEN_EVAL_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_split(split: str) -> list[dict[str, Any]]:
    if split == "baseline":
        rows = load_golden_eval_posts()[:BASELINE_EVAL_SIZE]
        for index, row in enumerate(rows):
            row["split"] = "baseline"
            row["example_id"] = f"baseline-{index + 1:04d}"
            row["dataset_index"] = index
        return rows

    path_map = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "eval": GOLDEN_EVAL_PATH,
    }
    try:
        path = path_map[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split '{split}'") from exc

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_by_category(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_split(
    *,
    split: str,
    size: int,
    seed: int,
    category_pool: list[str],
    perturbation_rate: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(size):
        template_key = category_pool[index % len(category_pool)]
        row_seed = rng.randint(0, 10_000_000)
        row = generate_post(template_key, seed=row_seed, perturbation_rate=perturbation_rate)
        row["split"] = split
        row["example_id"] = f"{split}-{index + 1:04d}"
        rows.append(row)

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["dataset_index"] = index
    return rows


def build_dataset_splits() -> dict[str, Any]:
    train_rows = _build_split(
        split="train",
        size=420,
        seed=11,
        category_pool=["clean", "spam", "threat", "hate_speech", "misinformation", "self_harm", "satire"],
        perturbation_rate=0.3,
    )
    validation_rows = _build_split(
        split="validation",
        size=105,
        seed=29,
        category_pool=["clean", "spam", "threat", "hate_speech", "misinformation", "self_harm", "satire"],
        perturbation_rate=0.35,
    )
    eval_rows = _build_split(
        split="eval",
        size=84,
        seed=47,
        category_pool=["clean", "spam", "threat", "hate_speech", "misinformation", "self_harm", "satire"],
        perturbation_rate=0.4,
    )
    baseline_rows = eval_rows[:BASELINE_EVAL_SIZE]
    for index, row in enumerate(baseline_rows):
        row["split"] = "baseline"
        row["example_id"] = f"baseline-{index + 1:04d}"
        row["dataset_index"] = index

    _write_jsonl(TRAIN_PATH, train_rows)
    _write_jsonl(VALIDATION_PATH, validation_rows)
    with GOLDEN_EVAL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(eval_rows, handle, indent=2, ensure_ascii=True)

    manifest = {
        "version": "2026.04",
        "splits": {
            "train": {"size": len(train_rows), "category_counts": _count_by_category(train_rows)},
            "validation": {
                "size": len(validation_rows),
                "category_counts": _count_by_category(validation_rows),
            },
            "baseline": {
                "size": len(baseline_rows),
                "category_counts": _count_by_category(baseline_rows),
            },
            "eval": {"size": len(eval_rows), "category_counts": _count_by_category(eval_rows)},
        },
    }
    with DATASET_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    return manifest
