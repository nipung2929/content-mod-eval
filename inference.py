"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The local Docker image name for the environment when using from_docker_image().

- Participants must use OpenAI Client for all LLM calls.
- The inference script must be named `inference.py` and placed in the root directory of the project.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, List, Optional

# Avoid expensive third-party Pydantic plugin discovery during OpenAI import.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

from openai import OpenAI

from content_mod import ContentModAction, ContentModEnv


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
CONTENT_MOD_BASE_URL = os.getenv("CONTENT_MOD_BASE_URL", "http://localhost:8000")
BENCHMARK = "content_mod"
MAX_STEPS = 128
TEMPERATURE = 0.2
MAX_TOKENS = 250
SUCCESS_SCORE_THRESHOLD = 0.6
TASK_RUNS = [
    ("easy_001", "easy", 11),
    ("easy_002", "easy", 12),
    ("easy_003", "easy", 13),
    ("medium_001", "medium", 21),
    ("medium_002", "medium", 22),
    ("medium_003", "medium", 23),
    ("hard_001", "hard", 31),
    ("hard_002", "hard", 32),
]

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a content moderation agent.
    Decide whether to allow, warn, or remove each post using the provided policy.
    Valid category values are exactly:
    clean, spam, threat, hate_speech, misinformation, self_harm, satire, reporting_context
    Return exactly one JSON object with these keys:
    - decision: one of allow, warn, remove
    - category: string or null
    - cited_rule_id: string or null
    - justification: string or null
    - confidence: number from 0.0 to 1.0
    Do not include markdown or any extra text.
    """
).strip()

VALID_CATEGORIES = {
    "clean",
    "spam",
    "threat",
    "hate_speech",
    "misinformation",
    "self_harm",
    "satire",
    "reporting_context",
}

CATEGORY_ALIASES = {
    "hate_and_harassment": "hate_speech",
    "hate": "hate_speech",
    "spam_and_scams": "spam",
    "scam": "spam",
    "violence_and_threats": "threat",
    "violence": "threat",
    "threats": "threat",
    "unverified_harmful_claims": "misinformation",
    "self_harm_support": "self_harm",
    "reporting": "reporting_context",
    "contextual_allowance": "reporting_context",
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def compact_action(action: ContentModAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def build_user_prompt(observation: Any, step: int, history: List[str]) -> str:
    rule_lines = [
        f"{rule.rule_id}: {rule.title} - {rule.description}"
        for rule in observation.policy.rules
    ]
    history_block = "\n".join(history[-4:]) if history else "None"
    requirements = observation.metadata.get("task_requirements", {})

    return textwrap.dedent(
        f"""
        Task: {observation.task_name}
        Step: {step}
        Instructions: {observation.instructions}
        Queue position: {observation.queue_position}
        Queue remaining: {observation.queue_remaining}

        Post:
        {observation.post.content}

        Platform: {observation.post.platform}
        Severity: {observation.post.severity}
        Author history: {", ".join(observation.post.author_history) or "None"}

        Policy:
        {chr(10).join(rule_lines)}

        Task requirements:
        {json.dumps(requirements, ensure_ascii=True)}

        Category vocabulary:
        clean, spam, threat, hate_speech, misinformation, self_harm, satire, reporting_context

        Recent history:
        {history_block}

        Return the moderation action JSON only.
        """
    ).strip()


def parse_action_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    return json.loads(raw_text[start : end + 1])


def fallback_action(task_name: str) -> ContentModAction:
    if task_name == "easy":
        return ContentModAction(decision="allow", confidence=0.5)
    if task_name == "medium":
        return ContentModAction(decision="allow", category="clean", confidence=0.5)
    return ContentModAction(
        decision="allow",
        category="clean",
        cited_rule_id="P6",
        justification="Safe discussion context.",
        confidence=0.5,
    )


def normalize_category(raw_category: Any) -> str:
    if raw_category is None:
        return "clean"
    value = str(raw_category).strip().lower()
    if value in VALID_CATEGORIES:
        return value
    value = value.replace(" ", "_")
    if value in VALID_CATEGORIES:
        return value
    return CATEGORY_ALIASES.get(value, "clean")


def normalize_action(task_name: str, payload: dict[str, Any]) -> ContentModAction:
    base_payload: dict[str, Any] = {
        "decision": payload.get("decision", "allow"),
        "confidence": payload.get("confidence", 0.5),
    }

    if task_name in {"medium", "hard", "baseline", "eval"}:
        base_payload["category"] = normalize_category(payload.get("category"))

    if task_name in {"hard", "baseline", "eval"}:
        base_payload["cited_rule_id"] = payload.get("cited_rule_id", "P6")
        base_payload["justification"] = payload.get(
            "justification", "Safe discussion context."
        )

    return ContentModAction(**base_payload)


def get_model_action(
    client: OpenAI,
    observation: Any,
    step: int,
    history: List[str],
) -> ContentModAction:
    user_prompt = build_user_prompt(observation, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = parse_action_payload(text)
        return normalize_action(observation.task_name, payload)
    except Exception:
        return fallback_action(observation.task_name)


async def create_env() -> ContentModEnv:
    if LOCAL_IMAGE_NAME:
        return await ContentModEnv.from_docker_image(LOCAL_IMAGE_NAME)
    env = ContentModEnv(base_url=CONTENT_MOD_BASE_URL)
    await env.connect()
    return env


async def run_episode(
    env: ContentModEnv,
    client: OpenAI,
    task_label: str,
    task_name: str,
    seed: int,
) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name, seed=seed)
        expected_steps = result.observation.queue_remaining + 1
        max_steps = max(MAX_STEPS, expected_steps)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_error: Optional[str] = None
            action = get_model_action(client, result.observation, step, history)

            try:
                result = await env.step(action)
            except Exception as exc:
                action_error = str(exc)
                result = await env.step(fallback_action(task_name))

            reward = float(result.reward or 0.0)
            done = result.done
            rewards.append(reward)
            steps_taken = step

            action_str = compact_action(action)
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=action_error,
            )

            history.append(
                f"step={step} action={action_str} reward={reward:.2f} done={str(done).lower()}"
            )

            if done:
                break

        total_reward = sum(rewards)
        denom = max(1, len(rewards))
        score = max(0.0, min(1.0, total_reward / denom))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await create_env()
    scores: List[tuple[str, float]] = []

    try:
        for task_label, task_name, seed in TASK_RUNS:
            score = await run_episode(
                env=env,
                client=client,
                task_label=task_label,
                task_name=task_name,
                seed=seed,
            )
            scores.append((task_label, score))
    finally:
        try:
            await env.close()
        except Exception:
            pass

    print("=== BASELINE SCORES ===", flush=True)
    for task_label, score in scores:
        print(f"{task_label}: {score:.2f}", flush=True)
    average = sum(score for _, score in scores) / max(1, len(scores))
    print(f"Average: {average:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
