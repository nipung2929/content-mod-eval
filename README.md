# Content Moderation OpenEnv

Policy-grounded OpenEnv benchmark for content moderation, designed for deterministic evaluation of agent decisions across simple classification, category prediction, and rule-based reasoning tasks.

## Overview

This project is an OpenEnv environment for **content moderation**, a real-world task performed by social platforms, community products, marketplaces, and support forums at very large scale. The environment presents an agent with a moderation queue and asks it to decide whether a post should be `allow`, `warn`, or `remove` under a platform policy.

The goal of the environment is to evaluate whether an agent can move from simple moderation decisions to more policy-grounded reasoning:
- easy tasks require only the moderation decision
- medium tasks require the decision and the violation category
- hard tasks require the decision, category, cited policy rule, and justification

This makes the environment useful both for agent evaluation and for training setups that need deterministic reward signals on a realistic workflow.

## Why This Environment Matters

Content moderation is a strong benchmark domain because it combines:
- safety-sensitive classification
- contextual reasoning
- policy interpretation
- partial credit for near-correct outputs
- real consequences for false positives and false negatives

The environment is designed to go beyond toy keyword matching by including contextual perturbations such as satire and reporting frames, where harmful words may appear in allowed content.

## Tasks

The environment currently supports four modes:

### `easy`

The agent must predict only the moderation decision:
- `allow`
- `warn`
- `remove`

This is the simplest task and is intended as the entry-level benchmark.

### `medium`

The agent must predict:
- moderation decision
- violation category

This requires the agent to distinguish between different policy buckets rather than only choosing the final action.

### `hard`

The agent must predict:
- moderation decision
- violation category
- cited policy rule ID
- short justification

This task is designed to test policy-grounded reasoning rather than only classification.

### `baseline`

This is the smaller fixed baseline split used by the default inference script for a faster reproducible run.

### `eval`

This is the fixed deterministic evaluation split used for reproducible baseline runs.

## Action Space

The environment accepts a typed `ContentModAction` with the following fields:

- `decision`
  One of `allow`, `warn`, or `remove`
- `category`
  Optional for `easy`, required for `medium`, `hard`, and `eval`
- `cited_rule_id`
  Optional for `easy` and `medium`, required for `hard` and `eval`
- `justification`
  Optional for `easy` and `medium`, required for `hard` and `eval`
- `confidence`
  Float in `[0.0, 1.0]`

The valid category vocabulary is:
- `clean`
- `spam`
- `threat`
- `hate_speech`
- `misinformation`
- `self_harm`
- `satire`
- `reporting_context`

## Observation Space

Each step returns a typed `ContentModObservation` containing:

- `task_name`
  The active task mode
- `instructions`
  Task-specific instructions for the agent
- `post`
  The current content item under review
- `policy`
  The moderation policy and rule definitions
- `queue_position`
  1-based index of the current item in the episode
- `queue_remaining`
  Number of items left after the current one
- `feedback`
  Optional structured feedback from the previous action
- `reward`
  Scalar reward from the previous step
- `done`
  Whether the episode has ended

The observation intentionally does **not** expose hidden ground-truth labels.

## Dataset Design

The environment uses a deterministic synthetic dataset generation pipeline.

### Components

- [`templates.py`](/Users/nipun/Documents/Content_Mod/content_mod/data/templates.py)
  Base moderation scenarios such as spam, threats, hate speech, misinformation, self-harm, satire, and clean content
- [`perturbations.py`](/Users/nipun/Documents/Content_Mod/content_mod/data/perturbations.py)
  Controlled mutations such as typo noise, urgency framing, and reporting-context label flips
- [`generator.py`](/Users/nipun/Documents/Content_Mod/content_mod/data/generator.py)
  Seeded generation of posts, episodes, and deterministic dataset splits

### Dataset Files

- [`train.jsonl`](/Users/nipun/Documents/Content_Mod/content_mod/data/train.jsonl)
- [`validation.jsonl`](/Users/nipun/Documents/Content_Mod/content_mod/data/validation.jsonl)
- [`golden_eval.json`](/Users/nipun/Documents/Content_Mod/content_mod/data/golden_eval.json)
- [`dataset_manifest.json`](/Users/nipun/Documents/Content_Mod/content_mod/data/dataset_manifest.json)

Each example includes:
- content
- label
- category
- rule ID
- severity
- perturbation metadata
- reproducible seed

The `baseline` task is the first 16 examples of the fixed full eval split, so the baseline script stays fast while the benchmark remains larger.

## Reward Design

Rewards are deterministic and depend on task difficulty.

### Easy

- reward `1.0` for correct decision
- reward `0.0` otherwise

### Medium / Hard / Eval

Reward combines:
- decision correctness
- category correctness
- cited rule correctness
- justification keyword coverage

Wrong decisions block most downstream credit. Overconfident wrong answers may incur a small penalty.

The reward function is implemented in [`grader.py`](/Users/nipun/Documents/Content_Mod/content_mod/env/grader.py).

## Environment Architecture

The environment is split into three layers:

### Data layer

Files in [`data/`](/Users/nipun/Documents/Content_Mod/content_mod/data) generate deterministic moderation examples and dataset splits.

### Environment layer

Files in [`env/`](/Users/nipun/Documents/Content_Mod/content_mod/env) define:
- typed models
- task configuration
- reward grading
- episode state
- `reset()`, `step()`, and `state()` behavior

### Serving layer

Files in [`server/`](/Users/nipun/Documents/Content_Mod/content_mod/server) expose the environment through OpenEnv’s FastAPI/WebSocket server wrapper.

## Setup

From the project root:

```bash
uv sync
```

Validate the environment:

```bash
uv run openenv validate
```

Regenerate the dataset splits if needed:

```bash
python -m data.build_dataset
```

## Running Locally

Start the server:

```bash
uv run --project . server
```

In another terminal, run inference:

```bash
set -a
source .env
set +a
uv run python inference.py
```

## Docker Usage

Build the Docker image:

```bash
docker build -t content-mod-env:latest .
```

Run the container:

```bash
docker run --rm -p 8000:8000 content-mod-env:latest
```

Smoke test:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/schema
```

## Baseline Inference Script

The project includes [`inference.py`](/Users/nipun/Documents/Content_Mod/content_mod/inference.py), which:
- uses the OpenAI client for model calls
- runs the deterministic `baseline` task
- emits `[START]`, `[STEP]`, and `[END]` logs in the required stdout format
- supports either a local running server or a local Docker image

Required runtime variables are documented in [`/.env.example`](/Users/nipun/Documents/Content_Mod/content_mod/.env.example).

## Baseline Scores

Current baseline run:

```text
Model: Qwen/Qwen2.5-72B-Instruct
Task: baseline
Episode length: 16
Completion: success=true
Rewards: 1.00,0.65,0.65,0.80,0.90,1.00,0.90,0.90,0.90,0.90,0.90,0.65,0.90,0.90,0.90,1.00
```

## Project Structure

```text
content_mod/
├── __init__.py
├── client.py
├── models.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── .env.example
├── .gitignore
├── data/
│   ├── templates.py
│   ├── perturbations.py
│   ├── generator.py
│   ├── build_dataset.py
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── golden_eval.json
│   └── dataset_manifest.json
├── env/
│   ├── models.py
│   ├── grader.py
│   ├── tasks.py
│   └── environment.py
└── server/
    ├── app.py
    └── content_mod_environment.py
```
