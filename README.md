# AMD Hackathon — Track 1: General-Purpose AI Agent

A containerized agent that answers a mixed batch of benchmark tasks by routing
each one to an appropriately sized Fireworks AI model and batching same-category
work to minimize token usage, since scoring ranks by lowest total tokens among
submissions that pass an accuracy gate.

## What it does

The agent reads `/input/tasks.json`, answers every task, and writes
`/output/results.json`. Tasks span eight categories:

| # | Category | Example |
|---|---|---|
| 1 | Factual knowledge | "What's the capital of France?" |
| 2 | Math reasoning | "What's 15% of 240?" |
| 3 | Sentiment classification | "Is this review positive or negative?" |
| 4 | Summarization | "Summarize this article in 2 sentences." |
| 5 | Named entity recognition | "List all people and organizations mentioned." |
| 6 | Code debugging | "Fix the bug in this function." |
| 7 | Logical/deductive reasoning | "Who sits where, given these constraints?" |
| 8 | Code generation | "Write a function that..." |

Rather than treating every task identically, `classify.py` classifies each
prompt into one of the eight categories using lightweight regex rules (no LLM
call spent on classification), then `main.py` routes it to a model tier:

- **strong** — math and logical reasoning, where accuracy benefits most from a
  more capable model
- **code** — debugging and code generation, routed to a code-specialist model
  if the allowed roster includes one
- **cheap** — factual, sentiment, summarization, and NER, where a smaller
  model is normally sufficient

Model tiers are picked at runtime from whatever `ALLOWED_MODELS` the harness
provides — nothing is hardcoded, since the exact launch-day roster isn't known
in advance. Tiering falls back conservatively (same model for every tier)
whenever the roster's naming doesn't give a reliable enough signal to rank
models by size, rather than guessing.

## Token-efficiency strategy

Since scoring ranks accuracy-passing submissions by total tokens, the agent
batches independent same-category tasks (debugging, logical reasoning, and
code generation — the categories measured to benefit) into a single API call
using a structured JSON request/response format, instead of one call per task.
Empirically this cuts token usage by roughly 50% on these categories, because
the model's reasoning overhead is amortized across several problems in one
generation rather than repeated from scratch per task. Categories where
answers are already short (factual, sentiment, summarization) are answered
individually, since batching them was measured to cost more tokens than it
saves.

If a batch response can't be parsed or is missing an answer for any task in
the group, that task is automatically retried as an individual call — no task
is lost to a formatting hiccup.

## Environment variables (provided by the harness at runtime)

```
FIREWORKS_API_KEY    # Fireworks AI key — never hardcoded
FIREWORKS_BASE_URL   # all API calls are routed through this URL
ALLOWED_MODELS        # comma-separated list of exact model IDs to use
```

## Running locally

```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY="..."
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export ALLOWED_MODELS="model-id-1,model-id-2,..."
python main.py
```

## Building the Docker image

```bash
docker buildx build --platform linux/amd64 -t <your-registry>/<image>:tag --push .
```

The image only contains `main.py`, `classify.py`, and their dependencies —
no test fixtures, virtual environments, or credentials are ever baked in.

## Files

- `main.py` — task orchestration: normalization, classification-driven
  routing, individual and batched API calls, retry/fallback logic
- `classify.py` — regex-based category classification and model-tier
  selection
- `Dockerfile` / `.dockerignore` — minimal `linux/amd64` submission image
- `requirements.txt` — single dependency (`openai`, used against the
  Fireworks-compatible endpoint)
