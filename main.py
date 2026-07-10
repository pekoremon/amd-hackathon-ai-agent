import asyncio
import json
import os
import re
import sys
import time

from openai import APIStatusError, AsyncOpenAI

from classify import (
    CODE_DEBUG,
    CODE_GEN,
    LOGIC,
    classify_category,
    model_for_category,
    pick_model_tiers,
)


def normalize_prompt(text: str) -> str:
    """Strip whitespace waste that costs tokens but carries no meaning.

    Only touches things that are safe regardless of content: outer whitespace,
    trailing spaces at line ends, and runs of blank lines (collapsed to at most
    one). Deliberately does NOT touch leading/internal horizontal whitespace,
    since prompts embed inline code (categories 6/8) where indentation is
    semantically significant — collapsing it would silently corrupt the code.
    """
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    result = []
    blank_streak = 0
    for line in lines:
        if line == "":
            blank_streak += 1
            if blank_streak <= 1:
                result.append(line)
        else:
            blank_streak = 0
            result.append(line)
    return "\n".join(result).strip()


INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

REQUEST_TIMEOUT_S = 29
MAX_RETRIES = 1
CONCURRENCY = 16

CATEGORY_INSTRUCTIONS = {
    1: "Answer directly and accurately, no preamble.",
    2: "Work it out internally; do not show intermediate steps. Output only the final result, with units if relevant.",
    3: "Output the sentiment label plus one short justifying sentence.",
    4: "Output only the summary, matching the length/format constraint given.",
    5: "Extract entities grouped by type (PERSON, ORGANIZATION, LOCATION, DATE) as a compact list.",
    6: "Output only the corrected, complete code, no visible reasoning — no explanation unless asked.",
    7: "Solve internally with no visible reasoning; verify every condition holds, then state the answer with each "
       "entity's assignment explicitly labeled (e.g. 'Position 1: Name' or 'Name: value') — never a bare unlabeled list.",
    8: "Output only a correct, well-structured implementation matching the spec, no visible reasoning.",
}

# 5000 caused 6/19 real tasks to time out (asyncio.TimeoutError at the 29s wall)
# on the hardest categories -- the model never finishes generating, so it comes
# back completely empty instead of a shorter-but-complete answer. Measured
# empirically against real hard prompts (math derivations, a 7-entity logic
# puzzle, full-program codegen): 1500 reliably finishes with 11+ seconds of
# margin (worst observed: 17.8s), while 2000 already timed out once. Current
# goal is maximizing how many of the 19 tasks get *any* complete answer, not
# token efficiency, so every category is raised well above its original ceiling
# (350/700/200/300/350/900/900/900) but capped at the empirically safe value.
CATEGORY_MAX_TOKENS = {
    1: 1500,
    2: 1500,
    3: 1000,
    4: 1000,
    5: 1000,
    6: 1500,
    7: 1500,
    8: 1500,
}

# Raised 2/6/7/8 from "low": these are the categories where models were observed
# spending their visible output arguing with themselves about instruction
# conflicts rather than reasoning privately. More reasoning budget gives the
# model room to work through genuinely hard problems (multi-step derivations,
# constraint puzzles, full programs) before answering, instead of doing that
# work out loud in the completion. "high" across the board pushed several
# tasks over the 29s wall (more hidden reasoning time = less margin), so
# "medium" is the current compromise -- less risk of timeout, still meant to
# reduce meta-commentary versus "low". 1/3/4/5 stay "none" since they don't
# need multi-step reasoning to answer correctly regardless of prompt difficulty.
CATEGORY_REASONING_EFFORT = {
    1: "none",
    2: "medium",
    3: "none",
    4: "none",
    5: "none",
    6: "medium",
    7: "medium",
    8: "medium",
}

# Previously told the model to be terse and skip chain-of-thought regardless of
# what the task itself asked for. That directly conflicted with prompts that
# explicitly request extensive output (full derivations, long-form
# justification, complete production code), and models were burning their
# entire token budget arguing with themselves about the contradiction instead
# of answering -- observed across math, logic, and codegen categories. Current
# priority is correctness, not token cost, so this now just asks the model to
# follow whatever the prompt itself asks for, with no length pressure.
SYSTEM_PREFIX = (
    "Answer the question fully and accurately, following any instructions in the "
    "prompt exactly, including any requested length, level of detail, or format.\n\n"
)

# Measured empirically (3 replicated runs): batching same-category tasks into one
# call consistently costs MORE tokens for categories 1/3/4 (already-terse answers,
# the batch's JSON overhead and longer system prompt don't get amortized away) but
# consistently saves 30-63% for these three — the model reasons noticeably less
# per-problem when it has several to get through in one call versus one in isolation.
#
# Re-tested after confirming batching wasn't the original cause of the accuracy
# failure -- disabled again after all 3 batches (categories 6/7/8) failed and
# fell back to individual calls on the real 19-task eval. The batch max_tokens
# ceiling (CATEGORY_MAX_TOKENS * batch size) assumes spare capacity to combine
# several tasks per call, but these tasks are individually already near the
# per-task safe ceiling -- there's no room to batch them without exceeding the
# 29s wall. Fallback logic recovered all 8 affected tasks with no data lost,
# but wall-clock time ballooned (168s vs ~30s) and likely caused collateral
# timeouts on unrelated tasks via semaphore contention while failed batches
# held their retry slots. Net negative for this task profile; keep disabled.
BATCH_CATEGORIES = set()
BATCH_SIZE = 5

BATCH_SYSTEM_PREFIX = (
    "You are completing multiple independent tasks from an automated benchmark. "
    "Answer each question completely independently — do not let the content, topic, or "
    "tone of one question influence your answer to another. Respond with only the final "
    "answer for each, no chain-of-thought, no preamble.\n\n"
    "Return ONLY a JSON array, one object per question, in this exact format, no markdown "
    'fences, no text outside the array:\n[{"task_id": "<id>", "answer": "<answer>"}, ...]\n\n'
)


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", (text or "").strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_tasks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def _complete(client: AsyncOpenAI, model: str, messages: list, max_tokens: int, reasoning_effort: str | None):
    """Chat completion with a fallback for reasoning_effort support, which varies
    by model: some reject a given value, others reject the field outright. Rather
    than pattern-matching specific error wording (which we can only ever observe
    from models we've actually called, and the launch-day roster may differ from
    those), any API-level rejection while reasoning_effort was set retries once
    with it stripped entirely -- a model that doesn't understand the field can
    still answer the question."""
    extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    try:
        return await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0, extra_body=extra_body,
        )
    except APIStatusError:
        if not extra_body:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0,
        )


async def answer_task(client: AsyncOpenAI, task: dict, tiers: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task["task_id"]
    prompt = normalize_prompt(task["prompt"])
    category = classify_category(prompt)
    model = model_for_category(category, tiers)
    system_prompt = SYSTEM_PREFIX + CATEGORY_INSTRUCTIONS[category]
    max_tokens = CATEGORY_MAX_TOKENS[category]
    reasoning_effort = CATEGORY_REASONING_EFFORT[category]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    last_error = None
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    _complete(client, model, messages, max_tokens, reasoning_effort),
                    timeout=REQUEST_TIMEOUT_S,
                )
                answer = (resp.choices[0].message.content or "").strip()
                return {"task_id": task_id, "answer": answer}
            except Exception as exc:
                # Broad on purpose: one task's failure must not crash the whole batch.
                last_error = exc
                print(f"[warn] task {task_id} attempt {attempt} failed: {exc}", file=sys.stderr)

    print(f"[error] task {task_id} failed after retries: {last_error}", file=sys.stderr)
    return {"task_id": task_id, "answer": ""}


async def answer_batch(client: AsyncOpenAI, category: int, batch_tasks: list[dict], tiers: dict, sem: asyncio.Semaphore) -> list[dict]:
    """Answer several same-category tasks in one call. Returns one dict per task;
    a task whose answer couldn't be recovered gets answer=None so the caller can
    fall back to answering it individually via answer_task instead of losing it."""
    model = model_for_category(category, tiers)
    system_prompt = BATCH_SYSTEM_PREFIX + CATEGORY_INSTRUCTIONS[category]
    reasoning_effort = CATEGORY_REASONING_EFFORT[category]
    max_tokens = CATEGORY_MAX_TOKENS[category] * len(batch_tasks) + 100

    payload = [{"task_id": t["task_id"], "prompt": normalize_prompt(t["prompt"])} for t in batch_tasks]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    expected_ids = {t["task_id"] for t in batch_tasks}

    last_error = None
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    _complete(client, model, messages, max_tokens, reasoning_effort),
                    timeout=REQUEST_TIMEOUT_S,
                )
                parsed = extract_json_array(resp.choices[0].message.content)
                if parsed is not None:
                    got = {
                        item["task_id"]: str(item["answer"]).strip()
                        for item in parsed
                        if isinstance(item, dict) and "task_id" in item and "answer" in item
                    }
                    missing = expected_ids - got.keys()
                    if missing:
                        print(f"[warn] batch cat={category} missing task_ids, falling back individually: {missing}", file=sys.stderr)
                    return [{"task_id": tid, "answer": got.get(tid)} for tid in expected_ids]
                last_error = "response was not a valid JSON array"
                print(f"[warn] batch cat={category} attempt {attempt} failed to parse JSON array", file=sys.stderr)
            except Exception as exc:
                last_error = exc
                print(f"[warn] batch cat={category} attempt {attempt} failed: {exc}", file=sys.stderr)

    print(f"[error] batch cat={category} failed after retries: {last_error}", file=sys.stderr)
    return [{"task_id": tid, "answer": None} for tid in expected_ids]


def _plan_batches(tasks: list[dict]) -> tuple[list[dict], list[tuple[int, list[dict]]]]:
    """Split tasks into (individual_tasks, batch_groups). A category only gets
    grouped into batches if it's in BATCH_CATEGORIES; a leftover chunk of size 1
    is answered individually instead, since batching a single task only adds the
    JSON-format overhead with no task to amortize it against."""
    by_category: dict[int, list[dict]] = {}
    for t in tasks:
        category = classify_category(normalize_prompt(t["prompt"]))
        by_category.setdefault(category, []).append(t)

    individual_tasks = []
    batch_groups = []
    for category, cat_tasks in by_category.items():
        if category not in BATCH_CATEGORIES:
            individual_tasks.extend(cat_tasks)
            continue
        for i in range(0, len(cat_tasks), BATCH_SIZE):
            chunk = cat_tasks[i:i + BATCH_SIZE]
            if len(chunk) == 1:
                individual_tasks.append(chunk[0])
            else:
                batch_groups.append((category, chunk))

    return individual_tasks, batch_groups


async def _answer_all(client: AsyncOpenAI, tasks: list[dict], tiers: dict, sem: asyncio.Semaphore) -> list[dict]:
    """Answer every task, batching where BATCH_CATEGORIES applies, falling back
    to an individual call for any task a batch failed to recover an answer for."""
    task_by_id = {t["task_id"]: t for t in tasks}
    individual_tasks, batch_groups = _plan_batches(tasks)

    individual_results = await asyncio.gather(*(answer_task(client, t, tiers, sem) for t in individual_tasks))
    batch_results = await asyncio.gather(*(answer_batch(client, cat, chunk, tiers, sem) for cat, chunk in batch_groups))

    results = list(individual_results)
    fallback_needed = []
    for batch in batch_results:
        for item in batch:
            if item["answer"] is None:
                fallback_needed.append(task_by_id[item["task_id"]])
            else:
                results.append(item)

    if fallback_needed:
        print(f"[info] falling back to individual calls for {len(fallback_needed)} tasks", file=sys.stderr)
        results.extend(await asyncio.gather(*(answer_task(client, t, tiers, sem) for t in fallback_needed)))

    return results


async def run() -> int:
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ["FIREWORKS_BASE_URL"]
    allowed_models = os.environ["ALLOWED_MODELS"].split(",")
    allowed_models = [m.strip() for m in allowed_models if m.strip()]

    if not allowed_models:
        print("[error] ALLOWED_MODELS is empty", file=sys.stderr)
        return 1

    overrides = {
        "strong": os.environ.get("MODEL_TIER_STRONG"),
        "cheap": os.environ.get("MODEL_TIER_CHEAP"),
        "code": os.environ.get("MODEL_TIER_CODE"),
    }
    tiers = pick_model_tiers(allowed_models, overrides)
    print(f"[info] model tiers: {tiers}", file=sys.stderr)

    tasks = load_tasks(INPUT_PATH)
    print(f"[info] loaded {len(tasks)} tasks", file=sys.stderr)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(CONCURRENCY)

    start = time.monotonic()
    results = await _answer_all(client, tasks, tiers, sem)
    print(f"[info] completed {len(results)} tasks in {time.monotonic() - start:.1f}s", file=sys.stderr)

    write_results(OUTPUT_PATH, results)
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(run())
    except Exception as exc:
        # Top-level guard: any unhandled failure must still exit non-zero.
        print(f"[fatal] {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
