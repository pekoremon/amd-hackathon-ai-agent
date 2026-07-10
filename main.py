import asyncio
import json
import os
import re
import subprocess
import sys
import time

from openai import APIStatusError, AsyncOpenAI

from classify import (
    CODE_DEBUG,
    CODE_GEN,
    LOGIC,
    MATH,
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

# Ultra-diet: every instruction minimal, and answer-FIRST ordering for the
# reasoning categories -- models put the conclusion last by default, so when a
# tight max_tokens cap truncates the tail, the answer itself is what gets cut.
# Leading with the final answer makes truncation cost explanation, not answer.
CATEGORY_INSTRUCTIONS = {
    1: "Give a direct answer first, then only the explanation the prompt explicitly requires.",
    2: "State the final result (with units) in the first line, then only the working the prompt "
       "explicitly requires.",
    3: "Output the sentiment label plus one short justifying sentence.",
    4: "Output only the summary, obeying the stated format/length constraint.",
    5: "List entities grouped by type (PERSON, ORGANIZATION, LOCATION, DATE), nothing else.",
    6: "Output only the corrected, complete code.",
    7: "First line: the final answer with each assignment explicitly labeled (e.g. 'Position 1: "
       "Name'). Then only the justification the prompt explicitly requires.",
    8: "Output only a correct implementation matching the spec.",
}

# Deliberate, bounded token/accuracy trade: the SACRIFICE_COUNT tasks expected
# to be most expensive get one cheap 200-token no-tool call instead of the full
# tool-chain pipeline. Selection is generic (category cost ranking measured
# across many runs, then prompt length) -- never tied to specific question
# content. Unlike the random-sacrifice experiment (which hit already-cheap
# tasks and saved nothing at the same accuracy cost), this targets exactly the
# tasks whose tool chains cost the most. Worst case: -SACRIFICE_COUNT correct.
# Overridable per-image via env (Dockerfile ARG) without a code edit.
SACRIFICE_COUNT = int(os.environ.get("SACRIFICE_COUNT", "2"))
SACRIFICE_MAX_TOKENS = 200
SACRIFICE_INSTRUCTION = (
    "State your best final answer as concisely as possible. No derivation, no preamble."
)
# Most-expensive-first. With tools disabled, per-task cost is driven by the
# completion ceiling, so the code categories (700-token caps for full
# programs) are now the costliest, then Math/Logic.
EXPENSIVE_CATEGORY_ORDER = (CODE_DEBUG, CODE_GEN, MATH, LOGIC)

# Ultra-diet ceilings: sized to the minimum a *correct* answer plausibly needs
# per category (code must fit whole programs; a sentiment label doesn't), not
# to what the model would like to say. Truncation risk is absorbed by the
# answer-first ordering in CATEGORY_INSTRUCTIONS -- if the tail gets cut, it's
# explanation, not the answer. Baseline slack: the low-effort build scored
# 100% (19/19), so the accuracy gate can absorb a few losses from this.
CATEGORY_MAX_TOKENS = {
    1: 350,
    2: 400,
    3: 120,
    4: 250,
    5: 200,
    6: 700,
    7: 350,
    8: 700,
}

# "low" for the reasoning-heavy categories: the low-effort build scored a
# perfect 19/19 on the real platform (vs 18/19 at "medium"), so "low" is both
# the accuracy-proven and cheaper choice. 1/3/4/5 stay "none" since they don't
# need multi-step reasoning to answer correctly regardless of prompt difficulty.
CATEGORY_REASONING_EFFORT = {
    1: "none",
    2: "low",
    3: "none",
    4: "none",
    5: "none",
    6: "low",
    7: "low",
    8: "low",
}

# Terseness is asked for carefully: an earlier blanket "no chain-of-thought"
# rule conflicted with prompts that explicitly demand derivations/long output
# and caused meta-commentary paralysis. This version only bans *unrequested*
# padding -- anything the prompt explicitly requires is still fair game.
SYSTEM_PREFIX = (
    "Answer accurately and follow the prompt's explicit requirements. Be brief: "
    "add nothing the prompt didn't ask for. "
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


# Ultra-diet: tools disabled entirely. Every tool round re-sends the whole
# conversation plus the tool schema, adding ~700-1500 tokens per tool-using
# task -- the single biggest remaining cost after the caps were tightened.
# The accuracy risk is real (tools drove the 63%->94.7% jump) but the current
# baseline is 19/19, so there is measured slack to spend; if accuracy falls
# below the gate, restore {MATH, CODE_DEBUG, LOGIC, CODE_GEN} first.
TOOL_CATEGORIES = set()
TOOL_EXECUTION_TIMEOUT_S = 10
# Each tool round re-sends the entire conversation (prompt + all prior tool
# results), so rounds are the single most expensive knob in the pipeline.
# 2 keeps the one verify-and-answer cycle that drove the accuracy jump while
# cutting the long exploratory chains observed at 3.
MAX_TOOL_ROUNDS = 2

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code in a sandbox, returning stdout. Use for exact arithmetic, "
                "brute-force search, or testing code. print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to run."}},
                "required": ["code"],
            },
        },
    }
]


def _run_python_tool_sync(code: str) -> str:
    """Runs model-generated code in a separate subprocess (never exec()/eval()
    in-process) with a hard timeout and output cap, so a hang, infinite loop,
    or runaway output can't stall the task or blow up the response.

    Blocking by design (subprocess.run) -- must be called via asyncio.to_thread,
    never awaited directly, or it freezes the whole event loop and stalls every
    other concurrent task for the duration of the call."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=TOOL_EXECUTION_TIMEOUT_S,
        )
        output = proc.stdout if proc.returncode == 0 else f"{proc.stdout}\nSTDERR: {proc.stderr}"
        # Tool output is re-sent to the model with every subsequent round, so
        # this cap is paid multiple times -- keep it tight.
        return output.strip()[:1000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(execution timed out after {TOOL_EXECUTION_TIMEOUT_S}s)"
    except Exception as exc:
        return f"(execution failed: {type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# Local model (zero-token path). Officially sanctioned by the organizers:
# answers produced by a model running inside the container count fully toward
# accuracy and cost zero score-tokens (only FIREWORKS_BASE_URL traffic is
# counted). Grading env is 4 GB RAM / 2 vCPU, no GPU, nothing pre-installed,
# so a ~2 GB 4-bit 3B GGUF via llama.cpp is the practical ceiling.
#
# Routing: LOCAL_CATEGORIES (default Sentiment+NER -- label/extraction tasks a
# 3B handles credibly; Factual/Math/Code stay on Fireworks where a small model
# would fail) plus any sacrificed task (a free local attempt strictly beats a
# 200-token Fireworks guess). Every local path falls back to the normal
# Fireworks path on any failure, so the worst case equals not having the model.
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/local-model.gguf")
LOCAL_CATEGORIES = {
    int(x) for x in os.environ.get("LOCAL_CATEGORIES", "3,5").split(",") if x.strip().isdigit()
}
# 2 vCPU means local generation is slow (~5-10 tok/s); cap sacrificed tasks'
# local attempts so two of them can't eat the 10-minute container budget.
LOCAL_SACRIFICE_MAX_TOKENS = 400

_local_llm = None


def load_local_model() -> bool:
    """Load the bundled GGUF via llama-cpp-python. Failure is non-fatal:
    everything just routes to Fireworks as before."""
    global _local_llm
    try:
        from llama_cpp import Llama
        _local_llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=max(1, os.cpu_count() or 2),
            verbose=False,
        )
        print("[info] local model loaded", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[warn] local model unavailable, using Fireworks for everything: {exc}", file=sys.stderr)
        return False


def _local_complete_sync(system_prompt: str, prompt: str, max_tokens: int) -> str:
    """Blocking llama.cpp generation -- call via asyncio.to_thread only."""
    resp = _local_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


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


async def _complete(
    client: AsyncOpenAI, model: str, messages: list, max_tokens: int,
    reasoning_effort: str | None, tools: list | None = None,
):
    """Chat completion with a fallback for reasoning_effort support, which varies
    by model: some reject a given value, others reject the field outright. Rather
    than pattern-matching specific error wording (which we can only ever observe
    from models we've actually called, and the launch-day roster may differ from
    those), any API-level rejection while reasoning_effort was set retries once
    with it stripped entirely -- a model that doesn't understand the field can
    still answer the question."""
    extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    tool_kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    try:
        return await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0,
            extra_body=extra_body, **tool_kwargs,
        )
    except APIStatusError:
        if not extra_body:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0, **tool_kwargs,
        )


async def _answer_with_tools(
    client: AsyncOpenAI, model: str, messages: list, max_tokens: int,
    reasoning_effort: str | None, tools: list | None,
) -> str:
    """Runs request/tool-execution cycles (feeding each tool result back to the
    model) until it returns a plain answer or MAX_TOOL_ROUNDS is used up. Each
    individual API call still gets its own independent REQUEST_TIMEOUT_S budget
    -- this chains multiple such calls together, it never pools or extends any
    single call's time limit."""
    last_partial = ""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = await asyncio.wait_for(
            _complete(client, model, messages, max_tokens, reasoning_effort, tools=tools),
            timeout=REQUEST_TIMEOUT_S,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            content = (msg.content or "").strip()
            # Some models' serving setup doesn't reliably parse their own
            # tool-call format into the API's structured tool_calls field --
            # the attempt leaks through as raw text instead (e.g. a literal
            # "<tool_call>" tag) rather than a real answer. Detecting this
            # generic protocol artifact (not any specific question's content)
            # and forcing a plain retry beats returning garbage as the answer.
            if "<tool_call" in content or content.startswith("run_python("):
                resp = await asyncio.wait_for(
                    _complete(client, model, messages, max_tokens, reasoning_effort, tools=None),
                    timeout=REQUEST_TIMEOUT_S,
                )
                return (resp.choices[0].message.content or "").strip()
            return content

        # Models often write real analysis in content alongside a tool call;
        # keep the latest such text as a last-resort answer if the final
        # forced call comes back empty.
        if (msg.content or "").strip():
            last_partial = msg.content.strip()
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = await asyncio.to_thread(_run_python_tool_sync, args.get("code", ""))
            except Exception as exc:
                result = f"(tool call failed: {type(exc).__name__}: {exc})"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Ran out of tool rounds -- force a final answer with no tools offered.
    # reasoning_effort deliberately dropped here: observed a model spend this
    # call's entire max_tokens on hidden reasoning and return empty visible
    # content; with no effort requested the budget goes to the answer itself.
    resp = await asyncio.wait_for(
        _complete(client, model, messages, max_tokens, None, tools=None),
        timeout=REQUEST_TIMEOUT_S,
    )
    final = (resp.choices[0].message.content or "").strip()
    # An empty final beat nothing only if we truly have nothing -- prefer the
    # best partial text the model produced during its tool rounds.
    return final or last_partial


def pick_sacrifices(tasks: list[dict]) -> set:
    """Pick the SACRIFICE_COUNT tasks expected to burn the most tokens if run
    through the full pipeline: most expensive category first (per
    EXPENSIVE_CATEGORY_ORDER), longest prompt first within a category. Purely
    structural signals -- category and length -- never question content."""
    if SACRIFICE_COUNT <= 0:
        return set()

    def cost_key(t):
        prompt = normalize_prompt(t["prompt"])
        category = classify_category(prompt)
        try:
            rank = EXPENSIVE_CATEGORY_ORDER.index(category)
        except ValueError:
            rank = len(EXPENSIVE_CATEGORY_ORDER)
        return (rank, -len(prompt))

    ordered = sorted(tasks, key=cost_key)
    return {t["task_id"] for t in ordered[:SACRIFICE_COUNT]}


# Local generation is CPU-bound on 2 vCPU -- one at a time; parallel local
# calls would just thrash each other. Fireworks calls (network-bound) still
# run concurrently alongside the local queue.
_local_sem = asyncio.Semaphore(1)


async def answer_task(
    client: AsyncOpenAI, task: dict, tiers: dict, sem: asyncio.Semaphore, sacrificed: bool = False,
) -> dict:
    task_id = task["task_id"]
    prompt = normalize_prompt(task["prompt"])
    category = classify_category(prompt)
    model = model_for_category(category, tiers)

    if _local_llm is not None and (sacrificed or category in LOCAL_CATEGORIES):
        try:
            local_max_tokens = CATEGORY_MAX_TOKENS[category]
            if sacrificed:
                local_max_tokens = min(local_max_tokens, LOCAL_SACRIFICE_MAX_TOKENS)
            async with _local_sem:
                answer = await asyncio.to_thread(
                    _local_complete_sync,
                    SYSTEM_PREFIX + CATEGORY_INSTRUCTIONS[category],
                    prompt,
                    local_max_tokens,
                )
            if answer:
                return {"task_id": task_id, "answer": answer}
            print(f"[warn] local model returned empty for {task_id}, falling back to Fireworks", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] local model failed for {task_id} ({exc}), falling back to Fireworks", file=sys.stderr)

    if sacrificed:
        system_prompt = SYSTEM_PREFIX + SACRIFICE_INSTRUCTION
        max_tokens = SACRIFICE_MAX_TOKENS
        reasoning_effort = "none"
        tools = None
    else:
        system_prompt = SYSTEM_PREFIX + CATEGORY_INSTRUCTIONS[category]
        max_tokens = CATEGORY_MAX_TOKENS[category]
        reasoning_effort = CATEGORY_REASONING_EFFORT[category]
        tools = TOOL_SPEC if category in TOOL_CATEGORIES else None

    last_error = None
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                answer = await _answer_with_tools(client, model, messages, max_tokens, reasoning_effort, tools)
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

    sacrificed_ids = pick_sacrifices(tasks)
    if sacrificed_ids:
        print(f"[info] sacrificing (cheap single-shot): {sorted(sacrificed_ids)}", file=sys.stderr)

    individual_results = await asyncio.gather(*(
        answer_task(client, t, tiers, sem, sacrificed=t["task_id"] in sacrificed_ids) for t in individual_tasks
    ))
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
        results.extend(await asyncio.gather(*(
            answer_task(client, t, tiers, sem, sacrificed=t["task_id"] in sacrificed_ids) for t in fallback_needed
        )))

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

    await asyncio.to_thread(load_local_model)

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
