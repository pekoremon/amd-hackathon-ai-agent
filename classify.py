"""Lightweight, keyword-based routing helpers.

We don't call the LLM to classify tasks (that would burn tokens we're scored on).
A regex/keyword guess is enough here: it only decides which model tier and which
system-prompt template to use, not the final answer. A misclassification costs a
few accuracy points on the prompt phrasing, not a hard failure.
"""

import re

FACTUAL = 1
MATH = 2
SENTIMENT = 3
SUMMARIZATION = 4
NER = 5
CODE_DEBUG = 6
LOGIC = 7
CODE_GEN = 8

# Categories that benefit from the general "strong" model.
STRONG_CATEGORIES = {MATH, LOGIC}
# Categories best handled by a code-specialist model, if one is available.
CODE_CATEGORIES = {CODE_DEBUG, CODE_GEN}

_CODE_HINT_RE = re.compile(r"```|\bdef \w+\(|\bfunction \w+\(|\bclass \w+\b")
_DEBUG_WORDS_RE = re.compile(r"\b(bug|fix|debug|incorrect|not working|broken|wrong output|error)\b")
_CODEGEN_RE = re.compile(r"\b(write a function|implement a function|write code|write a program|create a function|write a python|write a class)\b")
_SUMMARY_RE = re.compile(r"summar|condense|tl;dr")
_SENTIMENT_RE = re.compile(r"sentiment|positive or negative|classify.*sentiment")

_NER_PHRASE_RES = (
    re.compile(r"named entit"),
    re.compile(r"extract.*entit"),
    re.compile(r"identify (the )?(person|organi[sz]ation|location|date)s?"),
    re.compile(r"pull out.*(person|entit)"),
    re.compile(r"list (all|every).*(person|organi[sz]ation|location|date)"),
)
# A prompt naming 2+ distinct entity types (person/org/location/date) reads as
# NER regardless of which verb it uses ("pull out", "tag", "find", ...).
_ENTITY_TYPE_RES = tuple(re.compile(w) for w in ("person", "organi[sz]ation", "location", r"\bdate\b"))

_LOGIC_PHRASE_RES = (
    re.compile(r"\bpuzzle\b"),
    re.compile(r"exactly one"),
    re.compile(r"each (of|person|room|coworker|friend)\b"),
    re.compile(r"who (is|owns|lives|drives|sits)\b"),
    re.compile(r"sits where"),
    re.compile(r"\bseated\b"),
    re.compile(r"\badjacent\b"),
    re.compile(r"immediately (to the )?(left|right)"),
    re.compile(r"directly (to the )?(left|right)"),
    re.compile(r"\bdeduce\b"),
    re.compile(r"which (one|room|desk|seat|position)"),
    re.compile(r"constraints?\b"),
    re.compile(r"figure out (who|which|where)"),
)

_MATH_RE = re.compile(r"\d.*%|percent|average|total|how many|calculate|sum of|product of|difference of|multiply|divide|ratio|profit|discount|interest")


def _looks_like_ner(p: str) -> bool:
    if any(r.search(p) for r in _NER_PHRASE_RES):
        return True
    return sum(1 for r in _ENTITY_TYPE_RES if r.search(p)) >= 2


def _looks_like_logic(p: str) -> bool:
    return any(r.search(p) for r in _LOGIC_PHRASE_RES)


def classify_category(prompt: str) -> int:
    p = prompt.lower()

    if _CODE_HINT_RE.search(prompt) and _DEBUG_WORDS_RE.search(p):
        return CODE_DEBUG
    if _CODEGEN_RE.search(p):
        return CODE_GEN
    if _SUMMARY_RE.search(p):
        return SUMMARIZATION
    if _SENTIMENT_RE.search(p):
        return SENTIMENT
    if _looks_like_ner(p):
        return NER
    if _looks_like_logic(p) and "?" in prompt:
        return LOGIC
    if _MATH_RE.search(p):
        return MATH
    return FACTUAL


_QUANT_TAGS = ("nvfp4", "fp4", "fp8", "int4", "int8", "awq", "gptq", "w4a16", "w8a8")


def _parse_active_token(token: str):
    """Parse an MoE active-param token like "a4b" -> 4.0, else None."""
    if len(token) < 3 or token[0] != "a" or token[-1] != "b":
        return None
    digits = token[1:-1]
    return float(digits) if digits.isdigit() else None


def _parse_size_token(token: str):
    """Parse a plain size token like "31b" or "7.5b" -> that value, else None."""
    if len(token) < 2 or token[-1] != "b":
        return None
    digits = token[:-1]
    if digits.isdigit():
        return float(digits)
    whole, _, frac = digits.partition(".")
    if whole.isdigit() and frac.isdigit():
        return float(digits)
    return None


def _effective_size(model_id: str):
    """Rough inference-cost proxy parsed from the model id, or None if unknown.

    Tokenizes on "-"/"_" (e.g. "gemma-4-26b-a4b-it" -> ["gemma", "4", "26b", "a4b", "it"]).
    - an MoE active-param token ("a4b") wins over a plain size token ("26b"), since
      active params (not total params) drive latency/cost.
    - a quantization tag (nvfp4, fp8, int4, ...) halves the effective size, since
      quantized builds are cheaper/faster than full-precision same-size peers.
    """
    m = model_id.lower()
    tokens = m.replace("_", "-").split("-")

    size = None
    active = None
    for token in tokens:
        if active is None:
            active = _parse_active_token(token)
        if size is None:
            size = _parse_size_token(token)

    effective = active if active is not None else size
    if effective is None:
        return None

    if any(tag in m for tag in _QUANT_TAGS):
        effective /= 2
    return effective


def _is_code_specialist(model_id: str) -> bool:
    m = model_id.lower()
    return "code" in m or "coder" in m


def pick_model_tiers(allowed_models: list[str], overrides: dict | None = None) -> dict:
    """Return {'strong': ..., 'cheap': ..., 'code': ...} model ids from ALLOWED_MODELS.

    Ranking by embedded size only works when naming is consistent (e.g. Gemma's
    "31b"/"a4b" tags). Mixed rosters mostly don't embed a param count at all —
    an unsized, non-code model (e.g. "minimax-m3") is treated as the general
    "strong" pick, since proprietary flagship models typically don't publish a
    param count the way open-weight families do. Sized models are only used to
    rank "cheap": with 2+ of them we trust the ranking; with exactly one and no
    unsized candidate to compare it against, we still use it (it's the only
    option); but with exactly one sized model *and* unsized ones also present,
    we don't guess which is actually cheaper — that single stray number isn't a
    reliable enough signal (e.g. a lone "120b" among five otherwise-unsized ids
    could easily be the most expensive model, not the cheapest).

    `overrides` (from optional env vars) lets you pin any tier explicitly once you
    know the real characteristics of the launch-day roster; each override must be
    a member of allowed_models.
    """
    code_models = [m for m in allowed_models if _is_code_specialist(m)]
    sized = [(m, _effective_size(m)) for m in allowed_models]
    known = [(m, s) for m, s in sized if s is not None]
    unsized_general = [m for m, s in sized if s is None and m not in code_models]

    if unsized_general:
        strong = unsized_general[0]
    elif known:
        strong = max(known, key=lambda x: x[1])[0]
    else:
        strong = allowed_models[0]

    if len(known) >= 2:
        cheap = min(known, key=lambda x: x[1])[0]
    elif len(known) == 1 and not unsized_general:
        cheap = known[0][0]
    else:
        cheap = strong

    code = code_models[0] if code_models else strong

    tiers = {"strong": strong, "cheap": cheap, "code": code}
    for tier, value in (overrides or {}).items():
        if value is None:
            continue
        if value not in allowed_models:
            raise ValueError(f"model tier override {tier}={value!r} is not in ALLOWED_MODELS")
        tiers[tier] = value

    return tiers


def model_for_category(category: int, tiers: dict) -> str:
    if category in CODE_CATEGORIES:
        return tiers["code"]
    if category in STRONG_CATEGORIES:
        return tiers["strong"]
    return tiers["cheap"]
