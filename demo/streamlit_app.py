import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="AMD Hackathon — Track 1 Agent Demo", page_icon="🍌", layout="centered")

DATA_PATH = Path(__file__).parent / "demo_examples.json"
EXAMPLES = json.loads(DATA_PATH.read_text(encoding="utf-8"))

TIER_BY_CATEGORY = {
    1: "cheap", 2: "strong", 3: "cheap", 4: "cheap",
    5: "cheap", 6: "code", 7: "strong", 8: "code",
}
TIER_LABELS = {
    "strong": "🧠 Strong model",
    "cheap": "⚡ Cheap model",
    "code": "🛠️ Code-specialist model",
}

st.title("🍌 Track 1 Agent — Live Example Walkthrough")
st.write(
    "This agent answers benchmark tasks across 8 categories, routing each one to an "
    "appropriately sized Fireworks AI model. The examples below are **real prompts and "
    "real model answers** captured during development testing — not generated live, so "
    "this demo works without an API key."
)

st.divider()

labels = [f"{e['category_id']}. {e['category_name']}" for e in EXAMPLES]
choice = st.selectbox("Pick a task category to inspect:", labels)
example = EXAMPLES[labels.index(choice)]

tier = TIER_BY_CATEGORY[example["category_id"]]

col1, col2 = st.columns(2)
with col1:
    st.metric("Category", example["category_name"])
with col2:
    st.metric("Routed to", TIER_LABELS[tier])

st.subheader("Task prompt")
st.info(example["prompt"])

st.subheader("Agent's answer")
st.success(example["answer"])

st.caption(f"Tokens used for this answer: **{example['tokens']}** (task id: `{example['task_id']}`)")

st.divider()

st.subheader("All 8 categories at a glance")
st.bar_chart(
    {e["category_name"]: e["tokens"] for e in EXAMPLES},
    x_label="Category",
    y_label="Tokens used",
)
st.caption(
    "Each category is classified with zero extra token cost, then routed to a strong, "
    "cheap, or code-specialist model depending on what the task actually needs — "
    "reasoning depth and output limits are also tuned per category to avoid spending "
    "tokens on unnecessary chain-of-thought."
)
