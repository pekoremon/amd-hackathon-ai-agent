FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Local zero-token model (organizer-sanctioned): llama.cpp runtime via
# prebuilt CPU wheel, plus a ~1.9GB 4-bit Qwen2.5-3B GGUF baked into the
# image (grading env has no runtime pre-installed and 4GB RAM / 2 vCPU;
# a 3B Q4 fits comfortably). If either layer breaks at runtime, main.py
# degrades gracefully to Fireworks-only.
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Default: Qwen2.5-3B (best quality/RAM fit). Override MODEL_URL for the
# Gemma-family hedge variant (same family as the allowed Fireworks Gemma IDs).
ARG MODEL_URL=https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf
ADD ${MODEL_URL} /app/models/local-model.gguf

COPY main.py classify.py ./

# How many expected-most-expensive tasks get a cheap answer attempt instead
# of the full pipeline (see pick_sacrifices in main.py) -- the local model
# handles these for free when loaded. Build with --build-arg SACRIFICE=0 to
# disable. LOCAL_CATS picks which categories route to the local model.
ARG SACRIFICE=2
ARG LOCAL_CATS=3,5
ENV SACRIFICE_COUNT=${SACRIFICE}
ENV LOCAL_CATEGORIES=${LOCAL_CATS}

CMD ["python", "main.py"]
