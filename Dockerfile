# syntax=docker/dockerfile:1

# ---- models (optional): local Thai TTS (~330 MB) + Nemotron ASR (~700 MB) ----
# only downloaded with LOCAL_MODELS=1, for STT_PROVIDER/TTS_PROVIDER=local
FROM python:3.12-slim AS models
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG LOCAL_MODELS=0
ARG FASTTHAIG2P_URL=https://github.com/awslabs/FastThaiG2P/releases/download/v0.3.0
ARG NEMOTRON_URL=https://huggingface.co/wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx/resolve/main

RUN set -eux; \
    mkdir -p /models; \
    [ "$LOCAL_MODELS" = "1" ] || exit 0; \
    get() { curl -fL --retry 5 --retry-all-errors -C - --speed-limit 50000 --speed-time 30 -o "$2" "$1"; }; \
    d=/models/fastthaig2p; mkdir -p "$d"; \
    for f in config.json thai_som.npy kokoro_thai.onnx; do get "$FASTTHAIG2P_URL/$f" "$d/$f"; done; \
    d=/models/pythaiasr/typhoon-asr-streaming-nemotron-0.6b-int4-onnx; mkdir -p "$d"; \
    for f in tokens.txt nemotron_onnx_config.json joint.onnx decoder.onnx encoder.onnx; do get "$NEMOTRON_URL/$f" "$d/$f"; done

# ---- app ----
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    FASTTHAIG2P_HOME=/models/fastthaig2p \
    PYTHAIASR_DATA_DIR=/models/pythaiasr \
    PATH="/app/.venv/bin:$PATH"

COPY --from=models /models /models

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

COPY . .
RUN python agent.py download-files

CMD ["python", "agent.py", "start"]
