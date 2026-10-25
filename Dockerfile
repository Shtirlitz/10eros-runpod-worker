# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with ComfyUI, ComfyUI-LTXVideo node, and runtime deps
FROM ${BASE_IMAGE} AS base

# Build arguments
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL
ARG PYTORCH_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# System dependencies. --no-install-recommends keeps the image lean (no
# pocketsphinx, va-driver-all, vdpau-driver-all, etc. that ffmpeg's Recommends
# list otherwise drags in) AND sidesteps Ubuntu mirror flakes for those
# unneeded packages. Acquire::Retries handles transient mirror 5xx/timeouts.
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install uv and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# All subsequent python/pip calls use the venv
ENV PATH="/opt/venv/bin:${PATH}"

# comfy-cli + base pip deps
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI (with optional PyTorch pin for specific CUDA versions)
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi && \
    if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ] && [ -n "${PYTORCH_VERSION}" ]; then \
      uv pip install --force-reinstall torch==${PYTORCH_VERSION} torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    elif [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi && \
    rm -rf /root/.cache/pip /root/.cache/uv /root/.cache/comfy-cli /tmp/* && \
    uv cache clean

WORKDIR /comfyui

# comfy-cli's `install` doesn't always pull the full ComfyUI requirements.txt,
# so newer deps (e.g. sqlalchemy/alembic/aiosqlite for the asset DB) go missing
# and ComfyUI crashes on startup. Install them explicitly from ComfyUI's own
# requirements file.
RUN if [ -f /comfyui/requirements.txt ]; then \
        /opt/venv/bin/pip install -q --root-user-action=ignore -r /comfyui/requirements.txt; \
    fi && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    uv cache clean

# Network-volume model-path config
ADD src/extra_model_paths.yaml ./

# Install the ComfyUI-LTXVideo custom node (LTX 2.3 / 10Eros use the same nodes).
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo ComfyUI-LTXVideo && \
    if [ -f ComfyUI-LTXVideo/requirements.txt ]; then \
        /opt/venv/bin/pip install -q --root-user-action=ignore -r ComfyUI-LTXVideo/requirements.txt; \
    fi && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    uv cache clean

WORKDIR /

# RunPod handler runtime deps (boto3 for R2 input downloads + output uploads)
RUN uv pip install runpod~=1.7.12 requests websocket-client boto3

# Application code
ADD src/start.sh src/network_volume.py handler.py test_input_fp8.json test_input_bf16.json ./
RUN chmod +x /start.sh

# Utility scripts
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-node-install /usr/local/bin/comfy-manager-set-mode

ENV PIP_NO_INPUT=1

CMD ["/start.sh"]

# Stage 2: Final image — download 10Eros + LTX 2.3 ancillary models.
# MODEL_VARIANT picks fp8 (10Eros v1.5 learned fp8 by LokkenJP, 30.2 GB, default) or bf16
# (TenStrip v1.5 original, 46.1 GB). v1.5 is tuned for the DMD distill LoRA (TenStrip/LTX2.3_DMD_Lora).
FROM base AS final

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_VARIANT=fp8

WORKDIR /comfyui

# Validate MODEL_VARIANT up front (fail fast on typos)
RUN test "$MODEL_VARIANT" = "fp8" -o "$MODEL_VARIANT" = "bf16" || \
    (echo "ERROR: MODEL_VARIANT must be 'fp8' or 'bf16', got '$MODEL_VARIANT'" >&2 && exit 1)

# Create the model directory layout the workflow expects
RUN mkdir -p \
    models/checkpoints \
    models/loras \
    models/text_encoders \
    models/latent_upscale_models

# All model repos below are public for the typical case. Lightricks/LTX-2.3 may
# gate occasionally; pass HUGGINGFACE_ACCESS_TOKEN at build time if needed.
# One RUN per download: RunPod GitHub builds time out at ~30 min and cache layers, so a
# retried build resumes from the last finished download instead of starting over.
RUN uv pip install "huggingface_hub[hf_xet]"

# 10Eros v1.5 checkpoint (bundles CLIP + VAEs), sha256-checked against the converter's sidecar.
RUN HF_TOKEN="${HUGGINGFACE_ACCESS_TOKEN}" MODEL_VARIANT="${MODEL_VARIANT}" python -c "\
import os, hashlib; \
from huggingface_hub import hf_hub_download; \
token = os.environ.get('HF_TOKEN') or None; \
variant = os.environ.get('MODEL_VARIANT', 'fp8'); \
repo, ckpt, sha = (('LokkenJP/10EROS_1.5_fp8_exp_learned', '10Eros_v1.5_fp8mixed_experimental_learned.safetensors', '192c8f544fe1694bd974e30217ca0f006be9d629b502313921989ee101bfe483') if variant == 'fp8' else ('TenStrip/LTX2.3-10Eros', '10Eros_v1.5_bf16.safetensors', '')); \
path = hf_hub_download(repo_id=repo, filename=ckpt, local_dir='/comfyui/models/checkpoints', token=token); \
h = hashlib.sha256(); f = open(path, 'rb'); [h.update(b) for b in iter(lambda: f.read(1 << 24), b'')]; f.close(); \
assert not sha or h.hexdigest() == sha, 'checkpoint sha256 mismatch: ' + h.hexdigest(); \
print('checkpoint', ckpt, h.hexdigest()); \
" && rm -rf /root/.cache/huggingface /comfyui/models/checkpoints/.cache

# Text encoder (Gemma 3 12B fp4).
RUN HF_TOKEN="${HUGGINGFACE_ACCESS_TOKEN}" python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Comfy-Org/ltx-2', filename='split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors', local_dir='/tmp/te', token=os.environ.get('HF_TOKEN') or None); \
shutil.move('/tmp/te/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors', '/comfyui/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors'); \
" && rm -rf /root/.cache/huggingface /tmp/*

# DMD distill LoRA (v1.5 is tuned for it), the older cond-safe distilled LoRA, spatial upscaler x2 1.1
# (x2 1.0 was removed from Lightricks/LTX-2.3).
RUN HF_TOKEN="${HUGGINGFACE_ACCESS_TOKEN}" python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
token = os.environ.get('HF_TOKEN') or None; \
hf_hub_download(repo_id='TenStrip/LTX2.3_DMD_Lora', filename='LTX2.3_DMD_hybrid_v2.safetensors', local_dir='/comfyui/models/loras', token=token); \
hf_hub_download(repo_id='SulphurAI/Sulphur-2-base', filename='distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors', local_dir='/tmp/sl', token=token); \
shutil.move('/tmp/sl/distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors', '/comfyui/models/loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors'); \
hf_hub_download(repo_id='Lightricks/LTX-2.3', filename='ltx-2.3-spatial-upscaler-x2-1.1.safetensors', local_dir='/comfyui/models/latent_upscale_models', token=token); \
" && rm -rf /root/.cache/huggingface /tmp/* /comfyui/models/loras/.cache /comfyui/models/latent_upscale_models/.cache && \
    uv cache clean

# C compiler for triton: current ComfyUI JIT-compiles kernels on text encode (fp4 Gemma) and fails with
# 'Failed to find C compiler' on the bare CUDA runtime base. Kept as the LAST layer so the model
# download layers above stay cached.
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 gcc libc6-dev python3.12-dev && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc

# Surface the baked variant to the running container so logs/diagnostics can report it.
ENV MODEL_VARIANT=${MODEL_VARIANT}
