# Worker image: segment the viaduct deck, track vehicles, write an annotated mp4.
#
# GPU-only: the engine runs one process per model and pins each to a card, so
# there is no CPU path.  Weights, videos and output are NOT baked in; compose
# mounts them at /app/models, /app/videos and /app/output.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# opencv-python (non-headless) links against libGL/libGLib even for file I/O.
# ffmpeg is the RTMP sink of the live relay (`python -m drone_worker relay`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt is generated from uv.lock:
#   uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.txt
#
# NOTE: the Linux wheels on PyPI bundle CUDA 13, which needs driver >= 580.  For
# older drivers (the GPU box runs 535 = CUDA 12.2) install the torch stack from
# the CUDA 12.6 wheel index first -- 12.x minor versions run on any 12.x driver --
# and skip the CUDA-13 lines of the lock export.  Override the args for another
# CUDA line or a CPU-only image, e.g.
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#                --build-arg TORCH_VERSION=2.14.0+cu121 \
#                --build-arg TORCHVISION_VERSION=0.29.0+cu121 .
# The image stays ~6 GB either way: the CUDA runtime ships as pip wheels.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.14.0+cu126
ARG TORCHVISION_VERSION=0.29.0+cu126

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" --extra-index-url "$PIP_INDEX_URL" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
 && grep -vE '^(torch|torchvision|triton|cuda-(bindings|pathfinder|toolkit)|nvidia-(cublas|cuda-cupti|cuda-nvrtc|cuda-runtime|cudnn-cu13|cufft|cufile|curand|cusolver|cusparse|cusparselt-cu13|nccl-cu13|nvjitlink|nvshmem-cu13|nvtx))==' \
        requirements.txt > /tmp/requirements.base.txt \
 && pip install --no-cache-dir -r /tmp/requirements.base.txt

# Keep the source tree at /app/src: WORKER_ROOT is derived from
# src/algorithm/runtime/factory.py and must resolve to /app, where models/ and
# videos/ are mounted.
COPY src ./src
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "drone_worker"]
CMD ["--help"]
