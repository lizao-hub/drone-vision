# Worker image: segment the viaduct deck, track vehicles, write an annotated mp4.
#
# CPU-friendly: torch comes from PyPI and runs on CPU when `--device cpu` is
# passed.  Weights, videos and output are NOT baked in; compose mounts them at
# /app/models, /app/videos and /app/output.
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
# NOTE: on linux the pinned torch wheel pulls its CUDA runtime (nvidia-*), so the
# image is large (~6 GB).  For a CPU-only image install the CPU wheels first
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.2
# and drop the nvidia-* lines from requirements.txt.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Keep the source tree at /app/src: WORKER_ROOT is derived from
# src/algorithm/runtime/factory.py and must resolve to /app, where models/ and
# videos/ are mounted.
COPY src ./src
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "drone_worker"]
CMD ["--help"]
