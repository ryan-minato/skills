# syntax=docker/dockerfile:1
# Three stages, one identity: the image digest (not this file, not a tag)
# is what a run records as its environment. Enumerate the base tag live.

FROM nvidia/cuda:<tag — a runtime variant unless something compiles CUDA code> AS environment
# uv manages Python and packages inside the image; it downloads a managed
# interpreter, so the base image needs none.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
# Dependency layer first: rebuilt only when the lock changes. The torch
# index is routed in pyproject.toml, so a frozen sync reproduces the
# environment exactly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# requirements carrier instead:
#   ENV UV_TORCH_BACKEND=<backend matching the base image>
#   COPY requirements.txt .
#   RUN uv venv --python <version> && uv pip sync requirements.txt
ENV PATH="/app/.venv/bin:${PATH}"

FROM environment AS runtime
# Day-to-day training: the source is mounted (compose) or copied at run
# time; data, outputs, and secrets never enter the image.
CMD ["python", "train.py"]

FROM environment AS sealed
# A run reproducible from the image alone: the source tree is inside.
COPY . .
CMD ["python", "train.py"]
