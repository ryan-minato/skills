# syntax=docker/dockerfile:1
# Three stages, one identity: the image digest (not this file, not a tag)
# is what a run records as its environment. Enumerate the base tag live.
# `.dockerignore` (from the dockerignore asset) keeps data, outputs, secrets,
# caches, and `.git` out of every build context.

FROM nvidia/cuda:<tag — a runtime variant unless something compiles CUDA code> AS environment
# uv manages Python and packages inside the image; it downloads a managed
# interpreter, so the base image needs none.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# The environment lives outside /app so a source bind mount over /app
# (compose) can never hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
# Dependency layer first: rebuilt only when the lock changes. The torch
# index is routed in pyproject.toml, so a frozen sync reproduces the
# environment exactly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# requirements carrier instead:
#   ENV UV_TORCH_BACKEND=<backend matching the base image>
#   COPY requirements.txt .
#   RUN uv venv --python <version> /opt/venv && uv pip sync --python /opt/venv/bin/python requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

FROM environment AS runtime
# Day-to-day training: the source is mounted (compose) or copied at run
# time; data, outputs, and secrets never enter the image.
CMD ["python", "train.py"]

FROM environment AS sealed
# A run reproducible from the image alone: the source tree is inside.
# `.git` is not (see .dockerignore), so the commit is stamped in: the
# manifest reads GIT_COMMIT when no repository is present, and the revision
# label ties the image to it. `just docker-build sealed` refuses a dirty tree
# so the stamp is the code inside.
ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT
LABEL org.opencontainers.image.revision=$GIT_COMMIT
COPY . .
CMD ["python", "train.py"]
