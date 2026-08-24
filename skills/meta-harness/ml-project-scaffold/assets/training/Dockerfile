FROM nvidia/cuda:<tag — a runtime variant unless something compiles CUDA code>

# uv manages Python and packages inside the image; it downloads a managed
# interpreter, so the base image needs none.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer first: rebuilt only when the lockfile changes. The
# torch wheel index is already routed in pyproject.toml, so a frozen sync
# reproduces the training environment exactly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The environment is baked; data, outputs, and secrets never are.
COPY . .
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "train.py"]
