FROM nvidia/cuda:<tag — a runtime variant unless something compiles CUDA code>

# uv manages Python and packages inside the image; it downloads a managed
# interpreter, so the base image needs none.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_TORCH_BACKEND=<backend matching the base image's CUDA/ROCm>

WORKDIR /app

# Dependency layer first: rebuilt only when the lockfile changes.
COPY requirements.txt .
RUN uv venv --python <version> && uv pip sync requirements.txt
ENV PATH="/app/.venv/bin:${PATH}"

# The environment is baked; data, outputs, and secrets never are.
COPY . .

CMD ["python", "train.py"]
