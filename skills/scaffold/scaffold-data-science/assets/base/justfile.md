set shell := ["bash", "-cu"]

package := "__PACKAGE_NAME__"

# Synchronize the locked environment and install both hook stages.
setup:
    uv sync
    uv run pre-commit install --hook-type pre-commit --hook-type pre-push

# Replace with one dependency-aware call per configured source.
download-data *args:
    uv run python -m {{package}}.workflows.download___SOURCE_NAME__ {{args}}

# Replace with the ordered production workflow entries.
pipeline *args:
    uv run python -m {{package}}.workflows.__PIPELINE_STEP__ {{args}}

test:
    uv run pytest -m "not slow"

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run pytest -m "not slow"

# Replace with the selected Markdown validation or PDF build command.
report:
    __REPORT_COMMAND__

# Review small staged diffs directly; record programmatic sensitivity scans for larger ones.
safe-to-commit: check
    uv run pre-commit run --all-files
    git diff --cached --check

# Replace the final line with the current full-history Gitleaks command.
safe-to-push: check
    uv run pre-commit run --all-files
    __FULL_HISTORY_GITLEAKS_COMMAND__
