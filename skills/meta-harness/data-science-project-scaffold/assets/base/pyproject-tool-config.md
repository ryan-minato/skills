[dependency-groups]
dev = [
    "pre-commit",
    "pytest",
    "ruff",
]

[tool.ruff]
line-length = 100

[tool.ruff.format]
docstring-code-format = true

[tool.ruff.lint]
extend-select = [
    "I",
    "N",
    "W",
]

[tool.pytest.ini_options]
addopts = "-ra -q"
testpaths = ["tests"]
markers = [
    "slow: manual tests requiring large data, model weights, accelerators, network access, or long execution",
]
