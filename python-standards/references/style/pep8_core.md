# PEP 8 Core Standards (Summary)

## Layout & Whitespace
* **Indentation**: Strictly **4 spaces**. No tabs.
* **Line Length**: Max **79 chars** (code), **72 chars** (comments/docstrings).
    * *Modern Exception*: 88 chars (Black formatter default) or 100/120 is acceptable if configured in `pyproject.toml`.
* **Blank Lines**:
    * Top-level functions/classes: **2 blank lines** surrounding.
    * Methods inside classes: **1 blank line** surrounding.
* **Imports**:
    1.  Standard Library (`os`, `sys`)
    2.  Third Party (`numpy`, `requests`)
    3.  Local Application (`from . import models`)

## Naming Conventions
| Type | Convention | Example |
| :--- | :--- | :--- |
| **Function/Variable** | `snake_case` | `calculate_total`, `user_id` |
| **Class** | `CapWords` (PascalCase) | `DataProcessor`, `UserAccount` |
| **Constant** | `ALL_CAPS` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| **Protected Member** | `_leading_underscore` | `_internal_cache` |
| **Private Member** | `__double_underscore` | `__mangle_this` |

## Programming Recommendations
* Use `is` for `None` checks: `if x is not None:` (not `if x != None:`).
* Use `"".startswith()` and `"".endswith()` instead of string slicing.
* Context Managers: Always use `with open(...)` for file handling.
