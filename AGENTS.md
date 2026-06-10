# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Run Commands

```bash
# Regular Python scripts
uv run src/<script.py>

# Streamlit apps (requires CLI, not direct execution)
uv run streamlit run src/<app.py>

# Add dependencies
uv add <package>
```

## Critical Import Pattern

Scripts in `src/algo/` and `src/streamlit/` MUST manually add `src/` to sys.path before importing from `utils/`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input, save_output
```

This is required because the project lacks a proper package structure with `__init__.py` at root.

## I/O Utilities (`utils/io.py`)

- **Only `.txt` files are supported** - `.csv`, `.json`, `.pdf` will raise `NotImplementedError`
- `save_output()` automatically prefixes filenames with `YYYYMMDD_HHMMSS_`
- Output files accumulate in `/output` (not auto-cleaned)
- `ROOT` path is calculated as `Path(__file__).resolve().parents[2]` from `utils/io.py` (assumes fixed depth)

## Project Context

- Python 3.13 with `uv` package manager
- No test framework configured (test_*.py files are manual test scripts, not unit tests)
- No linter/formatter configured in pyproject.toml
- Documentation in French (comments, README, docs)