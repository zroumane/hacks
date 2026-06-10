# Advanced Mode Rules (Non-Obvious Only)

## Critical Import Pattern

Scripts in `src/algo/` and `src/streamlit/` MUST manually add `src/` to sys.path:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input, save_output
```

Required because project lacks `__init__.py` at root.

## I/O Utilities Constraints

- `utils/io.py` only supports `.txt` files - `.csv`, `.json`, `.pdf` raise `NotImplementedError`
- `save_output()` auto-prefixes with `YYYYMMDD_HHMMSS_` (not configurable)
- Output files accumulate in `/output` (manual cleanup required)
- `ROOT` path assumes fixed depth: `Path(__file__).resolve().parents[2]` from `utils/io.py`

## Execution Methods

- Regular scripts: `uv run src/<script.py>`
- Streamlit apps: `uv run streamlit run src/<app.py>` (CLI required, not direct execution)

## Project Constraints

- No test framework (test_*.py are manual scripts, not unit tests)
- No linter/formatter configured
- Documentation in French

## Advanced Mode Access

- Has access to MCP tools and Browser capabilities