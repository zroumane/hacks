# Ask Mode Rules (Non-Obvious Only)

## Project Structure Context

- `src/algo/` contains data processing scripts (not algorithms in the CS sense)
- `src/streamlit/` contains UI apps for visualizing results from `src/algo/`
- `test_*.py` files are manual test scripts, NOT unit tests (no test framework)

## Documentation Locations

- Architecture diagrams in `Docs/Architecture.md` (Mermaid format)
- External resources and links in `Docs/Ressources.md` (French)
- Main README in French with setup instructions

## Non-Standard Patterns

- Scripts require manual `sys.path` manipulation to import from `utils/`
- No package structure (`__init__.py` missing at root)
- Output files are timestamped and accumulate (not cleaned automatically)

## Workflow Context

1. Scripts in `src/algo/` read from `/input`, process data, write to `/output` with timestamps
2. Streamlit apps in `src/streamlit/` read timestamped files from `/output` for visualization
3. Two-step workflow: process first, then visualize (not integrated)

## Language Note

- All documentation, comments, and variable names are in French
- Code follows French naming conventions