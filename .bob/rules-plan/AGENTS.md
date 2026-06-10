# Plan Mode Rules (Non-Obvious Only)

## Architecture Constraints

- **Two-stage workflow**: Processing (`src/algo/`) and visualization (`src/streamlit/`) are separate
- **No direct integration**: Streamlit apps read from timestamped files, not live data
- **Manual path management**: All scripts must manually add `src/` to sys.path (no package structure)

## I/O Architecture

- `utils/io.py` provides centralized I/O but only supports `.txt` format
- `ROOT` path calculation assumes fixed directory depth from `utils/io.py`
- Output files accumulate with timestamps - no cleanup mechanism exists

## Execution Model

- Regular scripts run via `uv run src/<script.py>`
- Streamlit apps require CLI: `uv run streamlit run src/<app.py>` (cannot be executed directly)
- No test runner configured (test_*.py are manual scripts)

## Scalability Considerations

- Adding new file formats requires modifying `utils/io.py` (centralized bottleneck)
- Timestamped outputs will grow indefinitely without manual cleanup
- No dependency injection or configuration system (hardcoded paths)

## Development Constraints

- No linter/formatter configured - code style not enforced
- No CI/CD or automated testing
- Documentation in French may require translation for international teams