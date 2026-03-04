# Repository Guidelines

## Project Structure & Module Organization
- Main app code lives in `news-aggregator-2025-main/app.py` (single-file Streamlit application).
- Python dependencies are pinned in `news-aggregator-2025-main/requirements.txt`.
- Local IDE config is under `news-aggregator-2025-main/.idea/` and should not contain app logic.
- Keep new modules in `news-aggregator-2025-main/` and split by concern if `app.py` grows (for example: `services/search.py`, `utils/dates.py`).

## Build, Test, and Development Commands
- Create virtual env (Windows): `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
- Install dependencies: `pip install -r news-aggregator-2025-main/requirements.txt`
- Run locally: `streamlit run news-aggregator-2025-main/app.py`
- Optional dependency update flow: `pip freeze > news-aggregator-2025-main/requirements.txt` after validating changes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `UPPER_CASE` for constants, and clear, domain-focused names (for example, `filter_by_date`).
- Prefer small, pure helper functions for parsing/filtering logic; keep Streamlit UI sections grouped and clearly labeled.
- If you add tooling, use `black` for formatting and `ruff` for linting before opening a PR.

## Testing Guidelines
- There is no formal test suite yet. Add tests with `pytest` under `news-aggregator-2025-main/tests/`.
- Name test files `test_*.py` and test functions `test_<behavior>()`.
- Prioritize coverage for date normalization, filtering logic, and API response parsing.
- Run tests with: `pytest -q`

## Commit & Pull Request Guidelines
- Git history is not available in this workspace snapshot, so use Conventional Commits going forward.
- Example commit messages:
  - `feat: add fallback search limit control`
  - `fix: handle missing published_date in serpapi results`
- PRs should include:
  - Clear summary of user-visible behavior changes
  - Linked issue/ticket (if applicable)
  - Screenshots/GIFs for UI updates in Streamlit
  - Notes on config or dependency updates

## Security & Configuration Tips
- Store secrets in Streamlit secrets (`.streamlit/secrets.toml`), not in source files.
- Required keys include `OPENAI_API_KEY` and `SERPAPI_KEY`.
- Never commit real credentials or exported reports containing sensitive data.
