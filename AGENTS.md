# Repository Guidelines

## Project Structure & Module Organization
- app/: Streamlit app and agents. Entry UI in `app/main.py`; orchestrator logic in `app/orchestrator/{agent.py,radiology_agent.py,prompts/}`.
- model/: Local MedBLIP artifacts (e.g., `config.json`, tokenizer files, weights, `sample_image.png`). Do not commit large weights.
- docker/: Container configuration (`Dockerfile`, `docker-compose.yaml`).
- Makefile, pyproject.toml: Task runner and dependency metadata (Poetry).

## Build and Run (Production)
- make install [WITH=dev]: Locks and installs via Poetry.
- make add PKG=<name> [WITH=dev]: Adds a dependency to the project.
- make run: Runs the local UI (`streamlit run app/main.py`).
- make docker-prod-build | docker-prod-up | docker-prod-down: Production container workflow (Streamlit on 8501).

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8; 4-space indentation; type hints where practical; short, descriptive docstrings.
- snake_case for files/functions, PascalCase for classes; keep modules cohesive (prompts in `app/orchestrator/prompts/`).
- Avoid side effects at import; prefer pure functions and small units.

## Testing Guidelines
- Tests are not included in the production bundle.

## Commit & Pull Request Guidelines
- Commits: imperative and concise. Prefer tags like `feat:`, `fix:`, `chore:`, `docs:`; reference issues (e.g., `(#12)`).
- PRs: include what/why, test steps, and screenshots for UI changes (Streamlit). Link issues, update docs when changing behavior or config.

## Security & Configuration Tips
- Secrets: never commit API keys. Use `.env` with `OPENAI_API_KEY=...` (loaded via `python-dotenv`).
- Models: place artifacts under `model/` locally, or mount to `/app/model` in Docker.
- Docker: production profile runs Streamlit on `8501`.
