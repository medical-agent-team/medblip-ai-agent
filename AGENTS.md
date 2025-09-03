# Repository Guidelines

## Project Structure & Module Organization
- app/: Streamlit app and agents. Entry UI in `app/main.py` (run via `app/first_service.py` wrapper); orchestrator logic in `app/orchestrator/{agent.py,radiology_agent.py,prompts/}`.
- model/: Local MedBLIP artifacts (e.g., `config.json`, tokenizer files, weights, `sample_image.png`). Do not commit large weights.
- notebooks/: Exploratory notebooks for services and agents.
- docker/: Container configuration (`dockerfile`, `docker-compose.yaml`).
- test_medblip.py: Smoke tests for MedBLIP, OpenAI, and agents.
- Makefile, pyproject.toml: Task runner and dependency metadata (Poetry).

## Build, Test, and Development Commands
- make install [WITH=dev]: Locks and installs via Poetry.
- make add PKG=<name> [WITH=dev]: Adds a dependency to the project.
- make run: Runs the local UI (`streamlit run app/first_service.py`).
- make test: Runs integration smoke tests (model + agents). Skips OpenAI paths without key.
- make docker-dev-build | docker-dev-up | docker-dev-down: Dev container workflow (Streamlit on 8501).
- make docker-prod-up | docker-prod-down: Production profile containers (same app).

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8; 4-space indentation; type hints where practical; short, descriptive docstrings.
- snake_case for files/functions, PascalCase for classes; keep modules cohesive (prompts in `app/orchestrator/prompts/`).
- Avoid side effects at import; prefer pure functions and small units.

## Testing Guidelines
- Place tests as `test_*.py`; keep them self-contained and deterministic.
- Use local model files (`local_files_only=True`); avoid network calls in tests.
- OpenAI optional: tests skip OpenAI calls when `OPENAI_API_KEY` is not set; set `TEST_WITH_NETWORK=true` to actually invoke the API.
- Run `python test_medblip.py` before opening a PR; include output in the PR description if relevant.

## Commit & Pull Request Guidelines
- Commits: imperative and concise. Prefer tags like `feat:`, `fix:`, `chore:`, `docs:`; reference issues (e.g., `(#12)`).
- PRs: include what/why, test steps, and screenshots for UI changes (Streamlit). Link issues, update docs when changing behavior or config.

## Security & Configuration Tips
- Secrets: never commit API keys. Use `.env` with `OPENAI_API_KEY=...` (loaded via `python-dotenv`).
- Models: mount or place artifacts under `model/` locally, or `/app/model` in Docker.
- Docker: use profiles (`dev`/`prod`) via Makefile targets; verify the app with `docker-dev-up` when changing runtime code. The container runs Streamlit on `8501`.
