# medical-data-agent

Minimal Streamlit app for MedBLIP-based radiology image explanation with optional LLM assistance.

The project is offline-first: it loads a local MedBLIP model from `model/`. If `OPENAI_API_KEY` is provided, agents enhance explanations via OpenAI; without it, the app returns templated offline responses.

## Quickstart

1) Install dependencies (Poetry)
- `make install`

2) Configure environment
- Copy `.env.example` to `.env` and fill in values as needed.
- Optional: set `OPENAI_API_KEY` for enhanced consultation.

3) Place model files
- Put MedBLIP artifacts under `model/` (local) or mount at `/app/model` in Docker.
- Required files typically include: `config.json`, tokenizer files, preprocessor config, and weights (`pytorch_model.bin` or `model.safetensors`).

4) Run the app
- `make run` (equivalent to `streamlit run app/first_service.py`)

5) Run tests
- `make test` (no network calls by default; OpenAI tests are skipped unless you set `TEST_WITH_NETWORK=true` in `.env` and provide `OPENAI_API_KEY`).

## Model Paths
- Local: `./model`
- Docker: `/app/model`

## Docker (optional)
- Build and start (dev profile): `make docker-dev-up`
- Stop: `make docker-dev-down`
- The container runs Streamlit on port `8501` and mounts your repo and `model/` into `/app`.

## Notes
- The app entry is unified in `app/main.py`. `app/first_service.py` is a thin wrapper to preserve existing run commands.
- Agents use offline fallbacks when no `OPENAI_API_KEY` is provided; no network calls are made.
