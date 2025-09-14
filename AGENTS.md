# Repository Guidelines

## Project Structure & Module Organization
- app/: Streamlit app and agents. Entry UI in `app/main.py`; agent logic in `app/agents/{agent.py,radiology_agent.py,prompts/}`.
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
- snake_case for files/functions, PascalCase for classes; keep modules cohesive (prompts in `app/agents/prompts/`).
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

## Product Objective & Multi‑Agent Architecture
- Objective: Provide patients with clear, patient‑friendly medical explanations by combining symptoms, free‑text descriptions, and radiological imaging, using a multi‑agent workflow that reasons clinically and reaches consensus.
- Core outcome: A final answer summarizing likely diagnoses and recommended diagnostic tests, translated to non‑technical language and safely framed with disclaimers.

## Agents & Roles
- Admin Agent (entrypoint):
  - Gathers user inputs (symptoms, history, free‑text), accepts image uploads, and invokes MedBLIP to convert images into structured text findings.
  - Packages all context and a task brief for the Supervisor Agent.
  - After consensus, rewrites the final medical output into patient‑friendly language and returns it to the UI.
- Supervisor Agent:
  - Orchestrates a panel of exactly 3 Doctor Agents through iterative rounds (max 13) focused on diagnosis and diagnostic tests.
  - Evaluates suggestions, challenges gaps, facilitates discussion, and drives consensus; terminates early once consensus is reached.
- Doctor Agents (3 total, non‑specialized):
  - Provide diagnostic reasoning and recommendations.
  - Evaluate and comment on peers’ opinions, address feedback, and iteratively improve their output.

## Conversation Flow (High Level)
- Intake: Admin collects patient info and MedBLIP image‑to‑text findings.
- Deliberation: Supervisor runs up to 13 rounds with the 3 Doctor Agents:
  - Round k: each doctor proposes/updates diagnosis hypotheses and diagnostic tests; critiques peers with reasons.
  - Supervisor synthesizes, highlights conflicts/risks, and narrows options.
  - Termination if consensus achieved or after round 13.
- Patient Summary: Supervisor’s consensus package → Admin, which rewrites to plain language and displays to user.

## Module Map & File Layout
- `app/agents/`
  - `admin_agent.py`: intake, MedBLIP invocation, task packaging, patient‑friendly rewrite.
  - `supervisor_agent.py`: round controller, critique prompts, consensus logic, termination conditions.
  - `doctor_agent.py`: reasoning + critique loop, accepts shared context and peer comments.
  - `conversation_manager.py`: shared state, message contracts, round bookkeeping, safety checks; optionally builds a LangGraph StateGraph to orchestrate Admin/Supervisor/Doctors.
  - `prompts/`: prompt templates
    - `admin_prompts.py`, `supervisor_prompts.py`, `doctor_prompts.py`, `safety_prompts.py`.
- `model/`: MedBLIP artifacts (config/tokenizer/weights placeholder). Large weights not committed.

## Data Contracts (suggested)
- `CaseContext`: { demographics, symptoms, history, meds, vitals?, medblip_findings, free_text }.
- `DoctorOpinion`: { hypotheses[], diagnostic_tests[], reasoning, critique_of_peers }.
- `SupervisorDecision`: { consensus_hypotheses[], prioritized_tests[], rationale, termination_reason }.
- `PatientSummary`: admin‑level plain‑language rewrite of `SupervisorDecision` with safety framing.

Suggested `medblip_findings` structure (supports CUI codes):
- `description`: free‑text image description returned by MedBLIP.
- `entities`: list of { `label`, `cui` (optional), `confidence` (optional 0–1), `location` (optional) }.
- `impression` (optional): short summary if provided by model.

## Implementation Path (Phased)
- Phase 0 — Baseline UI (current):
  - Maintain simple intake and single‑agent explanation flow for demos.
- Phase 1 — Admin Agent + MedBLIP I/O:
  - Build `admin_agent.py` to standardize intake and MedBLIP findings; define `CaseContext` schema.
  - Add prompts for patient‑friendly rewrite of arbitrary clinical text.
- Phase 2 — Doctor Agent Prototype:
  - Implement `doctor_agent.py` (single instance) to produce hypotheses/tests + reasoning using `CaseContext`.
  - Add self‑critique loop and safety constraints (no treatment advice, no certainty).
- Phase 3 — Supervisor + Multi‑Doctor Loop:
  - Implement `supervisor_agent.py` to manage exactly 3 doctor instances across rounds with a hard cap of 13; network can be expressed using LangGraph nodes/edges.
  - Add critique exchange, conflict highlighting, and consensus/termination criteria.
  - Persist `DoctorOpinion` history per round in `conversation_manager.py`; use its LangGraph graph builder to wire Admin/Doctors/Supervisor when `langgraph` is installed.
- Phase 4 — Patient Summary & Hand‑off:
  - Convert `SupervisorDecision` to `PatientSummary` via Admin prompts; render in Streamlit.
  - Include disclaimers, next‑step guidance, and uncertainty framing.
- Phase 5 — Safety & Guardrails:
  - Add content filters, scope boundaries (no definitive diagnosis/treatment), and risk‑word moderation.
  - Log reasoning traces with PHI minimization; add opt‑in consent text in UI.

## Prompt Design Guidelines
- Keep roles and goals explicit; separate reasoning, critique, and output sections.
- Constrain outputs to the data contracts; require bullets/keys to simplify parsing.
- For Supervisor: emphasize critique, uncertainty handling, and termination conditions.
- For Admin: enforce plain‑language rewrite, empathy, and safety disclaimers.
- Language: respond in Korean for all user‑visible outputs.

## MedBLIP Integration
- Input: radiology image → MedBLIP → text description plus optional CUI codes (UMLS CUIs) and confidences.
- Admin normalizes findings into `CaseContext.medblip_findings` with `description`, `entities[]` including `cui` when present, and optional `impression`.
- Doctor/Supervisor agents may reference CUIs when reasoning; Admin strips jargon in the final patient summary.
- Keep model artifacts in `model/`; mount to `/app/model` in Docker.

## Safety, Privacy, and Scope
- No definitive diagnoses or treatment recommendations; educational and triage‑oriented only.
- Always advise consulting licensed professionals; include uncertainty and risk framing.
- Ephemeral sessions only: persist chat state in memory per session; no saving, no reconnect, no server‑side chat history.
- Avoid storing PHI; disable persistent logs for chat content; if any telemetry is enabled, ensure PHI redaction and zero retention of message bodies.

## Resolved Decisions
- MedBLIP returns text description and may include UMLS CUI codes to elaborate diagnoses.
- Use exactly 3 Doctor Agents in deliberation.
- Consensus is determined via Supervisor heuristics, not majority vote.
- Diagnostic test scope is broad (imaging, labs, and relevant clinical assessments).
- Output language is Korean for all patient‑facing content.
- Chat records exist only during the session; no saving and no reconnect.

## Open Questions (to finalize scope)
- MedBLIP entities: confirm exact fields beyond `label`/`cui` (e.g., body region, laterality, severity) and confidence scale (0–1 or 0–100?).
- Supervisor heuristic: specify tie‑breakers, confidence aggregation, and termination criteria details beyond round cap.
- Patient summary style: target reading level, fixed sections, and length constraints in Korean.
- Safety filters: preferred library/policy for content moderation and medical guardrails.
- Operational telemetry: with ephemeral chats, is any anonymized metrics collection allowed (counts, model latency) without content storage?
