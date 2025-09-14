# feat: 슬림화된 앱 리팩터 (오프라인 우선 에이전트, 모델 경로 통합, 문서/도커 정리)

## 요약
프로젝트를 단일 Streamlit 플로우로 슬림화하고, 에이전트를 오프라인 우선(네트워크 없이 동작)으로 전환했습니다. 모델 경로를 통합하고 Docker 구성을 단순화했으며, 테스트와 `.env.example`을 정비했습니다. `model/` 디렉토리는 변경하지 않았습니다.

## 배경/목표
- 중복/레거시 코드 경로를 제거하여 의존성과 러닝타임 복잡도를 낮춤
- API 키 없이도 재현 가능한 오프라인 로컬 실행 보장
- 컨테이너와 의존성 발자국 축소

## 주요 변경 사항
- 앱 진입점
  - `app/first_service.py`는 이제 `app/main.py`를 호출하는 얇은 래퍼입니다(기존 실행 명령 유지).
  - `app/main.py`는 `./model` 또는 `/app/model`에서 모델을 로드하며, OpenAI 키가 없어도 하드 실패 없이 동작합니다.
- 에이전트(오프라인 우선)
  - `app/orchestrator/agent.py`: OpenAI 선택적. 키가 없으면 최소 오프라인 플로우로 응답. 실험적 LangGraph 워크플로우와 관련 타입 제거.
  - `app/orchestrator/radiology_agent.py`: OpenAI 선택적. 키가 없으면 템플릿 기반 오프라인 상담 제공.
- 테스트(기본 무네트워크)
  - `test_medblip.py`: 모델 경로 통일, `sample_image.png` 사용, 로컬 가중치가 없으면 모델 테스트를 SKIP, OpenAI는 `TEST_WITH_NETWORK=true`일 때만 실제 호출.
- 도구/문서
  - `.env.example` 추가, Makefile에 `run`/`test` 타겟 추가.
  - `README.md`, `AGENTS.md`를 빠른 시작/오프라인 동작/Docker 사용법으로 업데이트.
- 의존성
  - `pyproject.toml` 슬림화(제거: opencv-python, scikit-learn, matplotlib, seaborn, langgraph, langchain-community). `poetry.lock` 재생성.
- Docker
  - Streamlit(8501)을 실행하는 단일 스테이지 Dockerfile. 저장소와 `model/`을 마운트하는 단일 `app` 서비스로 compose 간소화.

## 동작 변화
- 레거시 `first_service.py`(PDF/폰트/Bedrock)는 제거되었고, 해당 파일은 이제 `app/main.py`로 위임합니다.
- `OPENAI_API_KEY`가 없을 경우, 에이전트는 템플릿 기반 오프라인 응답(네트워크 미사용)을 반환합니다.

## 실행 방법
- 설치: `make install`
- UI 실행: `make run` → http://localhost:8501
- 테스트: `make test` (기본은 OpenAI 호출 SKIP, 필요 시 `.env`에 `TEST_WITH_NETWORK=true` 설정)
- Docker(dev): `make docker-dev-up` → http://localhost:8501 (로컬 `model/`을 `/app/model`로 마운트)

## 모델 아티팩트
- 로컬에서는 `model/` 아래에, Docker에서는 `/app/model`로 마운트하세요. 저장소 내 `model/`은 최소 형태로 유지됩니다.

## 롤백 계획
- 이 PR을 되돌리거나 이전 브랜치로 전환합니다.
