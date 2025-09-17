# 리포지터리 가이드라인 (Korean)

## 프로젝트 구조 & 모듈 구성
- `app/`: Streamlit 앱과 에이전트. UI 엔트리 `app/main.py`; 에이전트 로직은 `app/agents/{agent.py,radiology_agent.py,prompts/}`.
- `model/`: 로컬 MedBLIP 아티팩트(`config.json`, 토크나이저 파일, 가중치, `sample_image.png` 등). 대용량 가중치는 커밋하지 않음.
- `docker/`: 컨테이너 설정(`Dockerfile`, `docker-compose.yaml`).
- `Makefile`, `pyproject.toml`: 태스크 러너 및 의존성 메타데이터(Poetry).

## 빌드 및 실행(프로덕션)
- `make install [WITH=dev]`: Poetry로 락 및 설치.
- `make add PKG=<name> [WITH=dev]`: 의존성 추가.
- `make run`: 로컬 UI 실행(`streamlit run app/main.py`).
- `make docker-prod-build | docker-prod-up | docker-prod-down`: 프로덕션 컨테이너 워크플로(Streamlit 8501).

## 코딩 스타일 & 네이밍 컨벤션
- Python 3.11; PEP 8 준수; 4‑스페이스 들여쓰기; 가능하면 타입 힌트; 짧고 명확한 독스트링.
- 파일/함수는 스네이크 케이스, 클래스는 파스칼 케이스; 모듈 응집 유지(프롬프트는 `app/agents/prompts/`).
- 임포트 시 사이드이펙트 지양; 순수 함수와 작은 단위를 선호.

## 테스트 가이드라인
- 테스트는 프로덕션 번들에 포함되지 않음.

## 커밋 & PR 가이드라인
- 커밋: 명령형·간결. `feat:`, `fix:`, `chore:`, `docs:` 등 태그 권장; 이슈 참조 예: `(#12)`.
- PR: 변경 내용/이유, 테스트 방법, UI 변경 시 스크린샷 포함. 동작/설정 변경 시 문서 업데이트.

## 보안 & 설정 팁
- 시크릿: API 키 커밋 금지. `.env`에 `OPENAI_API_KEY=...` 저장하고 `python-dotenv`로 로드.
- 모델: 아티팩트는 로컬 `model/`에 두거나 Docker에서 `/app/model`로 마운트.
- Docker: 프로덕션 프로파일은 8501 포트에서 Streamlit 실행.

## 제품 목표 & 멀티 에이전트 아키텍처
- 목표: 증상·자유 텍스트·영상(방사선) 정보를 결합하고 임상적 추론·합의를 활용하여, 환자에게 이해하기 쉬운 의학 설명을 제공.
- 핵심 결과: 가능한 진단과 권장 진단 검사 요약을 비전문적(일상어) 한국어로 안전하게 제시(면책 포함).

## 에이전트 & 역할
- Admin(관리자) 에이전트(엔트리포인트):
  - 사용자 입력(증상, 병력, 자유 텍스트) 수집, 이미지 업로드 수락, MedBLIP 호출로 영상을 구조화된 텍스트 소견으로 변환.
  - 모든 컨텍스트와 수행 과제를 Supervisor에 전달.
  - 합의 결과를 환자 친화적 한국어로 재작성하여 UI에 반환.
- Supervisor(감독) 에이전트:
  - 정확히 3명의 Doctor(의사) 에이전트를 대상으로 최대 13라운드의 토론을 오케스트레이션(진단·진단 검사에 집중).
  - 제안 평가, 누락 지적, 토론 촉진, 합의 형성; 합의 시 조기 종료.
- Doctor(의사) 에이전트(총 3명, 비전문화):
  - 진단적 추론과 권장 검사를 제시.
  - 동료 의견 평가·비판, 피드백 반영, 산출물 반복 개선.

## 대화 플로(상위 수준)
- 인테이크: Admin이 환자 정보와 MedBLIP 이미지→텍스트 소견을 수집.
- 심의: Supervisor가 3명의 Doctor와 최대 13라운드 진행:
  - 매 라운드: 각 의사가 가설/검사 제시·갱신, 동료 비판(이유 포함).
  - Supervisor가 충돌/리스크를 요약·강조하고 옵션을 압축.
  - 합의 시 또는 13라운드 경과 시 종료.
- 환자 요약: Supervisor 합의 패키지를 Admin이 환자 친화적 한국어로 재작성하여 사용자에게 표시.

## 모듈 맵 & 파일 레이아웃
- `app/agents/`
  - `admin_agent.py`: 인테이크, MedBLIP 호출, 태스크 패키징, 환자 친화적 재작성.
  - `supervisor_agent.py`: 라운드 컨트롤러, 비판 프롬프트, 합의 로직, 종료 조건.
  - `doctor_agent.py`: 추론+비판 루프, 공유 컨텍스트와 동료 코멘트 처리.
  - `conversation_manager.py`: 공유 상태, 메시지 계약, 라운드 기록, 안전 점검; 필요 시 LangGraph StateGraph를 구성해 Admin/Doctor/Supervisor 노드를 오케스트레이션.
  - `prompts/`: 프롬프트 템플릿들
    - `admin_prompts.py`, `supervisor_prompts.py`, `doctor_prompts.py`, `safety_prompts.py`.
- `model/`: MedBLIP 아티팩트(설정/토크나이저/가중치 플레이스홀더). 대용량 가중치 비커밋.

## 데이터 계약(제안)
- `CaseContext`: { demographics, symptoms, history, meds, vitals?, medblip_findings, free_text }.
- `DoctorOpinion`: { hypotheses[], diagnostic_tests[], reasoning, critique_of_peers }.
- `SupervisorDecision`: { consensus_hypotheses[], prioritized_tests[], rationale, termination_reason }.
- `PatientSummary`: `SupervisorDecision`을 안전 프레이밍으로 환자 친화적 한국어로 재작성한 결과.

제안 `medblip_findings` 구조(CUI 코드 지원):
- `description`: MedBLIP이 반환한 자유 텍스트 영상 설명.
- `entities`: { `label`, `cui`(선택), `confidence`(선택, 0–1), `location`(선택) } 리스트.
- `impression`(선택): 모델이 제공할 경우 요약 소견.

## 구현 경로(단계별)
- 단계 0 — 베이스라인 UI(현재):
  - 데모를 위해 단일 에이전트 기반의 간단한 설명 플로 유지.
- 단계 1 — Admin 에이전트 + MedBLIP I/O:
  - `admin_agent.py`로 인테이크·MedBLIP 소견 표준화, `CaseContext` 스키마 정의.
  - 임의 임상 텍스트의 환자 친화적 재작성을 위한 프롬프트 추가.
- 단계 2 — Doctor 에이전트 프로토타입:
  - `doctor_agent.py`(단일 인스턴스)로 `CaseContext` 기반 가설/검사+추론 산출.
  - 자체 비판 루프와 안전 제약(치료 권고 금지, 확정적 표현 금지) 추가.
- 단계 3 — Supervisor + 다수 의사 루프:
  - `supervisor_agent.py`로 정확히 3명의 의사를 최대 13라운드 관리; 네트워크는 LangGraph 노드/엣지로 표현 가능.
  - 비판 교환, 충돌 강조, 합의/종료 기준 추가.
  - `conversation_manager.py`에 라운드별 `DoctorOpinion` 히스토리 보존; `langgraph`가 설치된 경우 해당 모듈의 그래프 빌더로 Admin/Doctor/Supervisor를 연결.
- 단계 4 — 환자 요약 & 핸드오프:
  - `SupervisorDecision`을 Admin 프롬프트로 `PatientSummary`로 변환; Streamlit 렌더링.
  - 면책문, 다음 단계 가이드, 불확실성 프레이밍 포함.
- 단계 5 — 안전 & 가드레일:
  - 콘텐츠 필터, 범위 경계(확정 진단·치료 금지), 리스크 단어 모더레이션.
  - PHI 최소화로 추론 로그 처리; UI에 옵트인 동의문 추가.

## 프롬프트 설계 가이드라인
- 역할·목표를 명시하고, 추론/비판/출력 섹션을 분리.
- 출력은 데이터 계약에 맞춘 키·불릿로 제한하여 파싱 용이성 확보.
- Supervisor: 비판, 불확실성 처리, 종료 조건을 강조.
- Admin: 환자 친화적 재작성, 공감, 안전 면책을 강제.
- 언어: 사용자 노출 출력은 한국어로 응답.

## MedBLIP 통합
- 입력: 영상 → MedBLIP → 텍스트 설명 + 선택적 CUI 코드(UMLS CUI)와 신뢰도.
- Admin은 `CaseContext.medblip_findings`에 `description`과 `entities[]`(가능 시 `cui`) 및 선택적 `impression`으로 표준화.
- Doctor/Supervisor는 추론 시 CUI를 참조할 수 있으나, Admin 최종 요약은 전문 용어를 비전문화.
- 모델 아티팩트는 `model/`에 두고, Docker에서는 `/app/model`로 마운트.

## 안전, 프라이버시, 범위
- 확정적 진단·치료 권고 금지; 교육·트리아지 목적에 한정.
- 항상 전문 의료진 상담을 권고하고, 불확실성과 리스크를 명시.
- 에페메럴 세션만: 세션 동안에만 메모리 내 상태 유지; 저장 없음, 재접속 없음, 서버 측 채팅 히스토리 없음.
- PHI 저장 회피; 채팅 콘텐츠에 대한 지속 로그 비활성화; 텔레메트리가 필요한 경우 식별자 제거 및 메시지 본문 무보관.

## 확정된 결정 사항(Resolved Decisions)
- MedBLIP은 텍스트 설명을 반환하며 진단 보강을 위해 UMLS CUI 코드를 포함할 수 있음.
- 의사 에이전트는 정확히 3명 사용.
- 합의 판단은 다수결이 아닌 Supervisor 휴리스틱 기반.
- 진단 검사의 범위는 광범위(영상, 검사실, 관련 임상 평가 포함).
- 환자 노출 출력 언어는 한국어.
- 채팅 기록은 세션 동안만 존재(저장·재접속 불가).

## 남은 결정 사항(범위 확정용)
- MedBLIP 엔티티: `label`/`cui` 외 확정 필드(신체 부위, 좌우, 중증도 등)와 신뢰도 스케일(0–1 vs 0–100) 정의.
- Supervisor 휴리스틱: 동점 처리, 신뢰도 집계, 라운드 캡 외 종료 기준 세부화.
- 환자 요약 스타일: 한국어 가독 수준, 고정 섹션, 길이 제약.
- 안전 필터: 선호 모더레이션/의료 가드레일 라이브러리.
- 운영 텔레메트리: 에페메럴 전제에서 콘텐츠 비저장 조건 하 익명화 지표(카운트, 지연) 허용 여부.
