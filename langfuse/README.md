Langfuse 셀프 호스팅 세트업

이 디렉터리는 MedBLIP AI Agent 애플리케이션의 옵저버빌리티 트레이스를 로컬에서 수집하기 위한 Langfuse Docker Compose 구성을 담고 있습니다.

개요

Langfuse는 오픈소스 LLM 옵저버빌리티 플랫폼으로, 에이전트에서 발생하는 트레이스, 메트릭, 로그를 수집합니다. 이 구성은 9011~9019 포트 범위만 사용하며, 로컬 MinIO를 통해 S3 없이도 동작합니다.

아키텍처

┌─────────────────────────────────────────────────────────┐
│  MedBLIP AI Agent (port 8501)                           │
│  └─ app/core/observability.py                           │
│     └─ Langfuse Client SDK                              │
│        └─ http://localhost:9019                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Langfuse Stack (this directory)                        │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ PostgreSQL │  │ ClickHouse │  │   Redis    │         │
│  │  (9011)    │  │(9012/9013) │  │  (9014)    │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│         │                │               │               │
│         └────────────────┴───────────────┘               │
│                      │                                   │
│      ┌──────────────────────────┐                        │
│      ▼                          ▼                        │
│  ┌────────────┐           ┌────────────┐                 │
│  │ Langfuse   │           │ Langfuse   │                 │
│  │    Web     │           │   Worker   │                 │
│  │   (9019)   │           │ (internal) │                 │
│  └────────────┘           └────────────┘                 │
│                      │                                   │
│               ┌───────────────┐                          │
│               │    MinIO      │                          │
│               │ (9015 / 9016) │                          │
│               └───────────────┘                          │
└─────────────────────────────────────────────────────────┘

포트 구성

서비스	내부 포트	외부 포트	목적
PostgreSQL	5432	9011	Langfuse 백엔드 데이터베이스
ClickHouse (HTTP)	8123	9012	분석 DB HTTP API
ClickHouse (Native)	9000	9013	분석 DB 네이티브 TCP(마이그레이션)
Redis	6379	9014	캐시/큐
MinIO S3 API	9000	9015	S3 대체 API (이벤트 업로드 대상)
MinIO Console	9001	9016	MinIO 관리 콘솔(UI)
Langfuse Web UI	3000	9019	웹 UI 및 API 엔드포인트
Langfuse Worker	3030	(없음)	백그라운드 잡 처리

예비 포트: 9017, 9018

참고: 본 구성은 S3 없이도 동작하도록 로컬 MinIO를 포함합니다. Langfuse v3는 S3 Event Uploads 버킷이 필수이므로, MinIO가 해당 역할을 수행합니다.

빠른 시작

1) Langfuse 서비스 기동

docker compose up -d

2) 서비스 상태 확인

docker compose ps
docker compose logs -f web

3) 웹 UI 접속

브라우저에서 http://localhost:9019 접속

기본 관리자 계정:
	•	Email: admin@example.com
	•	Password: changeme123!

4) 메인 애플리케이션 설정

루트 .env 업데이트:

LANGFUSE_HOST=http://localhost:9019
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=lf_pk_local_demo_12345
LANGFUSE_SECRET_KEY=lf_sk_local_demo_12345

5) 메인 애플리케이션 실행

cd ..
make run

관리 명령어

docker compose up -d
docker compose down
docker compose down -v
docker compose logs -f [service_name]
docker compose restart web
docker compose ps

데이터 영속성
	•	pgdata — PostgreSQL 데이터
	•	chdata — ClickHouse 데이터
	•	redisdata — Redis 캐시
	•	miniodata — MinIO 객체 스토리지

모두 삭제하고 초기화:

docker compose down -v

보안 및 주의사항
	•	운영 시 DB/Redis/MinIO 포트(9011~9016)는 닫고, 9019만 노출
	•	비밀번호 및 키는 openssl rand 명령어로 강력하게 변경
	•	필요 시 전용 네트워크(langfuse-net) 구성

문제 해결

docker compose logs web
curl http://localhost:9019/api/public/health

업그레이드

docker compose pull
docker compose up -d
docker compose logs -f web

Langfuse는 기동 시 자동으로 DB 마이그레이션을 수행합니다.

참고 링크
	•	Langfuse 공식 문서: https://langfuse.com/docs
	•	셀프 호스팅 가이드: https://langfuse.com/docs/deployment/self-host
	•	API 레퍼런스: https://langfuse.com/docs/api
	•	LangChain 연동: https://langfuse.com/docs/integrations/langchain