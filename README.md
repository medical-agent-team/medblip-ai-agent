# medical-data-agent (Production)

MedBLIP 기반 방사선 이미지 설명용 Streamlit 앱의 프로덕션 구성입니다. 
로컬 `model/` 디렉토리의 MedBLIP 아티팩트를 사용하며, `OPENAI_API_KEY`가 있으면 보강 답변을 제공합니다. 키가 없어도 오프라인으로 동작합니다.

## 빠른 시작 (Production)

1) 의존성 설치 (Poetry)
- `make install`

2) 환경 변수 설정
- `.env.example`를 `.env`로 복사 후 필요한 값을 채웁니다.

3) 모델 파일 배치
- MedBLIP 아티팩트를 로컬 `model/` 아래에 둡니다. (Docker 사용 시 `/app/model`로 마운트됨)
- 필요 파일: `config.json`, 토크나이저 파일들, `preprocessor_config.json`, 가중치(`pytorch_model.bin` 또는 `model.safetensors`)

4) 앱 실행
- 로컬: `make run` (내부적으로 `streamlit run app/main.py`)
- Docker (권장):
  - 빌드: `make docker-prod-build`
  - 실행: `make docker-prod-up`
  - 중지: `make docker-prod-down`

## 모델 경로
- 로컬: `./model`
- Docker: `/app/model` (Compose에서 로컬 `model/`만 마운트)

## 참고
- 테스트 및 노트북 자산은 프로덕션 구성에서 제거되었습니다.
- 앱 엔트리포인트는 `app/main.py`입니다.
