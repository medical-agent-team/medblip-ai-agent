# vLLM Docker 간단 가이드

목표: 특정 GPU를 선택하고, 로컬 ./model 디렉토리에 미리 다운받은 모델을 vLLM Docker 컨테이너에서 사용해 OpenAI 호환 서버를 띄우는 방법.

---

1) 사전 준비
	•	NVIDIA 드라이버 & Docker 설치
	•	nvidia-container-toolkit 설치 후 Docker가 GPU를 인식하는지 확인

`docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`

 * (선택) Hugging Face 모델을 허브에서 직접 받으려면 HF 토큰 준비

---

2) 로컬 ./model 디렉토리에 모델 미리 다운로드

프로젝트 루트에서:

`mkdir -p ./model`
# 방법 A) huggingface-cli
`huggingface-cli download openai/gpt-oss-20b --local-dir ./model/Qwen3-0.6B`

# (대안) Git LFS로 받는 저장소의 경우
`git lfs install && git clone https://huggingface.co/<org>/<repo> ./model/<name>`

조직/환경상 외부 네트워크 접근이 제한된다면, 오프라인/사내 미러를 통해 ./model에 그대로 배치해도 됩니다.

---

3) GPU 선택 방법 (둘 중 택1)

방법 1: Docker --gpus 옵션으로 특정 디바이스 선택

```
docker run --rm --runtime nvidia --gpus "device=0" \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

 * device=0 → 0번 GPU만 사용. 여러 개라면 device=0,1 처럼 지정 가능.

방법 2: 환경변수로 고정

`export CUDA_VISIBLE_DEVICES=0`

 * 이후 실행되는 컨테이너/프로세스는 0번 GPU만 인식
 * Compose나 시스템 유닛에서 환경변수로 지정해도 동일한 효과.

---

4) vLLM Docker로 로컬 모델 사용해 서버 실행

## gpt-oss 20B + A40 2장(48GB×2) 권장 설정
```
docker run -d --name vllm-20b \
  --runtime=nvidia \
  --gpus all \
  --env NVIDIA_VISIBLE_DEVICES=6,7 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --ipc=host -p 9020:8000 \
  -v $(pwd)/models:/models \
  vllm/vllm-openai:latest \
  --model /models/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

 * `-v $(pwd)/models:/models` : 로컬 ./model → 컨테이너 /models 마운트
 * `--model /models/...` : 로컬에 미리 받은 모델 경로 사용


허브에서 실시간 다운로드를 원한다면 캐시 볼륨과 토큰 지정:
```
docker run -d --name vllm \
  --runtime nvidia --gpus all \
  --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

---

5) Docker Compose 예시 (선호 시)

docker-compose.yml:
```
services:
  vllm-20b:
    image: vllm/vllm-openai:latest
    container_name: vllm-20b
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ["6", "7"]
    ports:
      - "8000:8000"
    ipc: host
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"
    volumes:
      - ./model:/models
    command: [
      "--model", "/models/gpt-oss-20b",
      "--tensor-parallel-size", "2",
      "--gpu-memory-utilization", "0.90",
      "--max-model-len", "8192"
    ]
```

실행:

`docker compose up -d`

---

6) 서버 테스트 (OpenAI 호환 API)

cURL
```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role":"user","content":"안녕!"}],
    "max_tokens": 128
  }'
```

Python (openai 패키지)
```
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role":"user", "content":"안녕!"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

---

7) 자주 쓰는 옵션 요약
	•	--model <path|repo>: 모델 경로 또는 HF repo 이름
	•	--gpu-memory-utilization 0.6~0.95: GPU 메모리 사용 상한 비율
	•	--max-model-len <tokens>: 최대 컨텍스트 길이 (길수록 KV 캐시 메모리 증가)
	•	--tensor-parallel-size <N>: 모델 병렬 분할 수 (GPU 여러 장일 때)
	•	--port 8000: 서버 포트
	•	--download-dir <dir>: 허브 다운로드 캐시 경로
	•	--api-key <key>: OpenAI 호환 서버 접근용 키(미설정 시 인증 없이 접근 가능)

---

8) 트러블슈팅 간단 체크리스트
	•	nvidia-smi가 컨테이너 안에서도 정상 동작하는지 확인
	•	모델 경로 오타/권한 확인 (./model → /models 마운트 여부)
	•	VRAM 부족: --max-model-len 감소, --gpu-memory-utilization 조정, 더 작은/양자화 모델 사용
	•	여러 인스턴스 동시 실행 시: GPU 격리(device=...)와 메모리 비율 분배

---

9) 보안/운영 팁
	•	외부 노출 시 리버스 프록시(HTTPS)와 인증 헤더 적용 (--api-key)
	•	캐시/모델 볼륨의 접근 권한(읽기 전용 마운트 고려: :ro)
	•	로그 민감정보 마스킹 및 요청 로깅 제어 (--disable-log-requests 옵션 등)
    