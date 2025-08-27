#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import sys
import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))
if project_root not in sys.path:
    sys.path.append(project_root)

# from app.orchestrator.prompts.prompt import ORCHESTRATOR_AGENT_PROMPT
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# In[ ]:


# Streamlit 페이지 설정
st.set_page_config(
    page_title="의료 AI 상담 서비스",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 설정
with st.sidebar:
    st.title("🏥 의료 AI 상담")
    st.markdown("---")
    st.markdown("### 상담 진행 단계")

    # 세션 상태 초기화
    if "conversation_stage" not in st.session_state:
        st.session_state.conversation_stage = "greeting"
    if "collected_info" not in st.session_state:
        st.session_state.collected_info = {}
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 진행 상태 표시
    stages = {
        "greeting": "🤝 인사 및 안내",
        "basic_info": "📝 기본 정보 수집",
        # "symptoms": "🩺 증상 확인",
        # "history": "📋 병력 조사",
        # "lifestyle": "🏃‍♂️ 생활습관 확인",
        "image_upload": "📷 이미지 업로드",
        "analysis": "🔍 분석 중",
        "completed": "✅ 상담 완료"
    }

    for stage, description in stages.items():
        if stage == st.session_state.conversation_stage:
            st.markdown(f"**➤ {description}**")
        else:
            st.markdown(f"   {description}")

    st.markdown("---")
    if st.button("대화 초기화"):
        for key in ["conversation_stage", "collected_info", "conversation_history", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# In[3]:


# BLIP 모델 초기화
@st.cache_resource
def load_blip_model():
    """BLIP 모델을 로드합니다."""
    try:
        model = BlipForConditionalGeneration.from_pretrained("./model")
        processor = BlipProcessor.from_pretrained("./model")
        return model, processor
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {str(e)}")
        return None, None

def analyze_xray_image(image, model, processor):
    """X-ray 이미지를 BLIP 모델로 분석합니다."""
    try:
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # 모델 입력 준비
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values
        
        # 모델 추론
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
    except Exception as e:
        return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

# 오케스트레이터 에이전트 클래스
class MedicalOrchestrator:
    def __init__(self):
        # 실제 환경에서는 환경변수에서 API 키를 가져오세요
        # self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        pass

    def process_conversation(self, user_input, conversation_stage, collected_info, conversation_history, has_image=False):
        """
        사용자 입력을 처리하고 응답을 생성합니다.
        실제 환경에서는 여기서 LangChain을 사용하여 LLM을 호출합니다.
        """
        # 데모용 응답 로직 (실제로는 ORCHESTRATOR_AGENT_PROMPT를 사용)
        if conversation_stage == "greeting":
            return {
                "decision": "CONTINUE_CONVERSATION",
                "message": "안녕하세요! 건강검진 상담을 도와드리는 의료 AI입니다. 😊\\n\\n먼저 기본적인 정보를 여쭤보겠습니다. 성함과 나이를 알려주시겠어요?",
                "next_stage": "basic_info",
                "collected_info": collected_info
            }
        elif conversation_stage == "basic_info":
            # 기본 정보 수집
            collected_info["basic_response"] = user_input
            return {
                "decision": "CONTINUE_CONVERSATION", 
                "message": f"감사합니다! 현재 어떤 증상이나 불편한 점이 있으신지 자세히 말씀해 주세요.",
                "next_stage": "symptoms",
                "collected_info": collected_info
            }
        elif conversation_stage == "symptoms":
            collected_info["symptoms"] = user_input
            return {
                "decision": "CONTINUE_CONVERSATION",
                "message": "증상에 대해 말씀해 주셔서 감사합니다. 과거에 큰 병을 앓으신 적이 있거나 수술을 받으신 경험이 있나요?",
                "next_stage": "history", 
                "collected_info": collected_info
            }
        elif conversation_stage == "history":
            collected_info["medical_history"] = user_input
            return {
                "decision": "CONTINUE_CONVERSATION",
                "message": "네, 알겠습니다. 평소 흡연이나 음주는 하시는지, 그리고 운동은 얼마나 자주 하시는지 알려주세요.",
                "next_stage": "lifestyle",
                "collected_info": collected_info
            }
        elif conversation_stage == "lifestyle":
            collected_info["lifestyle"] = user_input
            return {
                "decision": "REQUEST_IMAGE",
                "message": "생활습관에 대해 잘 알겠습니다. 이제 X-ray 이미지가 있으시면 업로드해 주세요. 더 정확한 분석을 도와드릴 수 있습니다.",
                "next_stage": "image_upload",
                "collected_info": collected_info
            }
        elif conversation_stage == "analysis":
            # 이미지 분석 후 상담 계속
            if "image_analysis" in collected_info:
                return {
                    "decision": "CONTINUE_CONVERSATION",
                    "message": f"분석 결과를 바탕으로 답변드리겠습니다.\n\n{user_input}에 대한 추가 설명을 드리면, 업로드해주신 X-ray 이미지에서 '{collected_info['image_analysis']}'가 관찰됩니다.\n\n다른 궁금한 점이 있으시면 언제든 말씀해 주세요.",
                    "next_stage": "analysis",
                    "collected_info": collected_info
                }
            else:
                return {
                    "decision": "CONTINUE_CONVERSATION",
                    "message": "죄송합니다. 이미지 분석 결과를 찾을 수 없습니다. 이미지를 다시 업로드해 주시겠어요?",
                    "next_stage": "image_upload",
                    "collected_info": collected_info
                }
        else:
            return {
                "decision": "END_CONSULTATION",
                "message": "상담이 완료되었습니다. 궁금한 점이 더 있으시면 언제든 문의해 주세요!",
                "next_stage": "completed",
                "collected_info": collected_info
            }

# 오케스트레이터 인스턴스 생성
orchestrator = MedicalOrchestrator()

# BLIP 모델 로드
model, processor = load_blip_model()


# In[4]:


# 메인 인터페이스
st.title("🏥 의료 AI 상담 서비스")
st.markdown("건강검진을 위한 AI 상담을 시작합니다. 편안하게 대화해 주세요.")

# 채팅 히스토리 표시
chat_container = st.container()

with chat_container:
    # 초기 메시지 표시
    if not st.session_state.messages:
        initial_response = orchestrator.process_conversation(
            "", "greeting", st.session_state.collected_info, 
            st.session_state.conversation_history
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_response["message"]})
        st.session_state.conversation_stage = initial_response["next_stage"]

    # 메시지 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(f"사용자: {prompt}")

    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            response = orchestrator.process_conversation(
                prompt,
                st.session_state.conversation_stage,
                st.session_state.collected_info,
                st.session_state.conversation_history
            )

            # 응답 표시
            st.markdown(response["message"])

            # 세션 상태 업데이트
            st.session_state.messages.append({"role": "assistant", "content": response["message"]})
            st.session_state.conversation_stage = response["next_stage"]
            st.session_state.collected_info = response["collected_info"]
            st.session_state.conversation_history.append(f"AI: {response['message']}")

# 이미지 업로드 섹션 (image_upload 단계에서만 표시)
if st.session_state.conversation_stage == "image_upload":
    st.markdown("---")
    st.subheader("📷 X-ray 이미지 업로드")

    uploaded_file = st.file_uploader(
        "X-ray 이미지를 업로드하세요",
        type=['png', 'jpg', 'jpeg', 'dcm'],
        help="PNG, JPG, JPEG, DICOM 형식을 지원합니다."
    )

    if uploaded_file is not None:
        # 이미지 표시
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="업로드된 X-ray 이미지", use_column_width=True)

        with col2:
            st.info("이미지가 업로드되었습니다!")
            if st.button("이미지 분석 시작", type="primary"):
                if model is not None and processor is not None:
                    # 이미지 분석 실행
                    with st.spinner("AI가 X-ray 이미지를 분석하는 중입니다..."):
                        analysis_result = analyze_xray_image(uploaded_file, model, processor)
                    
                    # 분석 결과를 세션에 저장
                    st.session_state.collected_info["image_analysis"] = analysis_result
                    st.session_state.conversation_stage = "analysis"
                    
                    # 분석 결과 메시지 추가
                    analysis_message = f"""X-ray 이미지 분석이 완료되었습니다.

**분석 결과:**
{analysis_result}

이 분석 결과를 참고하여 추가적인 상담을 진행하겠습니다. 분석 결과에 대해 궁금한 점이나 추가로 알고 싶은 내용이 있으시면 말씀해 주세요."""

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": analysis_message
                    })
                    st.rerun()
                else:
                    st.error("모델이 로드되지 않았습니다. 페이지를 새로고침해 주세요.")

# 수집된 정보 표시 (디버깅용)
if st.session_state.collected_info:
    with st.expander("수집된 정보 확인"):
        st.json(st.session_state.collected_info)


# ## Streamlit 실행 방법
# 
# 위의 코드를 실행하려면 다음 명령어를 사용하세요:
# 
# ```bash
# streamlit run notebooks/first_service.ipynb
# ```
# 
# 또는 Python 스크립트로 변환 후 실행:
# 
# ```bash
# jupyter nbconvert --to script notebooks/first_service.ipynb
# streamlit run notebooks/first_service.py
# ```
# 
# ## 주요 기능
# 
# 1. **단계별 상담 진행**: 사이드바에서 현재 진행 단계를 확인할 수 있습니다
# 2. **채팅 인터페이스**: 자연스러운 대화형 상담
# 3. **이미지 업로드**: X-ray 이미지 업로드 및 분석 요청
# 4. **세션 관리**: 대화 히스토리와 수집된 정보를 세션에 저장
# 5. **정보 수집**: 기본 정보, 증상, 병력, 생활습관 등을 체계적으로 수집
# 
# ## 실제 배포 시 추가할 사항
# 
# - OpenAI API 키 설정 및 LangChain 연동
# - 실제 의료 이미지 분석 모델 연동 (medblip 에이전트)
# - 데이터베이스 연결 및 상담 기록 저장
# - 보안 및 개인정보 보호 강화
