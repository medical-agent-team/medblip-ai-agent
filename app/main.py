#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import sys
import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from pathlib import Path
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime
import requests

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))
if project_root not in sys.path:
    sys.path.append(project_root)

from orchestrator.prompts.prompt import ORCHESTRATOR_AGENT_PROMPT
from langchain_aws import ChatBedrock

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
        "QandA": "❓ Q&A",
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
        model_path = os.path.join(os.getcwd(), "app/model")
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        processor = BlipProcessor.from_pretrained(model_path)
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

def generate_medical_report_pdf(collected_info, conversation_history):
    """수집된 의료 정보를 바탕으로 PDF 보고서를 생성합니다."""
    
    # PDF 생성
    pdf = KoreanPDF()
    pdf.add_page()
    
    # 생성 일자
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    pdf.add_section("보고서 생성일", current_time)
    
    # 기본 정보
    if 'basic_response' in collected_info:
        pdf.add_section("1. 기본 정보", collected_info['basic_response'])
    
    # 증상 정보
    if 'symptoms' in collected_info:
        pdf.add_section("2. 주요 증상", collected_info['symptoms'])
    
    # 병력 정보
    if 'medical_history' in collected_info:
        pdf.add_section("3. 과거 병력", collected_info['medical_history'])
    
    # 생활습관
    if 'lifestyle' in collected_info:
        pdf.add_section("4. 생활습관", collected_info['lifestyle'])
    
    # 이미지 분석 결과
    if 'image_analysis' in collected_info:
        pdf.add_section("5. X-ray 이미지 분석 결과", collected_info['image_analysis'])
    
    # 상담 요약
    ai_responses = []
    for msg in conversation_history:
        if msg.startswith("AI: ") and len(msg) > 20:
            ai_responses.append(msg[4:])
    
    if ai_responses:
        summary_text = ""
        for i, response in enumerate(ai_responses[-3:], 1):
            display_text = response[:300] + "..." if len(response) > 300 else response
            summary_text += f"[상담 {i}] {display_text}\n\n"
        
        pdf.add_section("6. 상담 요약", summary_text)
    
    # 주의사항
    disclaimer = """본 보고서는 AI 상담 시스템에 의해 생성된 것으로, 실제 의료진의 진단이나 처방을 대체할 수 없습니다. 정확한 진단과 치료를 위해서는 반드시 의료기관을 방문하여 전문의와 상담하시기 바랍니다."""
    pdf.add_section("7. 주의사항", disclaimer)
    
    # PDF를 바이트로 출력하고 bytes로 변환
    pdf_output = pdf.output()
    
    # bytearray를 bytes로 변환
    if isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    else:
        return pdf_output

# 오케스트레이터 에이전트 클래스
class MedicalOrchestrator:
    def __init__(self):
        self.llm = ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            region_name="us-east-1",  # or your region
            temperature=0.0
            )

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
        elif conversation_stage == "image_upload":
            if "image_analysis" in collected_info:
                # 이미지가 이미 분석된 경우
                return {
                    "decision": "CONTINUE_CONVERSATION",
                    "message": "이미지 분석이 완료되었습니다. 분석 결과에 대해 궁금한 점을 말씀해 주세요.",
                    "next_stage": "analysis",
                    "collected_info": collected_info
                }
            else:
                # 이미지가 아직 업로드되지 않은 경우
                return {
                    "decision": "REQUEST_IMAGE",
                    "message": "X-ray 이미지를 먼저 업로드해 주세요. 이미지 분석 후 상담을 계속 진행하겠습니다.",
                    "next_stage": "image_upload",
                    "collected_info": collected_info
                }
            
        elif conversation_stage == "analysis":
            # 이미지 분석 후 상담 계속
            if "image_analysis" in collected_info:
                if user_input == "끝":
                    return {
                        "decision": "END_CONSULTATION",
                        "message": "상담 종료.",
                        "next_stage": "completed",
                        "collected_info": collected_info
                    }
                # prompt
                patient_analysis_context = f"""
                        다음은 환자의 X-ray 이미지 분석이 완료된 상황입니다.

                        **환자 정보:**
                        - 기본 정보: {collected_info['basic_response']}
                        - 증상: {collected_info['symptoms']}
                        - 병력: {collected_info['medical_history']}
                        - 생활습관: {collected_info['lifestyle']}

                        **X-ray 이미지 분석 결과:**
                        {collected_info["image_analysis"]}

                        이미지 분석을 바탕으로 환자의 질문에 답변해 주세요.
                        """
                prompt_variables = {
                    "user_input": user_input,
                    "conversation_stage": conversation_stage,
                    "collected_info": str(collected_info),
                    "has_image": True,
                    "conversation_history": '\n'.join(conversation_history[-5:]) if conversation_history else '없음',
                    "patient_analysis_context": patient_analysis_context
                }
                
                messages = ORCHESTRATOR_AGENT_PROMPT.format_messages(**prompt_variables)
                response = self.llm.invoke(messages)
                llm_response = response.content
                
                return {
                    "decision": "CONTINUE_CONVERSATION",
                    "message": llm_response,
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
            
        elif conversation_stage == "completed":
            return {
                "decision": "END_CONSULTATION",
                "message": "상담이 완료되었습니다. 궁금한 점이 더 있으시면 언제든 문의해 주세요!",
                "next_stage": "completed",
                "collected_info": collected_info
            }

class KoreanPDF(FPDF):
    """한국어를 지원하는 PDF 클래스"""
    
    def __init__(self):
        super().__init__()
        self.font_downloaded = False
        self.setup_korean_font()
    
    def setup_korean_font(self):
        """한국어 폰트를 설정합니다."""
        try:
            # 폰트 디렉토리 생성
            font_dir = "fonts"
            os.makedirs(font_dir, exist_ok=True)
            
            # NanumGothic 폰트 다운로드
            font_file = os.path.join(font_dir, "NanumGothic.ttf")
            
            if not os.path.exists(font_file):
                print("한국어 폰트 다운로드 중...")
                try:
                    # GitHub에서 NanumGothic 다운로드
                    font_url = "https://github.com/naver/nanumfont/raw/master/fonts/NanumGothic.ttf"
                    response = requests.get(font_url, timeout=30)
                    response.raise_for_status()
                    
                    with open(font_file, 'wb') as f:
                        f.write(response.content)
                    print("폰트 다운로드 완료!")
                except Exception as e:
                    print(f"폰트 다운로드 실패: {e}")
                    return
            
            # 폰트 추가
            self.add_font('NanumGothic', '', font_file, uni=True)
            self.font_downloaded = True
            print("한국어 폰트 설정 완료!")
            
        except Exception as e:
            print(f"폰트 설정 실패: {e}")
            self.font_downloaded = False
    
    def header(self):
        """PDF 헤더"""
        if self.font_downloaded:
            self.set_font('NanumGothic', size=16)
        else:
            self.set_font('Arial', 'B', 16)
        
        self.ln(10)
        self.cell(0, 10, '의료 AI 상담 보고서', 0, 1, 'C')
        self.ln(10)
    
    def add_section(self, title, content):
        """섹션 추가"""
        # 새 페이지가 필요한지 확인
        if self.get_y() > 250:
            self.add_page()
        
        # 제목
        if self.font_downloaded:
            self.set_font('NanumGothic', size=14)
        else:
            self.set_font('Arial', 'B', 12)
        
        self.set_text_color(0, 0, 139)  # 다크블루
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(3)
        
        # 내용
        if self.font_downloaded:
            self.set_font('NanumGothic', size=10)
        else:
            self.set_font('Arial', '', 10)
        
        self.set_text_color(0, 0, 0)  # 검정
        
        # 텍스트 처리
        if content and content.strip():
            try:
                # multi_cell 사용해서 자동 줄바꿈
                self.multi_cell(0, 6, content)
            except Exception as e:
                # multi_cell 실패시 기본 cell 사용
                lines = content.split('\n')
                for line in lines:
                    if len(line) > 80:
                        # 긴 줄은 분할
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word + " ") > 80:
                                if current_line:
                                    self.cell(0, 6, current_line.strip(), 0, 1, 'L')
                                current_line = word + " "
                            else:
                                current_line += word + " "
                        if current_line:
                            self.cell(0, 6, current_line.strip(), 0, 1, 'L')
                    else:
                        self.cell(0, 6, line, 0, 1, 'L')
        
        self.ln(8)
        
        
load_dotenv()

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

# 여기서, st.session_state.conversation_stage == "image_upload"
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
                    st.session_state.conversation_stage = "analysis"        # 다음 세션으로 넘어감
                    
                    # 분석 결과 메시지 추가
                    analysis_message = f"""X-ray 이미지 분석이 완료되었습니다.

**분석 결과:**
{analysis_result}

이 분석 결과를 참고하여 추가적인 상담을 진행하겠습니다. 분석 결과에 대해 궁금한 점이나 추가로 알고 싶은 내용을 말씀해 주세요."""

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": analysis_message
                    })
                    st.rerun()
                else:
                    st.error("모델이 로드되지 않았습니다. 페이지를 새로고침해 주세요.")

if st.session_state.conversation_stage == "completed":
    st.markdown("---")
    st.subheader("📋 상담 보고서")
    st.success("상담이 완료되었습니다!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("상담 내용을 PDF 보고서로 다운로드할 수 있습니다.")
        
        # 수집된 정보 요약 표시
        if st.session_state.collected_info:
            st.write("**수집된 정보:**")
            info_summary = ""
            if 'basic_response' in st.session_state.collected_info:
                info_summary += f"• 기본정보: {st.session_state.collected_info['basic_response'][:50]}...\n"
            if 'symptoms' in st.session_state.collected_info:
                info_summary += f"• 증상: {st.session_state.collected_info['symptoms'][:50]}...\n"
            if 'medical_history' in st.session_state.collected_info:
                info_summary += f"• 병력: {st.session_state.collected_info['medical_history'][:50]}...\n"
            if 'lifestyle' in st.session_state.collected_info:
                info_summary += f"• 생활습관: {st.session_state.collected_info['lifestyle'][:50]}...\n"
            if 'image_analysis' in st.session_state.collected_info:
                info_summary += f"• 이미지 분석: {st.session_state.collected_info['image_analysis'][:50]}...\n"
            
            st.text(info_summary)
    
    with col2:
        if st.button("📄 PDF 보고서 생성", type="primary"):
            try:
                with st.spinner("PDF 보고서를 생성하는 중..."):
                    # PDF 생성
                    pdf_data = generate_medical_report_pdf(
                        st.session_state.collected_info,
                        st.session_state.conversation_history
                    )
                    
                    # 데이터 타입 확인 및 변환
                    if not isinstance(pdf_data, bytes):
                        pdf_data = bytes(pdf_data)
                
                # 파일명 생성
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"의료상담보고서_{current_time}.pdf"
                
                st.download_button(
                    label="📥 PDF 다운로드",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    type="secondary"
                )
                
                st.success("PDF 보고서가 생성되었습니다!")
                
            except Exception as e:
                st.error(f"PDF 생성 중 오류가 발생했습니다: {str(e)}")
                
                # 디버깅 정보 표시
                try:
                    pdf = KoreanPDF()
                    pdf.add_page()
                    test_output = pdf.output()
                    st.info(f"PDF 출력 타입: {type(test_output)}")
                    st.info(f"폰트 다운로드 상태: {pdf.font_downloaded}")
                except Exception as debug_e:
                    st.error(f"디버그 중 오류: {debug_e}")
        
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
