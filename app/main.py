#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical AI Consultation Service - Minimal Radiological Image Analysis Scenario
"""

import streamlit as st
import sys
import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.orchestrator.agent import OrchestratorAgent
from app.orchestrator.radiology_agent import RadiologyAnalysisAgent


@st.cache_resource
def load_medblip_model():
    """Load finetuned MedBLIP model for medical image analysis"""
    
    # 가능한 모델 경로들 (우선순위 순)
    possible_paths = [
        "/app/model",  # Docker 컨테이너 내부 경로

    ]
    
    for model_path in possible_paths:
        try:
            # 경로 존재 확인
            if os.path.exists(model_path):
                st.info(f"MedBLIP 모델을 로딩 중입니다: {model_path}")
                model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
                processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
                st.success(f"MedBLIP 모델 로딩 완료: {model_path}")
                return model, processor
            else:
                st.info(f"경로에서 모델을 찾을 수 없습니다: {model_path}")
                
        except Exception as e:
            st.warning(f"경로 {model_path}에서 모델 로딩 실패: {str(e)}")
            continue
    
    # 모든 경로에서 실패한 경우
    st.error("""
    MedBLIP 모델을 찾을 수 없습니다. 다음을 확인해주세요:
    
    1. 모델 파일이 다음 경로 중 하나에 있는지 확인:
       - /app/model/
       
    2. 모델 디렉토리에 다음 파일들이 있는지 확인:
       - config.json
       - pytorch_model.bin 또는 model.safetensors
       - tokenizer.json
       - preprocessor_config.json
    """)
    return None, None


@st.cache_resource
def load_agents(orchestrator_prompt: str = "default", radiology_prompt: str = "default"):
    """Load AI agents with specified prompt versions"""
    try:
        orchestrator = OrchestratorAgent(prompt_version=orchestrator_prompt)
        radiology_agent = RadiologyAnalysisAgent(prompt_version=radiology_prompt)
        return orchestrator, radiology_agent
    except Exception as e:
        st.error(f"AI 에이전트 로딩 중 오류가 발생했습니다: {str(e)}")
        return None, None


def analyze_medical_image(image, model, processor):
    """Analyze medical image using finetuned MedBLIP model"""
    try:
        # Convert to PIL Image
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Prepare model input
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values
        
        # Model inference with medical-specific generation
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
    except Exception as e:
        return f"의료 이미지 분석 중 오류가 발생했습니다: {str(e)}"


def initialize_session_state():
    """Initialize session state for minimal scenario"""
    if "conversation_stage" not in st.session_state:
        st.session_state.conversation_stage = "greeting"
    if "collected_info" not in st.session_state:
        st.session_state.collected_info = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False


def render_minimal_sidebar():
    """Render simplified sidebar for minimal scenario"""
    with st.sidebar:
        st.title("🏥 MedBLIP 의료 상담")
        st.markdown("---")
        
        # Prompt experiment controls
        st.markdown("### 🧪 프롬프트 실험")
        
        # Import prompt loader for UI
        try:
            from app.orchestrator.prompts.prompt_loader import list_prompts
            available_prompts = list_prompts()
            
            # Orchestrator prompt selection
            orch_options = ["default"] + available_prompts.get("orchestrator", [])
            orch_options = [opt.replace("orchestrator_", "") if opt.startswith("orchestrator_") else opt for opt in orch_options]
            
            selected_orch = st.selectbox(
                "Orchestrator 프롬프트:",
                options=orch_options,
                key="orchestrator_prompt_version"
            )
            
            # Radiology prompt selection  
            rad_options = ["default"] + available_prompts.get("radiology", [])
            rad_options = [opt.replace("radiology_", "") if opt.startswith("radiology_") else opt for opt in rad_options]
            
            selected_rad = st.selectbox(
                "Radiology 프롬프트:",
                options=rad_options,
                key="radiology_prompt_version"
            )
            
            # Store selected prompts in session state
            st.session_state.selected_orchestrator_prompt = selected_orch
            st.session_state.selected_radiology_prompt = selected_rad
            
        except Exception as e:
            st.warning(f"프롬프트 로딩 오류: {e}")
            st.session_state.selected_orchestrator_prompt = "default"
            st.session_state.selected_radiology_prompt = "default"
        
        st.markdown("---")
        st.markdown("### 진행 단계")

        # Simplified stages for minimal scenario
        stages = {
            "greeting": "🤝 서비스 소개",
            "basic_info": "📝 기본 정보",
            "image_upload": "📷 이미지 업로드",
            "analysis": "🔍 이미지 분석",
            "explanation": "📋 결과 설명"
        }

        for stage, description in stages.items():
            if stage == st.session_state.conversation_stage:
                st.markdown(f"**➤ {description}**")
            else:
                st.markdown(f"   {description}")

        st.markdown("---")
        st.markdown("### 서비스 안내")
        st.info("병원에서 촬영한 방사선 이미지를 업로드하시면, 파인튜닝된 MedBLIP AI가 전문적인 의료 상담을 제공해드립니다.")
        
        if st.button("새로 시작"):
            for key in ["conversation_stage", "collected_info", "messages", "image_uploaded"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def handle_image_upload_and_analysis(orchestrator, radiology_agent, model, processor):
    """Handle image upload and analysis in minimal scenario"""
    st.markdown("---")
    st.subheader("📷 방사선 이미지 업로드")
    st.markdown("병원에서 촬영하신 X-ray, CT, MRI 등의 이미지를 업로드해주세요.")

    uploaded_file = st.file_uploader(
        "방사선 이미지 선택",
        type=['png', 'jpg', 'jpeg', 'dcm'],
        help="PNG, JPG, JPEG, DICOM 형식을 지원합니다."
    )

    if uploaded_file is not None and not st.session_state.image_uploaded:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="업로드된 방사선 이미지", use_column_width=True)

        with col2:
            st.success("이미지가 성공적으로 업로드되었습니다!")
            
            # 모델 상태 표시
            if model is None or processor is None:
                st.error("⚠️ MedBLIP 모델이 로드되지 않았습니다.")
                st.info("모델이 없어도 데모 분석을 진행할 수 있습니다.")
                
            if st.button("이미지 분석 시작", type="primary"):
                if model is not None and processor is not None:
                    # Execute MedBLIP image analysis
                    with st.spinner("MedBLIP AI가 의료 이미지를 분석하는 중입니다..."):
                        medblip_analysis = analyze_medical_image(uploaded_file, model, processor)
                    
                    # Execute medical consultation using AI agent
                    with st.spinner("전문적인 의료 상담을 준비하는 중입니다..."):
                        medical_consultation = radiology_agent.provide_medical_consultation(
                            medblip_analysis, 
                            st.session_state.collected_info
                        )
                    
                    # Update session state
                    st.session_state.collected_info["medblip_analysis"] = medblip_analysis
                    st.session_state.collected_info["medical_consultation"] = medical_consultation
                    st.session_state.conversation_stage = "explanation"
                    st.session_state.image_uploaded = True
                    
                    # Add medical consultation result to messages
                    consultation_message = f"""### 🩺 MedBLIP 기반 의료 상담 결과

**MedBLIP 분석 소견:**
{medblip_analysis}

**전문 상담 내용:**
{medical_consultation}

---
💡 **중요:** 이 상담 내용은 참고용이며, 정확한 진단과 치료를 위해서는 반드시 의료진과 직접 상담하시기 바랍니다."""

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": consultation_message
                    })
                    st.rerun()
                else:
                    # 모델이 없는 경우 데모 분석 제공
                    st.warning("MedBLIP 모델이 로드되지 않았습니다. 데모 분석을 진행합니다.")
                    
                    with st.spinner("데모 분석을 진행하는 중입니다..."):
                        # 데모용 분석 결과
                        demo_medblip_analysis = "chest x-ray demonstrates normal cardiopulmonary findings with clear lung fields and normal heart size"
                        
                        # Execute medical consultation using AI agent  
                        with st.spinner("전문적인 의료 상담을 준비하는 중입니다..."):
                            medical_consultation = radiology_agent.provide_medical_consultation(
                                demo_medblip_analysis, 
                                st.session_state.collected_info
                            )
                        
                        # Update session state
                        st.session_state.collected_info["medblip_analysis"] = demo_medblip_analysis
                        st.session_state.collected_info["medical_consultation"] = medical_consultation
                        st.session_state.conversation_stage = "explanation"
                        st.session_state.image_uploaded = True
                        
                        # Add medical consultation result to messages
                        consultation_message = f"""### 🩺 데모 의료 상담 결과

**데모 분석 소견:**
{demo_medblip_analysis}

**전문 상담 내용:**
{medical_consultation}

---
💡 **참고:** 실제 MedBLIP 모델이 로드되지 않아 데모 분석을 표시하였습니다. 
정확한 분석을 위해서는 MedBLIP 모델 파일이 필요합니다."""

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": consultation_message
                        })
                        st.rerun()


def render_minimal_chat_interface(orchestrator):
    """Render simplified chat interface for minimal scenario"""
    # Initialize with greeting if no messages
    if not st.session_state.messages:
        try:
            # Use orchestrator for initial greeting
            initial_response = orchestrator.process_conversation(
                "", 
                has_image=False
            )
            greeting_message = "안녕하세요! MedBLIP 기반 의료 상담 서비스입니다. 🏥\n\n병원에서 촬영하신 X-ray, CT, MRI 등의 이미지를 업로드해주시면, 파인튜닝된 MedBLIP AI가 분석하여 전문적인 의료 상담을 제공해드립니다.\n\n먼저 간단한 정보를 알려주세요. 나이와 현재 느끼시는 증상이 있다면 말씀해주세요."
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": greeting_message
            })
        except Exception as e:
            # Fallback if orchestrator fails
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "안녕하세요! MedBLIP 기반 의료 상담 서비스입니다. 🏥\n\n병원에서 촬영하신 이미지를 업로드해주시면 전문적인 의료 상담을 제공해드립니다."
            })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input (only before image analysis is complete)
    if st.session_state.conversation_stage != "explanation":
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Process with orchestrator
            with st.chat_message("assistant"):
                with st.spinner("답변을 준비하는 중..."):
                    try:
                        # Update collected info based on user input
                        if "basic_response" not in st.session_state.collected_info:
                            st.session_state.collected_info["basic_response"] = prompt
                        elif "symptoms" not in st.session_state.collected_info:
                            st.session_state.collected_info["symptoms"] = prompt
                        
                        # Generate response
                        if st.session_state.conversation_stage == "greeting":
                            response_message = "감사합니다! 추가로 현재 느끼시는 증상이나 불편한 점이 있으시면 말씀해주세요. 없으시다면 바로 이미지를 업로드해주셔도 됩니다."
                            st.session_state.conversation_stage = "basic_info"
                        else:
                            response_message = "정보를 알려주셔서 감사합니다. 이제 방사선 이미지를 업로드해주세요."
                            st.session_state.conversation_stage = "image_upload"
                        
                        st.markdown(response_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_message
                        })
                        
                    except Exception as e:
                        error_message = "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
                        st.markdown(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_message
                        })


def main():
    """Main application function for minimal radiological image analysis scenario"""
    # Streamlit page configuration
    st.set_page_config(
        page_title="방사선 이미지 AI 분석",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_minimal_sidebar()

    # Load models and agents with selected prompts
    model, processor = load_medblip_model()
    
    # Get selected prompt versions from session state
    orch_prompt = st.session_state.get("selected_orchestrator_prompt", "default")
    rad_prompt = st.session_state.get("selected_radiology_prompt", "default")
    
    # Clear agent cache if prompt versions changed
    if (st.session_state.get("last_orch_prompt") != orch_prompt or 
        st.session_state.get("last_rad_prompt") != rad_prompt):
        load_agents.clear()
        st.session_state.last_orch_prompt = orch_prompt
        st.session_state.last_rad_prompt = rad_prompt
    
    agents = load_agents(orch_prompt, rad_prompt)
    
    if agents is None:
        st.error("AI 서비스를 초기화할 수 없습니다. 환경설정을 확인해주세요.")
        st.stop()
    
    orchestrator, radiology_agent = agents

    # Main interface
    st.title("🏥 MedBLIP 기반 의료 상담 서비스")
    st.markdown("병원에서 촬영한 방사선 이미지를 MedBLIP AI가 분석하여 전문적인 의료 상담을 제공해드립니다.")

    # Render chat interface
    render_minimal_chat_interface(orchestrator)

    # Image upload and analysis section
    if st.session_state.conversation_stage in ["image_upload", "basic_info"]:
        handle_image_upload_and_analysis(orchestrator, radiology_agent, model, processor)

    # Display collected information (for debugging)
    if st.session_state.collected_info and st.checkbox("수집된 정보 확인 (개발용)", value=False):
        with st.expander("수집된 정보"):
            st.json(st.session_state.collected_info)


if __name__ == "__main__":
    main()