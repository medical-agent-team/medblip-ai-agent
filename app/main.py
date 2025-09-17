#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical AI Consultation Service - Minimal Radiological Image Analysis Scenario
"""

import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.agents.agent import OrchestratorAgent
from app.agents.radiology_agent import RadiologyAnalysisAgent
from app.core.model_utils import load_medblip_model as _load_medblip_model


@st.cache_resource
def load_admin_agent():
    """Load Admin Agent (cached)."""
    try:
        agent = AdminAgent()
        st.success("🤖 Admin Agent 초기화 완료")
        return agent
    except Exception as e:
        st.error(f"Admin Agent 초기화 실패: {str(e)}")
        return None






def initialize_session_state():
    """Initialize session state for Admin Agent"""
    if "admin_agent" not in st.session_state:
        st.session_state.admin_agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "greeting"
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "handoff_data" not in st.session_state:
        st.session_state.handoff_data = {}


def render_sidebar():
    """Render sidebar for Admin Agent"""
    with st.sidebar:
        st.title("🏥 Multi-Agent 의료 상담")
        st.markdown("---")
        st.markdown("### 진행 단계")

        # Multi-agent workflow stages
        stages = {
            "greeting": "🤝 서비스 소개",
            "basic_info": "📝 기본 정보 수집",
            "medical_history": "📋 과거 병력 문진",
            "current_symptoms": "🩺 현재 증상 문진",
            "image_request": "📷 이미지 업로드 요청",
            "image_analysis": "🔍 MedBLIP 이미지 분석",
            "data_preparation": "📊 데이터 준비",
            "handoff": "🔄 다음 에이전트로 인계"
        }

        current_stage = st.session_state.current_stage
        for stage, description in stages.items():
            if stage == current_stage:
                st.markdown(f"**➤ {description}**")
            else:
                st.markdown(f"   {description}")

        st.markdown("---")
        st.markdown("### Admin Agent 상태")

        if st.session_state.admin_agent:
            st.success("✅ Admin Agent 활성화")
        else:
            st.warning("⚠️ Admin Agent 비활성화")

        if st.session_state.conversation_complete:
            st.info("✅ 문진 완료 - 다음 에이전트 대기 중")

        st.markdown("---")
        st.markdown("### 서비스 안내")
        st.info("LangGraph 기반 Multi-Agent 시스템으로 체계적인 의료 상담을 제공합니다.")

        if st.button("새로 시작"):
            keys_to_clear = ["admin_agent", "messages", "current_stage", "conversation_complete", "handoff_data"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def handle_image_upload():
    """Handle image upload for Admin Agent"""
    if st.session_state.current_stage in ["image_request", "image_analysis"]:
        st.markdown("---")
        st.subheader("📷 방사선 이미지 업로드")

        uploaded_file = st.file_uploader(
            "방사선 이미지 선택",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            help="PNG, JPG, JPEG, DICOM 형식을 지원합니다.",
            key="image_uploader"
        )

        if uploaded_file is not None:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_file, caption="업로드된 방사선 이미지", use_column_width=True)

            with col2:
                st.success("이미지 업로드 완료!")
                if st.button("이미지 분석 시작", type="primary"):
                    # Admin Agent에서 이미지 처리
                    process_with_admin_agent("이미지를 업로드했습니다.", image)


def process_with_admin_agent(user_input: str, image=None):
    """Process user input with Admin Agent"""
    if st.session_state.admin_agent:
        with st.spinner("Admin Agent가 처리 중입니다..."):
            try:
                result = st.session_state.admin_agent.process_user_input(user_input, image)

                if result["success"]:
                    # Update session state
                    st.session_state.current_stage = result["current_stage"]
                    st.session_state.conversation_complete = result["conversation_complete"]

                    # Add new messages
                    new_messages = result["messages"]
                    if new_messages:
                        latest_message = new_messages[-1]
                        st.session_state.messages.append(latest_message)

                    # Store handoff data if conversation is complete
                    if result["conversation_complete"]:
                        st.session_state.handoff_data = result["collected_data"]

                    st.rerun()
                else:
                    st.error(f"처리 중 오류 발생: {result['error']}")

            except Exception as e:
                st.error(f"Admin Agent 처리 중 예외 발생: {str(e)}")
    else:
        st.error("Admin Agent가 초기화되지 않았습니다.")


def render_chat_interface():
    """Render chat interface for Admin Agent"""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (only if conversation not complete)
    if not st.session_state.conversation_complete:
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # Add user message to display
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Process with Admin Agent
            process_with_admin_agent(prompt)
    else:
        st.info("✅ 문진이 완료되었습니다. 다음 에이전트에서 상세 분석을 진행합니다.")


def display_handoff_data():
    """Display handoff data for next agent"""
    if st.session_state.conversation_complete and st.session_state.handoff_data:
        st.markdown("---")
        st.subheader("📊 다음 에이전트로 전달되는 데이터")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**환자 정보:**")
            if st.session_state.handoff_data.get("patient_info"):
                st.json(st.session_state.handoff_data["patient_info"])

            st.markdown("**과거 병력:**")
            if st.session_state.handoff_data.get("medical_history"):
                st.json(st.session_state.handoff_data["medical_history"])

        with col2:
            st.markdown("**현재 증상:**")
            if st.session_state.handoff_data.get("symptoms"):
                st.json(st.session_state.handoff_data["symptoms"])

            st.markdown("**MedBLIP 분석 결과:**")
            if st.session_state.handoff_data.get("medblip_analysis"):
                st.text(st.session_state.handoff_data["medblip_analysis"])

        # Tasks for next agent
        handoff_data = st.session_state.admin_agent.get_handoff_data()
        if handoff_data.get("tasks_for_next_agent"):
            st.markdown("**다음 에이전트 수행 태스크:**")
            for task in handoff_data["tasks_for_next_agent"]:
                st.markdown(f"- {task}")


def main():
    """Main application function for Multi-Agent medical consultation"""
    # Streamlit page configuration
    st.set_page_config(
        page_title="Multi-Agent 의료 상담",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Load Admin Agent
    if not st.session_state.admin_agent:
        st.session_state.admin_agent = load_admin_agent()

    # Render sidebar
    render_sidebar()

    # Main interface
    st.title("🤖 Multi-Agent 의료 상담 시스템")
    st.markdown("LangGraph 기반 Admin Agent가 체계적인 건강 문진을 진행하고, MedBLIP으로 이미지를 분석합니다.")

    # Initialize conversation with greeting
    if not st.session_state.messages and st.session_state.admin_agent:
        process_with_admin_agent("안녕하세요")

    # Render chat interface
    render_chat_interface()

    # Image upload section
    handle_image_upload()

    # Display handoff data if available
    if st.checkbox("다음 에이전트 전달 데이터 확인 (개발용)", value=False):
        display_handoff_data()


if __name__ == "__main__":
    main()
