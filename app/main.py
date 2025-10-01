#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical AI Consultation Service - Minimal Radiological Image Analysis Scenario
"""

import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Enable LangChain verbose logging
os.environ["LANGCHAIN_VERBOSE"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Set up basic logging for agent outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
agent_logger = logging.getLogger("AGENT_OUTPUT")

from app.agents.admin_agent import AdminAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.doctor_agent import create_doctor_panel


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


@st.cache_resource
def load_supervisor_and_doctors():
    """Load Supervisor Agent and Doctor Panel (cached)."""
    try:
        supervisor = SupervisorAgent()
        doctors = create_doctor_panel()
        st.success("🎯 Supervisor Agent와 Doctor Panel 초기화 완료")
        return supervisor, doctors
    except Exception as e:
        st.error(f"Multi-Agent 초기화 실패: {str(e)}")
        return None, None


def initialize_session_state():
    """Initialize session state for Multi-Agent system"""
    if "admin_agent" not in st.session_state:
        st.session_state.admin_agent = None
    if "supervisor_agent" not in st.session_state:
        st.session_state.supervisor_agent = None
    if "doctor_agents" not in st.session_state:
        st.session_state.doctor_agents = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "greeting"
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "handoff_data" not in st.session_state:
        st.session_state.handoff_data = {}
    if "intake_started" not in st.session_state:
        st.session_state.intake_started = False
    if "deliberation_started" not in st.session_state:
        st.session_state.deliberation_started = False
    if "deliberation_result" not in st.session_state:
        st.session_state.deliberation_result = None
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def render_sidebar():
    """Render sidebar for Admin Agent"""
    with st.sidebar:
        st.title("🏥 Multi-Agent 의료 상담")
        st.markdown("---")
        st.markdown("### 진행 단계")

        # Multi-agent workflow stages
        stages = {
            "greeting": "🤝 서비스 소개",
            "demographics": "📝 인구학적 정보 수집",
            "history": "📋 과거 병력 문진",
            "symptoms": "🩺 현재 증상 문진",
            "medications": "💊 복용 약물 확인",
            "image_request": "📷 이미지 업로드 요청",
            "image_analysis": "🔍 MedBLIP 이미지 분석",
            "deliberation": "🎯 Multi-Agent 심의",
            "completed": "✅ 분석 완료"
        }

        current_stage = st.session_state.current_stage
        for stage, description in stages.items():
            if stage == current_stage:
                st.markdown(f"**➤ {description}**")
            else:
                st.markdown(f"   {description}")

        st.markdown("---")
        st.markdown("### 서비스 안내")
        st.info("LangGraph 기반 Multi-Agent 시스템으로 체계적인 의료 상담을 제공합니다.")

        if st.button("새로 시작"):
            keys_to_clear = [
                "admin_agent", "supervisor_agent", "doctor_agents",
                "messages", "current_stage", "conversation_complete",
                "handoff_data", "intake_started", "deliberation_started",
                "deliberation_result", "session_id"
            ]
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
                st.image(
                    uploaded_file,
                    caption="업로드된 방사선 이미지",
                    use_column_width=True
                )

            with col2:
                st.success("이미지 업로드 완료!")
                if st.button("이미지 분석 시작", type="primary"):
                    # 상태를 이미지 분석으로 먼저 업데이트
                    st.session_state.current_stage = "image_analysis"
                    # Admin Agent에서 이미지 처리 (중복 spinner 제거)
                    with st.spinner("MedBLIP이 이미지 분석중.."):
                        process_with_admin_agent("이미지를 업로드했습니다.", image, show_spinner=False)


def process_with_admin_agent(user_input: str, image=None, show_spinner=True):
    """Process user input with Admin Agent"""
    if st.session_state.admin_agent:
        spinner_text = "Admin Agent가 처리 중입니다..." if show_spinner else None

        if show_spinner:
            with st.spinner(spinner_text):
                _execute_admin_processing(user_input, image)
        else:
            _execute_admin_processing(user_input, image)
    else:
        st.error("Admin Agent가 초기화되지 않았습니다.")


def _execute_admin_processing(user_input: str, image=None):
    """Execute admin agent processing without spinner wrapper"""
    try:
        # Process user input directly (intake should already be started)
        result = st.session_state.admin_agent.process_user_input(
            user_input, image
        )

        if result["success"]:
            # Update session state
            st.session_state.current_stage = result["current_stage"]
            st.session_state.conversation_complete = result[
                "conversation_complete"
            ]

            # Add new messages
            new_messages = result["messages"]
            if new_messages:
                # Only add new messages (avoid duplicates)
                existing_count = len(st.session_state.messages)
                if len(new_messages) > existing_count:
                    for msg in new_messages[existing_count:]:
                        st.session_state.messages.append(msg)

            # Store case context if conversation is complete
            if (result["conversation_complete"] and
                    result.get("case_context")):
                st.session_state.handoff_data = result["case_context"]

            st.rerun()
        else:
            st.error(f"처리 중 오류 발생: {result['error']}")

    except Exception as e:
        st.error(f"Admin Agent 처리 중 예외 발생: {str(e)}")


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
        st.info(
            "✅ 문진이 완료되었습니다. 다음 에이전트에서 상세 분석을 진행합니다."
        )


def start_multi_agent_deliberation():
    """Start multi-agent deliberation with supervisor and doctors"""
    if (st.session_state.conversation_complete and
            st.session_state.handoff_data and
            not st.session_state.deliberation_started):

        st.info("🚀 Multi-Agent 심의를 시작합니다...")

        # Load supervisor and doctors
        if not st.session_state.supervisor_agent or not st.session_state.doctor_agents:
            supervisor, doctors = load_supervisor_and_doctors()
            if supervisor and doctors:
                st.session_state.supervisor_agent = supervisor
                st.session_state.doctor_agents = doctors
            else:
                st.error("Multi-Agent 시스템 로드 실패")
                return

        # Start deliberation
        try:
            # Update progress to deliberation stage
            st.session_state.current_stage = "deliberation"

            with st.spinner("Multi-Agent 심의 진행 중..."):
                result = st.session_state.supervisor_agent.start_deliberation(
                    session_id=st.session_state.session_id,
                    case_context=st.session_state.handoff_data,
                    doctors=st.session_state.doctor_agents
                )

                st.session_state.deliberation_started = True
                st.session_state.deliberation_result = result

                # Update progress to completed when deliberation finishes
                if result.get("success"):
                    st.session_state.current_stage = "completed"

                # Force rerun to update the sidebar status
                st.rerun()

        except Exception as e:
            st.error(f"Multi-Agent 심의 중 오류 발생: {str(e)}")
            st.session_state.deliberation_started = True  # 재시도 방지


def display_deliberation_results(result):
    """Display multi-agent deliberation results"""
    if result.get("success"):
        st.markdown("---")
        st.subheader("🎯 Multi-Agent 심의 결과")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("총 라운드", result.get("total_rounds", 0))
            st.metric("종료 이유", result.get("termination_reason", "unknown"))

        with col2:
            consensus = "✅ 합의 도달" if result.get("consensus_reached") else "⏰ 최대 라운드 도달"
            st.markdown(f"**합의 상태:** {consensus}")

        # Final decision
        final_decision = result.get("final_decision")
        if final_decision:
            st.subheader("📋 최종 의료진 합의")

            st.markdown("**합의된 진단 가설:**")
            hypotheses = final_decision.get("consensus_hypotheses", [])
            for i, hypothesis in enumerate(hypotheses, 1):
                st.markdown(f"{i}. {hypothesis}")

            st.markdown("**권장 진단 검사:**")
            tests = final_decision.get("prioritized_tests", [])
            for i, test in enumerate(tests, 1):
                st.markdown(f"{i}. {test}")

            st.markdown("**의학적 근거:**")
            st.text_area("", final_decision.get("rationale", ""), height=200, disabled=True, key="medical_rationale")

        # Generate patient summary
        if st.button("환자 친화적 요약 생성", type="primary", key="generate_summary_btn"):
            generate_patient_summary(final_decision)

    else:
        st.error(f"Multi-Agent 심의 실패: {result.get('error', 'Unknown error')}")


def generate_patient_summary(supervisor_decision):
    """Generate patient-friendly summary using admin agent"""
    if st.session_state.admin_agent:
        try:
            with st.spinner("환자 친화적 요약 생성 중..."):
                patient_summary = st.session_state.admin_agent.create_patient_summary(
                    supervisor_decision
                )

                st.markdown("---")
                st.subheader("📝 환자용 요약")

                st.markdown("### 상담 결과")
                st.markdown(patient_summary["summary_text"])

                st.markdown("### ⚠️ 중요 안내사항")
                for disclaimer in patient_summary["disclaimers"]:
                    st.warning(disclaimer)

        except Exception as e:
            st.error(f"환자 요약 생성 중 오류 발생: {str(e)}")
    else:
        st.error("Admin Agent가 초기화되지 않았습니다.")


def display_handoff_data():
    """Display handoff data for next agent"""
    if st.session_state.conversation_complete and st.session_state.handoff_data:
        st.markdown("---")
        st.subheader("📊 다음 에이전트로 전달되는 데이터")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**인구학적 정보:**")
            if st.session_state.handoff_data.get("demographics"):
                st.json(st.session_state.handoff_data["demographics"])

            st.markdown("**과거 병력:**")
            if st.session_state.handoff_data.get("history"):
                st.json(st.session_state.handoff_data["history"])

        with col2:
            st.markdown("**현재 증상:**")
            if st.session_state.handoff_data.get("symptoms"):
                st.json(st.session_state.handoff_data["symptoms"])

            st.markdown("**복용 약물:**")
            if st.session_state.handoff_data.get("meds"):
                st.json(st.session_state.handoff_data["meds"])

        st.markdown("**MedBLIP 분석 결과:**")
        if st.session_state.handoff_data.get("medblip_findings"):
            findings = st.session_state.handoff_data["medblip_findings"]
            if isinstance(findings, dict) and findings.get("description"):
                st.text(findings["description"])
            else:
                st.json(findings)

        st.markdown("**자유 텍스트:**")
        if st.session_state.handoff_data.get("free_text"):
            st.text(st.session_state.handoff_data["free_text"])


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

    # ✅ Ensure intake session started before first user input
    if (st.session_state.admin_agent and
            not st.session_state.intake_started):
        intake_result = st.session_state.admin_agent.start_intake()
        if intake_result["success"]:
            st.session_state.intake_started = True
            st.session_state.messages.extend(intake_result["messages"])
            st.session_state.current_stage = intake_result.get(
                "current_stage", "demographics"
            )
        else:
            error_msg = intake_result.get('error', 'Unknown error')
            st.error(f"Intake 시작 실패: {error_msg}")
            return

    # Render sidebar
    render_sidebar()

    # Main interface
    st.title("🤖 Multi-Agent 의료 상담 시스템")
    st.markdown(
        "LangGraph 기반 Admin Agent가 체계적인 건강 문진을 진행하고, "
        "MedBLIP으로 이미지를 분석합니다."
    )

    # Render chat interface
    render_chat_interface()

    # Image upload section
    handle_image_upload()

    # Auto-start multi-agent deliberation when admin conversation completes
    if (st.session_state.conversation_complete and
            st.session_state.handoff_data and
            not st.session_state.deliberation_started):
        start_multi_agent_deliberation()

    # Display deliberation results if available
    if st.session_state.deliberation_result:
        # Ensure status is set to completed when showing results
        if (st.session_state.deliberation_result.get("success") and
            st.session_state.current_stage != "completed"):
            st.session_state.current_stage = "completed"
        display_deliberation_results(st.session_state.deliberation_result)


    # Display handoff data if available (development option)
    if st.checkbox(
        "다음 에이전트 전달 데이터 확인 (개발용)",
        value=False
    ):
        display_handoff_data()


if __name__ == "__main__":
    main()
