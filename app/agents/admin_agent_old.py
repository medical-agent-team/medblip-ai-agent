#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admin Agent - Multi-agent system의 관리 에이전트

역할:
- 사용자 입력 수집 (symptoms, history, free-text)
- 이미지 업로드 및 MedBLIP 분석을 통한 구조화된 텍스트 findings 생성
- CaseContext 패키징 및 Supervisor Agent로 작업 전달
- 합의 후 의료 출력을 환자 친화적 한국어로 번역
"""

import os
from typing import Dict, Any, Optional
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.conversation_manager import CaseContext, PatientSummary
from app.agents.admin_workflow import AdminWorkflow, AdminWorkflowState
from app.tools.medblip_tool import MedBLIPTool


# AdminAgentState는 이제 admin_workflow.py의 AdminWorkflowState를 사용


class AdminAgent:
    """
    Multi-agent system의 Admin Agent

    AGENTS.md 명세에 따른 역할:
    - 사용자 입력 수집 (symptoms, history, free-text)
    - 이미지 업로드 및 MedBLIP 분석을 통한 구조화된 텍스트 findings 생성
    - 모든 컨텍스트와 작업 브리프를 Supervisor Agent로 패키징
    - 합의 후 최종 의료 출력을 환자 친화적 한국어로 재작성
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # LLM 초기화 (환자 친화적 재작성용)
        if self.api_key:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.7
            )
        else:
            self.llm = None

        # MedBLIP 도구 초기화
        self.medblip_tool = MedBLIPTool()

        # Admin 워크플로우 초기화
        self.admin_workflow = AdminWorkflow(self.medblip_tool)

        # 현재 워크플로우 상태
        self.current_state: Optional[AdminWorkflowState] = None

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(AdminAgentState)

        # 노드 추가
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("collect_basic_info", self._collect_basic_info_node)
        workflow.add_node("collect_medical_history", self._collect_medical_history_node)
        workflow.add_node("collect_symptoms", self._collect_symptoms_node)
        workflow.add_node("request_image", self._request_image_node)
        workflow.add_node("analyze_image", self._analyze_image_node)
        workflow.add_node("prepare_handoff", self._prepare_handoff_node)

        # 시작점 설정
        workflow.set_entry_point("greeting")

        # 엣지 추가 (조건부 라우팅)
        workflow.add_conditional_edges(
            "greeting",
            self._should_continue_to_basic_info,
            {
                "continue": "collect_basic_info",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "collect_basic_info",
            self._should_continue_to_medical_history,
            {
                "continue": "collect_medical_history",
                "repeat": "collect_basic_info"
            }
        )

        workflow.add_conditional_edges(
            "collect_medical_history",
            self._should_continue_to_symptoms,
            {
                "continue": "collect_symptoms",
                "repeat": "collect_medical_history"
            }
        )

        workflow.add_conditional_edges(
            "collect_symptoms",
            self._should_continue_to_image,
            {
                "continue": "request_image",
                "repeat": "collect_symptoms"
            }
        )

        workflow.add_conditional_edges(
            "request_image",
            self._should_analyze_image,
            {
                "analyze": "analyze_image",
                "wait": "request_image"
            }
        )

        workflow.add_edge("analyze_image", "prepare_handoff")
        workflow.add_edge("prepare_handoff", END)

        return workflow.compile()

    def _greeting_node(self, state: AdminAgentState) -> AdminAgentState:
        """인사 및 서비스 소개 노드"""
        greeting_message = """
        안녕하세요! 🏥 MedBLIP 기반 의료 상담 서비스입니다.

        저는 Admin Agent로, 여러분의 건강 상담을 도와드리겠습니다.

        📋 진행 과정:
        1. 기본 정보 수집 (나이, 성별 등)
        2. 과거 병력 문진
        3. 현재 증상 확인
        4. 방사선 이미지 분석
        5. 전문 의료진 상담으로 연결

        편안하게 말씀해 주시면 됩니다. 시작하겠습니다!
        """

        state["messages"].append({
            "role": "assistant",
            "content": greeting_message,
            "stage": "greeting"
        })
        state["current_stage"] = "basic_info"
        return state

    def _collect_basic_info_node(self, state: AdminAgentState) -> AdminAgentState:
        """기본 정보 수집 노드"""
        if not state["patient_info"]:
            question = """
            기본 정보를 알려주세요:

            1. 나이 (또는 연령대)
            2. 성별
            3. 직업 (선택사항)

            예: "32세 남성, 사무직입니다" 또는 "30대 여성"
            """

            state["messages"].append({
                "role": "assistant",
                "content": question,
                "stage": "basic_info"
            })

        return state

    def _collect_medical_history_node(self, state: AdminAgentState) -> AdminAgentState:
        """과거 병력 수집 노드"""
        if not state["medical_history"]:
            question = """
            과거 병력에 대해 알려주세요:

            1. 기존에 진단받은 질환이 있으신가요?
            2. 현재 복용 중인 약물이 있나요?
            3. 과거 수술 경험이 있으신가요?
            4. 가족력 중 특별한 질환이 있나요?

            없으시면 "없습니다" 라고 말씀해 주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": question,
                "stage": "medical_history"
            })

        return state

    def _collect_symptoms_node(self, state: AdminAgentState) -> AdminAgentState:
        """현재 증상 수집 노드"""
        if not state["symptoms"]:
            question = """
            현재 증상에 대해 자세히 알려주세요:

            1. 어떤 증상이 있으신가요?
            2. 언제부터 시작되었나요?
            3. 증상의 정도는 어떤가요? (1-10점)
            4. 증상이 악화되거나 완화되는 특정 상황이 있나요?

            증상이 없으시면 "검진 목적" 이라고 말씀해 주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": question,
                "stage": "current_symptoms"
            })

        return state

    def _request_image_node(self, state: AdminAgentState) -> AdminAgentState:
        """이미지 업로드 요청 노드"""
        if state["uploaded_image"] is None:
            request_message = """
            📷 방사선 이미지 업로드

            병원에서 촬영하신 다음 이미지를 업로드해 주세요:
            - X-ray (흉부, 복부 등)
            - CT 스캔
            - MRI 이미지
            - 기타 방사선 검사 이미지

            지원 형식: PNG, JPG, JPEG, DICOM

            ⚠️ 개인정보 보호를 위해 환자 정보가 포함된 부분은 가려주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": request_message,
                "stage": "image_request"
            })

        return state

    def _analyze_image_node(self, state: AdminAgentState) -> AdminAgentState:
        """이미지 분석 노드 - MedBLIP 도구 사용"""
        if state["uploaded_image"] and not state["medblip_analysis"]:
            # MedBLIP 도구로 이미지 분석
            try:
                analysis_result = self.medblip_tool.analyze_medical_image(
                    state["uploaded_image"]
                )
                state["medblip_analysis"] = analysis_result

                analysis_message = f"""
                🔍 이미지 분석이 완료되었습니다!

                **MedBLIP 분석 결과:**
                {analysis_result}

                이제 수집된 정보를 종합하여 전문 의료진 상담으로 연결해드리겠습니다.
                """

                state["messages"].append({
                    "role": "assistant",
                    "content": analysis_message,
                    "stage": "image_analysis"
                })

            except Exception as e:
                error_message = f"""
                ⚠️ 이미지 분석 중 오류가 발생했습니다: {str(e)}

                다른 이미지를 시도해주시거나, 데모 모드로 진행하시겠습니까?
                """

                state["messages"].append({
                    "role": "assistant",
                    "content": error_message,
                    "stage": "image_analysis_error"
                })

        return state

    def _prepare_handoff_node(self, state: AdminAgentState) -> AdminAgentState:
        """다음 에이전트로 전달할 데이터 준비"""

        # 다음 에이전트를 위한 태스크 정의
        tasks_for_next_agent = []

        # 환자 정보 기반 태스크
        if state["symptoms"]:
            tasks_for_next_agent.append("증상_분석_및_관련_질환_검토")

        if state["medical_history"]:
            tasks_for_next_agent.append("기존_병력과의_연관성_분석")

        # 이미지 분석 기반 태스크
        if state["medblip_analysis"]:
            tasks_for_next_agent.append("MedBLIP_결과_의학적_해석")
            tasks_for_next_agent.append("환자_맞춤형_설명_생성")
            tasks_for_next_agent.append("추가_검사_필요성_검토")

        tasks_for_next_agent.extend([
            "종합_상담_리포트_작성",
            "환자_교육_자료_제공",
            "후속_조치_권고안_작성"
        ])

        state["tasks_for_next_agent"] = tasks_for_next_agent
        state["conversation_complete"] = True

        handoff_message = """
        📋 정보 수집이 완료되었습니다!

        **수집된 정보:**
        ✅ 기본 정보
        ✅ 과거 병력
        ✅ 현재 증상
        ✅ 방사선 이미지 분석

        이제 전문 의료 상담 에이전트가 종합 분석을 진행하겠습니다.
        잠시만 기다려주세요... 🔄
        """

        state["messages"].append({
            "role": "assistant",
            "content": handoff_message,
            "stage": "handoff"
        })

        return state

    # 조건부 라우팅 함수들
    def _should_continue_to_basic_info(self, state: AdminAgentState) -> str:
        return "continue"

    def _should_continue_to_medical_history(self, state: AdminAgentState) -> str:
        if state["patient_info"]:
            return "continue"
        return "repeat"

    def _should_continue_to_symptoms(self, state: AdminAgentState) -> str:
        if state["medical_history"]:
            return "continue"
        return "repeat"

    def _should_continue_to_image(self, state: AdminAgentState) -> str:
        if state["symptoms"]:
            return "continue"
        return "repeat"

    def _should_analyze_image(self, state: AdminAgentState) -> str:
        if state["uploaded_image"]:
            return "analyze"
        return "wait"

    def process_user_input(self, user_input: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """사용자 입력 처리"""

        # 현재 상태 준비
        if not hasattr(self, 'current_state'):
            self.current_state = AdminAgentState(
                messages=[],
                current_stage="greeting",
                patient_info={},
                medical_history={},
                symptoms={},
                uploaded_image=None,
                medblip_analysis=None,
                tasks_for_next_agent=[],
                conversation_complete=False
            )

        # 사용자 메시지 추가
        self.current_state["messages"].append({
            "role": "user",
            "content": user_input,
            "stage": self.current_state["current_stage"]
        })

        # 이미지가 있으면 상태에 저장
        if image:
            self.current_state["uploaded_image"] = image

        # 사용자 입력을 바탕으로 정보 추출 및 업데이트
        self._extract_and_update_info(user_input)

        # 워크플로우 실행
        try:
            result = self.workflow.invoke(self.current_state)
            return {
                "success": True,
                "messages": result["messages"],
                "current_stage": result["current_stage"],
                "conversation_complete": result["conversation_complete"],
                "tasks_for_next_agent": result.get("tasks_for_next_agent", []),
                "collected_data": {
                    "patient_info": result["patient_info"],
                    "medical_history": result["medical_history"],
                    "symptoms": result["symptoms"],
                    "medblip_analysis": result["medblip_analysis"]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "messages": [{"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {str(e)}"}]
            }

    def _extract_and_update_info(self, user_input: str):
        """사용자 입력에서 의료 정보 추출 및 상태 업데이트"""
        stage = self.current_state["current_stage"]

        if stage == "basic_info":
            # 기본 정보 추출 로직
            info = {}
            if any(keyword in user_input for keyword in ["살", "세", "년생"]):
                info["age_mentioned"] = True
            if any(keyword in user_input for keyword in ["남", "여", "남성", "여성"]):
                info["gender_mentioned"] = True
            if any(keyword in user_input for keyword in ["직", "업무", "일"]):
                info["occupation_mentioned"] = True

            if info:
                self.current_state["patient_info"].update(info)
                self.current_state["patient_info"]["raw_input"] = user_input

        elif stage == "medical_history":
            # 병력 정보 추출
            history = {"raw_input": user_input}
            if "없" in user_input:
                history["has_history"] = False
            else:
                history["has_history"] = True

            self.current_state["medical_history"].update(history)

        elif stage == "current_symptoms":
            # 증상 정보 추출
            symptoms = {"raw_input": user_input}
            if any(keyword in user_input for keyword in ["없", "검진"]):
                symptoms["has_symptoms"] = False
            else:
                symptoms["has_symptoms"] = True

            self.current_state["symptoms"].update(symptoms)

    def get_handoff_data(self) -> Dict[str, Any]:
        """다음 에이전트로 전달할 데이터 반환"""
        if hasattr(self, 'current_state') and self.current_state["conversation_complete"]:
            return {
                "patient_info": self.current_state["patient_info"],
                "medical_history": self.current_state["medical_history"],
                "symptoms": self.current_state["symptoms"],
                "medblip_analysis": self.current_state["medblip_analysis"],
                "tasks_for_next_agent": self.current_state["tasks_for_next_agent"],
                "conversation_history": self.current_state["messages"]
            }
        return {}

    def reset(self):
        """에이전트 상태 초기화"""
        if hasattr(self, 'current_state'):
            delattr(self, 'current_state')