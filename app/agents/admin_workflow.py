#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admin Agent LangGraph Workflow
Admin Agent의 intake 워크플로우를 관리하는 별도 모듈
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from PIL import Image
from langgraph.graph import StateGraph, START, END

from app.agents.conversation_manager import CaseContext
from app.tools.medblip_tool import MedBLIPTool


class AdminWorkflowState(CaseContext, total=False):
    """Admin Agent 워크플로우 상태 - CaseContext 확장"""
    messages: list[Dict[str, Any]]
    current_stage: str
    uploaded_image: Optional[Image.Image]
    conversation_complete: bool
    error_message: Optional[str]


class AdminWorkflow:
    """Admin Agent의 intake 워크플로우를 관리하는 클래스"""

    def __init__(self, medblip_tool: Optional[MedBLIPTool] = None):
        self.medblip_tool = medblip_tool or MedBLIPTool()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(AdminWorkflowState)

        # 노드 추가
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("collect_demographics",
                          self._collect_demographics_node)
        workflow.add_node("collect_history", self._collect_history_node)
        workflow.add_node("collect_symptoms", self._collect_symptoms_node)
        workflow.add_node("collect_medications",
                          self._collect_medications_node)
        workflow.add_node("request_image", self._request_image_node)
        workflow.add_node("analyze_image", self._analyze_image_node)
        workflow.add_node("prepare_case_context",
                          self._prepare_case_context_node)

        # 시작점 설정
        workflow.add_edge(START, "greeting")

        # 조건부 흐름 - 각 단계에서 사용자 입력 확인
        workflow.add_conditional_edges(
            "greeting",
            self._should_continue_to_demographics,
            {
                "continue": "collect_demographics",
                "wait": END
            }
        )

        workflow.add_conditional_edges(
            "collect_demographics",
            self._should_continue_to_history,
            {
                "continue": "collect_history",
                "wait": END
            }
        )

        workflow.add_conditional_edges(
            "collect_history",
            self._should_continue_to_symptoms,
            {
                "continue": "collect_symptoms",
                "wait": END
            }
        )

        workflow.add_conditional_edges(
            "collect_symptoms",
            self._should_continue_to_medications,
            {
                "continue": "collect_medications",
                "wait": END
            }
        )

        workflow.add_conditional_edges(
            "collect_medications",
            self._should_continue_to_image,
            {
                "continue": "request_image",
                "wait": END
            }
        )

        # 조건부 이미지 분석
        workflow.add_conditional_edges(
            "request_image",
            self._should_analyze_image,
            {
                "analyze": "analyze_image",
                "skip": "prepare_case_context"
            }
        )

        workflow.add_conditional_edges(
            "analyze_image",
            self._should_prepare_context,
            {
                "continue": "prepare_case_context",
                "wait": END
            }
        )

        workflow.add_edge("prepare_case_context", END)

        return workflow.compile()

    def _greeting_node(self, state: AdminWorkflowState) -> AdminWorkflowState:
        """인사 및 서비스 소개"""
        # greeting 메시지가 이미 추가되었는지 확인
        messages = state.get("messages", [])
        greeting_exists = any(
            msg.get("stage") == "greeting" for msg in messages
        )

        if not greeting_exists:
            greeting_message = """
            안녕하세요! 🏥 Multi-Agent 의료 상담 서비스입니다.

            저는 Admin Agent로, 여러분의 의료 상담을 위한 정보를 수집하겠습니다.

            📋 진행 과정:
            1. 기본 정보 수집 (인구학적 정보)
            2. 과거 병력 및 가족력
            3. 현재 증상 상세 문진
            4. 복용 중인 약물 정보
            5. 방사선 이미지 분석 (선택사항)
            6. 전문 의료진 Multi-Agent 상담으로 연결

            ⚠️ 본 서비스는 교육 및 참고 목적이며,
            확정적 진단이나 치료를 제공하지 않습니다.
            응급상황에서는 즉시 응급실을 방문하시기 바랍니다.

            편안하게 말씀해 주세요. 시작하겠습니다!
            """

            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": greeting_message.strip(),
                "stage": "greeting"
            })

        state["current_stage"] = "demographics"
        return state

    def _collect_demographics_node(self,
                                   state: AdminWorkflowState
                                   ) -> AdminWorkflowState:
        """인구학적 정보 수집"""
        demographics = state.get("demographics", {})
        # 사용자 입력이 아직 없으면 질문 표시
        if not demographics.get("raw_input"):
            question = """
            기본 정보를 알려주세요:

            1. 나이 (또는 연령대)
            2. 성별
            3. 직업 (선택사항)
            4. 거주 지역 (선택사항)

            예: "35세 여성, 간호사, 서울 거주" 또는 "40대 남성"
            """

            state["messages"].append({
                "role": "assistant",
                "content": question.strip(),
                "stage": "demographics"
            })
            state["current_stage"] = "demographics"

        return state

    def _collect_history_node(self,
                              state: AdminWorkflowState
                              ) -> AdminWorkflowState:
        """과거 병력 및 가족력 수집"""
        history = state.get("history", {})
        if not history.get("raw_input"):
            question = """
            과거 병력 및 가족력에 대해 알려주세요:

            📋 과거 병력:
            1. 기존에 진단받은 질환이 있으신가요?
            2. 과거 수술이나 입원 경험이 있으신가요?
            3. 알레르기가 있으신가요?

            👨‍👩‍👧‍👦 가족력:
            4. 가족 중 특별한 질환이 있으신가요?
               (심장병, 당뇨, 암 등)

            해당사항이 없으시면 "없습니다"라고 말씀해 주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": question.strip(),
                "stage": "history"
            })
            state["current_stage"] = "history"

        return state

    def _collect_symptoms_node(self,
                               state: AdminWorkflowState
                               ) -> AdminWorkflowState:
        """현재 증상 상세 수집"""
        symptoms = state.get("symptoms", {})
        if not symptoms.get("raw_input"):
            question = """
            현재 증상에 대해 자세히 알려주세요:

            🩺 주 증상:
            1. 어떤 증상이 있으신가요?
            2. 언제부터 시작되었나요?
            3. 증상의 강도는 어떤가요? (1-10점)
            4. 증상의 양상은?
               (지속적/간헐적/악화/호전)

            📍 부가 정보:
            5. 증상이 악화되거나 완화되는
               특정 상황이 있나요?
            6. 동반되는 다른 증상이 있나요?

            증상이 없으시면 "검진 목적"이라고 말씀해 주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": question.strip(),
                "stage": "symptoms"
            })
            state["current_stage"] = "symptoms"

        return state

    def _collect_medications_node(self,
                                  state: AdminWorkflowState
                                  ) -> AdminWorkflowState:
        """복용 중인 약물 정보 수집"""
        meds = state.get("meds", {})
        if not meds.get("raw_input"):
            question = """
            현재 복용 중인 약물에 대해 알려주세요:

            💊 복용 중인 약물:
            1. 처방약이 있으신가요?
               (약물명, 용량, 복용 기간)
            2. 일반의약품이나 건강기능식품을 드시나요?
            3. 한약이나 민간요법을 사용하시나요?

            복용하는 약물이 없으시면 "없습니다"라고
            말씀해 주세요.
            """

            state["messages"].append({
                "role": "assistant",
                "content": question.strip(),
                "stage": "medications"
            })
            state["current_stage"] = "medications"

        return state

    def _request_image_node(self,
                            state: AdminWorkflowState
                            ) -> AdminWorkflowState:
        """방사선 이미지 업로드 요청"""
        request_message = """
        📷 방사선 이미지 업로드 (선택사항)

        병원에서 촬영하신 의료 이미지가 있으시면
        업로드해 주세요:

        📋 지원 이미지:
        - X-ray (흉부, 복부, 골절 등)
        - CT 스캔
        - MRI 이미지
        - 초음파 이미지
        - 기타 방사선 검사 이미지

        📁 지원 형식: PNG, JPG, JPEG, DICOM

        ⚠️ 개인정보 보호:
        - 환자 정보가 포함된 부분은 가려주세요
        - 의료진 소견이 포함된 텍스트는
          제거해 주세요

        이미지가 없으시면 "이미지 없음"이라고
        말씀해 주세요.
        """

        state["messages"].append({
            "role": "assistant",
            "content": request_message.strip(),
            "stage": "image_request"
        })
        state["current_stage"] = "image_request"

        return state

    def _analyze_image_node(self,
                            state: AdminWorkflowState
                            ) -> AdminWorkflowState:
        """MedBLIP을 사용한 이미지 분석"""
        if (state.get("uploaded_image") and
                not state.get("medblip_findings")):
            try:
                # MedBLIP 분석 수행
                analysis_result = self.medblip_tool.analyze_medical_image(
                    state["uploaded_image"]
                )

                # AGENTS.md 명세에 따른 구조화된 findings
                medblip_findings = {
                    "description": analysis_result,
                    "entities": [],  # 향후 UMLS CUI 코드 추출 시 사용
                    "impression": None  # 향후 요약 기능 추가 시 사용
                }

                state["medblip_findings"] = medblip_findings

                analysis_message = f"""
                🔍 이미지 분석이 완료되었습니다!

                **MedBLIP 분석 결과:**
                {analysis_result}

                이제 수집된 모든 정보를 종합하여
                전문 의료진 Multi-Agent 상담으로 연결해드리겠습니다.
                """

                state["messages"].append({
                    "role": "assistant",
                    "content": analysis_message.strip(),
                    "stage": "image_analysis"
                })

            except Exception as e:
                error_message = f"""
                ⚠️ 이미지 분석 중 오류가 발생했습니다: {str(e)}

                다른 이미지를 시도해주시거나,
                이미지 없이 진행하시겠습니까?
                """

                state["error_message"] = str(e)
                state["messages"].append({
                    "role": "assistant",
                    "content": error_message.strip(),
                    "stage": "image_analysis_error"
                })

        return state

    def _prepare_case_context_node(self,
                                   state: AdminWorkflowState
                                   ) -> AdminWorkflowState:
        """CaseContext 준비 및 Supervisor Agent로 핸드오프"""

        # 자유 텍스트 생성 (모든 사용자 입력 통합)
        user_messages = [msg for msg in state.get("messages", [])
                         if msg.get("role") == "user"]
        free_text = " ".join([msg.get("content", "")
                              for msg in user_messages])

        state["free_text"] = free_text
        state["conversation_complete"] = True

        handoff_message = """
        📋 정보 수집이 완료되었습니다!

        **수집된 정보:**
        ✅ 인구학적 정보 (나이, 성별, 직업 등)
        ✅ 과거 병력 및 가족력
        ✅ 현재 증상 상세 정보
        ✅ 복용 중인 약물 정보
        ✅ 방사선 이미지 분석 (해당시)

        이제 전문 의료진 Multi-Agent 패널이 다음 과정을 진행합니다:

        🏥 다음 단계:
        1. Supervisor Agent가 3명의 Doctor Agent와 협업
        2. 최대 13라운드의 집중적인 의학적 검토
        3. 진단 가설 및 권장 검사 도출
        4. 합의된 결론을 환자 친화적 언어로 번역

        잠시만 기다려주세요... 🔄
        """

        state["messages"].append({
            "role": "assistant",
            "content": handoff_message.strip(),
            "stage": "handoff"
        })

        return state

    def _should_analyze_image(self, state: AdminWorkflowState) -> str:
        """이미지 분석 여부 결정"""
        if state.get("uploaded_image"):
            return "analyze"
        # 사용자가 "이미지 없음"이라고 답했거나 이미지를 업로드하지 않은 경우
        user_messages = [
            msg for msg in state.get("messages", [])
            if msg.get("role") == "user"
        ]
        if user_messages:
            last_user_message = user_messages[-1].get("content", "").lower()
            if any(keyword in last_user_message for keyword in
                   ["이미지 없음", "없음", "없습니다", "skip"]):
                return "skip"
        return "skip"

    def _should_continue_to_demographics(
        self, state: AdminWorkflowState
    ) -> str:
        """인구학적 정보 수집 단계로 진행할지 결정"""
        # 인사 후 항상 다음 단계로 진행
        _ = state  # Suppress unused parameter warning
        return "continue"

    def _should_continue_to_history(self, state: AdminWorkflowState) -> str:
        """병력 수집 단계로 진행할지 결정"""
        # 인구학적 정보가 수집되었는지 확인
        demographics = state.get("demographics", {})
        if demographics and demographics.get("raw_input"):
            return "continue"
        return "wait"

    def _should_continue_to_symptoms(self, state: AdminWorkflowState) -> str:
        """증상 수집 단계로 진행할지 결정"""
        # 병력 정보가 수집되었는지 확인
        history = state.get("history", {})
        if history and history.get("raw_input"):
            return "continue"
        return "wait"

    def _should_continue_to_medications(
        self, state: AdminWorkflowState
    ) -> str:
        """약물 수집 단계로 진행할지 결정"""
        # 증상 정보가 수집되었는지 확인
        symptoms = state.get("symptoms", {})
        if symptoms and symptoms.get("raw_input"):
            return "continue"
        return "wait"

    def _should_continue_to_image(self, state: AdminWorkflowState) -> str:
        """이미지 요청 단계로 진행할지 결정"""
        # 약물 정보가 수집되었는지 확인
        meds = state.get("meds", {})
        if meds and meds.get("raw_input"):
            return "continue"
        return "wait"

    def _should_prepare_context(self, state: AdminWorkflowState) -> str:
        """컨텍스트 준비 단계로 진행할지 결정"""
        # 이미지 분석이 완료되었으면 진행
        _ = state  # Suppress unused parameter warning
        return "continue"

    def get_case_context(self, state: AdminWorkflowState) -> CaseContext:
        """완성된 CaseContext 반환"""
        return CaseContext(
            demographics=state.get("demographics", {}),
            symptoms=state.get("symptoms", {}),
            history=state.get("history", {}),
            meds=state.get("meds", {}),
            vitals=state.get("vitals", {}),  # 향후 추가 가능
            medblip_findings=state.get("medblip_findings", {}),
            free_text=state.get("free_text", "")
        )
