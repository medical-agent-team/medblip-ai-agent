#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent Medical Consultation Orchestrator

LangGraph 기반으로 Admin, Doctor, Supervisor Agent를 조율하는 메인 오케스트레이터입니다.
모든 에이전트가 공유 상태(SharedMedicalState)를 사용합니다.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from app.agents.shared_state import (
    SharedMedicalState,
    StateManager,
    update_state_demographics,
    update_state_history,
    update_state_symptoms,
    update_state_medications,
    update_state_medblip
)
from app.agents.admin_agent import AdminAgent
from app.agents.doctor_agent import DoctorAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.tools.medblip_tool import MedBLIPTool

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Multi-Agent 의료 상담 시스템의 메인 오케스트레이터

    모든 에이전트가 SharedMedicalState를 공유하며,
    LangGraph를 통해 워크플로우를 관리합니다.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key

        # 개별 에이전트 초기화 (기존 코드 재사용)
        self.admin_agent = AdminAgent(openai_api_key)
        self.doctor_agents = [
            DoctorAgent(doctor_id="doctor_1", openai_api_key=openai_api_key),
            DoctorAgent(doctor_id="doctor_2", openai_api_key=openai_api_key),
            DoctorAgent(doctor_id="doctor_3", openai_api_key=openai_api_key)
        ]
        self.supervisor_agent = SupervisorAgent(openai_api_key)
        self.medblip_tool = MedBLIPTool()

        # LangGraph 워크플로우 생성
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(SharedMedicalState)

        # 노드 추가
        workflow.add_node("admin_intake", self._admin_intake_node)
        workflow.add_node("medblip_analysis", self._medblip_analysis_node)
        workflow.add_node("doctor_consultation", self._doctor_consultation_node)
        workflow.add_node("supervisor_consensus", self._supervisor_consensus_node)
        workflow.add_node("patient_summary", self._patient_summary_node)

        # 엣지 추가
        workflow.add_edge(START, "admin_intake")
        workflow.add_conditional_edges(
            "admin_intake",
            self._should_analyze_image,
            {
                "analyze": "medblip_analysis",
                "skip": "doctor_consultation"
            }
        )
        workflow.add_edge("medblip_analysis", "doctor_consultation")
        workflow.add_edge("doctor_consultation", "supervisor_consensus")
        workflow.add_conditional_edges(
            "supervisor_consensus",
            self._should_continue_consultation,
            {
                "continue": "doctor_consultation",
                "finalize": "patient_summary"
            }
        )
        workflow.add_edge("patient_summary", END)

        return workflow.compile()

    # === 노드 구현 ===

    def _admin_intake_node(self, state: SharedMedicalState) -> SharedMedicalState:
        """Admin Agent - 환자 정보 수집"""
        logger.info("🏥 Admin 정보 수집 단계 시작")

        # 현재 단계 확인
        current_stage = state.get("current_stage", "demographics")

        if current_stage == "demographics" and not state["stage_completed"].get("demographics", False):
            # 기본정보 수집 로직 (실제 구현에서는 UI와 연동)
            demographics_input = self._get_user_input_for_stage("demographics")
            state = update_state_demographics(state, demographics_input)
            logger.info("✅ 기본정보 수집 완료")

        elif current_stage == "history" and not state["stage_completed"].get("history", False):
            # 병력 수집
            history_input = self._get_user_input_for_stage("history")
            state = update_state_history(state, history_input)
            logger.info("✅ 병력정보 수집 완료")

        elif current_stage == "symptoms" and not state["stage_completed"].get("symptoms", False):
            # 증상 수집
            symptoms_input = self._get_user_input_for_stage("symptoms")
            state = update_state_symptoms(state, symptoms_input)
            logger.info("✅ 증상정보 수집 완료")

        elif current_stage == "medications" and not state["stage_completed"].get("medications", False):
            # 약물 수집
            medications_input = self._get_user_input_for_stage("medications")
            state = update_state_medications(state, medications_input)
            logger.info("✅ 약물정보 수집 완료")

        # 다음 단계 결정
        state["current_stage"] = self._get_next_intake_stage(state)

        return state

    def _medblip_analysis_node(self, state: SharedMedicalState) -> SharedMedicalState:
        """MedBLIP 이미지 분석"""
        logger.info("🖼️ MedBLIP 이미지 분석 시작")

        if state.get("uploaded_image") and not state.get("image_processed", False):
            try:
                # MedBLIP 분석 수행
                caption = self.medblip_tool.analyze_medical_image(state["uploaded_image"])
                state = update_state_medblip(state, caption)
                logger.info(f"✅ MedBLIP 분석 완료: {caption[:100]}...")

            except Exception as e:
                logger.error(f"❌ MedBLIP 분석 실패: {str(e)}")
                state["error_messages"].append(f"이미지 분석 실패: {str(e)}")

        return state

    def _doctor_consultation_node(self, state: SharedMedicalState) -> SharedMedicalState:
        """3명의 Doctor Agent 상담"""
        logger.info(f"👩‍⚕️ Doctor 상담 시작 - 라운드 {state['current_round'] + 1}")

        state["current_round"] += 1

        # 각 Doctor Agent로부터 의견 수집
        for doctor_agent in self.doctor_agents:
            try:
                # 상태를 Doctor Agent가 이해할 수 있는 형식으로 변환
                case_context = self._convert_state_to_case_context(state)

                # Doctor 의견 생성
                opinion = doctor_agent.analyze_case(
                    case_context=case_context,
                    round_number=state["current_round"],
                    peer_opinions=list(state["doctor_opinions"].values())
                )

                # 상태에 의견 추가
                StateManager.add_doctor_opinion(state, doctor_agent.doctor_id, opinion)
                logger.info(f"✅ {doctor_agent.doctor_id} 의견 수집 완료")

            except Exception as e:
                logger.error(f"❌ {doctor_agent.doctor_id} 의견 수집 실패: {str(e)}")
                state["error_messages"].append(f"{doctor_agent.doctor_id} 의견 수집 실패: {str(e)}")

        return state

    def _supervisor_consensus_node(self, state: SharedMedicalState) -> SharedMedicalState:
        """Supervisor Agent - 합의 도출"""
        logger.info("👨‍⚕️ Supervisor 합의 평가 시작")

        try:
            # 현재 라운드의 Doctor 의견들 수집
            current_opinions = list(state["doctor_opinions"].values())

            # Supervisor가 합의 평가
            decision = self.supervisor_agent.evaluate_consensus(
                doctor_opinions=current_opinions,
                round_number=state["current_round"],
                max_rounds=state["max_rounds"]
            )

            # 상태에 Supervisor 결정 추가
            StateManager.add_supervisor_decision(state, decision)

            logger.info(f"✅ Supervisor 평가 완료 - 합의 도달: {decision.get('consensus_reached', False)}")

        except Exception as e:
            logger.error(f"❌ Supervisor 평가 실패: {str(e)}")
            state["error_messages"].append(f"Supervisor 평가 실패: {str(e)}")

        return state

    def _patient_summary_node(self, state: SharedMedicalState) -> SharedMedicalState:
        """Admin Agent - 환자 친화적 요약 생성"""
        logger.info("📋 환자 요약 생성 시작")

        try:
            # 최종 Supervisor 결정 가져오기
            if state["supervisor_decisions"]:
                final_decision = state["supervisor_decisions"][-1]

                # Admin Agent를 통해 환자 친화적 요약 생성
                summary = self.admin_agent.create_patient_summary(final_decision)
                state["patient_summary"] = summary
                state["consultation_complete"] = True

                logger.info("✅ 환자 요약 생성 완료")
            else:
                logger.warning("⚠️ Supervisor 결정이 없어 요약을 생성할 수 없습니다")

        except Exception as e:
            logger.error(f"❌ 환자 요약 생성 실패: {str(e)}")
            state["error_messages"].append(f"환자 요약 생성 실패: {str(e)}")

        return state

    # === 조건부 라우팅 함수들 ===

    def _should_analyze_image(self, state: SharedMedicalState) -> str:
        """이미지 분석이 필요한지 확인"""
        if state.get("uploaded_image") and not state.get("image_processed", False):
            return "analyze"
        return "skip"

    def _should_continue_consultation(self, state: SharedMedicalState) -> str:
        """상담을 계속할지 종료할지 결정"""
        if not state["supervisor_decisions"]:
            return "continue"

        latest_decision = state["supervisor_decisions"][-1]

        # 합의에 도달했거나 최대 라운드에 도달한 경우
        if (latest_decision.get("consensus_reached", False) or
            state["current_round"] >= state["max_rounds"]):
            return "finalize"

        return "continue"

    # === 헬퍼 메소드들 ===

    def _get_user_input_for_stage(self, stage: str) -> str:
        """실제 구현에서는 UI로부터 사용자 입력을 받음 (현재는 더미)"""
        # 이 메소드는 실제 UI 연동 시 구현
        return f"사용자 입력 for {stage}"

    def _get_next_intake_stage(self, state: SharedMedicalState) -> str:
        """다음 정보 수집 단계 결정"""
        stages = ["demographics", "history", "symptoms", "medications"]

        for stage in stages:
            if not state["stage_completed"].get(stage, False):
                return stage

        return "intake_complete"

    def _convert_state_to_case_context(self, state: SharedMedicalState) -> Dict[str, Any]:
        """SharedMedicalState를 기존 CaseContext 형식으로 변환"""
        return {
            "demographics": state["demographics"],
            "symptoms": state["symptoms"],
            "history": state["history"],
            "meds": state["medications"],
            "medblip_findings": {
                "description": state["medblip_analysis"].get("caption", ""),
                "entities": state["medblip_analysis"].get("entities", []),
                "impression": state["medblip_analysis"].get("impression")
            },
            "free_text": f"{state['demographics'].get('raw_input', '')} {state['history'].get('raw_input', '')} {state['symptoms'].get('raw_input', '')} {state['medications'].get('raw_input', '')}"
        }

    # === 공개 API ===

    def start_consultation(self, session_id: str) -> SharedMedicalState:
        """새로운 상담 세션 시작"""
        logger.info(f"🆕 새로운 상담 세션 시작: {session_id}")

        # 초기 상태 생성
        initial_state = StateManager.create_initial_state(session_id)

        return initial_state

    def process_consultation(self, state: SharedMedicalState) -> SharedMedicalState:
        """상담 처리 - LangGraph 워크플로우 실행"""
        logger.info("🔄 상담 워크플로우 실행 시작")

        try:
            # LangGraph 워크플로우 실행
            result = self.workflow.invoke(state)
            logger.info("✅ 상담 워크플로우 완료")
            return result

        except Exception as e:
            logger.error(f"❌ 상담 워크플로우 실행 실패: {str(e)}")
            state["error_messages"].append(f"워크플로우 실행 실패: {str(e)}")
            return state

    def get_consultation_status(self, state: SharedMedicalState) -> Dict[str, Any]:
        """상담 진행 상태 반환"""
        return {
            "session_id": state["session_id"],
            "current_stage": state["current_stage"],
            "current_round": state["current_round"],
            "intake_complete": StateManager.is_intake_complete(state),
            "consultation_complete": state.get("consultation_complete", False),
            "error_count": len(state.get("error_messages", []))
        }


__all__ = ['MultiAgentOrchestrator']