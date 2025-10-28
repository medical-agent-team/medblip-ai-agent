#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admin Agent - Multi-agent system의 관리 에이전트

AGENTS.md 명세에 따른 역할:
- 사용자 입력 수집 (symptoms, history, free-text)
- 이미지 업로드 및 MedBLIP 분석을 통한 구조화된 텍스트 findings 생성
- 모든 컨텍스트와 작업 브리프를 Supervisor Agent로 패키징
- 합의 후 최종 의료 출력을 환자 친화적 한국어로 재작성
"""

import os
import logging
from typing import Dict, Any, Optional
from PIL import Image
from langchain_core.messages import HumanMessage

from app.core.llm_factory import get_llm_for_agent
from app.core.observability import get_callbacks
from app.agents.conversation_manager import CaseContext, PatientSummary
from app.agents.admin_workflow import AdminWorkflow, AdminWorkflowState
from app.tools.medblip_tool import MedBLIPTool
from app.agents.prompts.admin_prompts import (
    ADMIN_PATIENT_SUMMARY_PROMPT,
    ADMIN_SAFETY_DISCLAIMERS
)

# Docker 로그에서 확인 가능한 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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
        logger.info("🚀 AdminAgent 초기화 시작")

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # LLM 초기화 (환자 친화적 재작성용) - vLLM/Langfuse 지원
        try:
            logger.info("🔑 LLM 초기화 중 (vLLM endpoint: {})".format(
                os.getenv("OPENAI_API_BASE", "OpenAI API")
            ))
            callbacks = get_callbacks()
            self.llm = get_llm_for_agent(
                agent_type="admin",
                api_key=self.api_key,
                callbacks=callbacks
            )
            logger.info("✅ LLM 초기화 완료 (Langfuse callbacks: {})".format(len(callbacks)))
        except Exception as e:
            logger.warning(f"⚠️ LLM 초기화 실패: {e} - 오프라인 모드로 동작")
            self.llm = None

        # MedBLIP 도구 초기화
        logger.info("🔧 MedBLIP 도구 초기화 중...")
        try:
            self.medblip_tool = MedBLIPTool()
            logger.info("✅ MedBLIP 도구 초기화 완료")
        except Exception as e:
            logger.error(f"❌ MedBLIP 도구 초기화 실패: {str(e)}")
            raise

        # Admin 워크플로우 초기화
        logger.info("🔄 Admin 워크플로우 초기화 중...")
        try:
            self.admin_workflow = AdminWorkflow(self.medblip_tool)
            logger.info("✅ Admin 워크플로우 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Admin 워크플로우 초기화 실패: {str(e)}")
            raise

        # 현재 워크플로우 상태
        self.current_state: Optional[AdminWorkflowState] = None

        logger.info("🎉 AdminAgent 초기화 성공적으로 완료")

    def start_intake(self) -> Dict[str, Any]:
        """새로운 intake 세션 시작"""
        logger.info("🆕 새로운 intake 세션 시작")

        self.current_state = AdminWorkflowState(
            messages=[],
            current_stage="greeting",
            demographics={},
            history={},
            symptoms={},
            meds={},
            vitals={},
            medblip_findings={},
            free_text="",
            uploaded_image=None,
            conversation_complete=False,
            error_message=None
        )

        # 인사 메시지와 첫 번째 질문 실행
        try:
            logger.info("🔄 인사 메시지 생성 중...")
            result = self.admin_workflow._greeting_node(self.current_state)
            self.current_state.update(result)

            logger.info("🔄 인구학적 정보 수집 질문 생성 중...")
            result = self.admin_workflow._collect_demographics_node(self.current_state)
            self.current_state.update(result)

            logger.info("✅ Intake 세션 시작 완료")
            return self._format_response(success=True)
        except Exception as e:
            logger.error(f"❌ Intake 세션 시작 실패: {str(e)}")
            return self._format_response(success=False, error=str(e))

    def process_user_input(self, user_input: str,
                           image: Optional[Image.Image] = None
                           ) -> Dict[str, Any]:
        """사용자 입력 처리 및 워크플로우 진행"""
        logger.info(f"📝 사용자 입력 처리: {user_input[:50]}...")

        if not self.current_state:
            logger.error("❌ 세션이 시작되지 않음")
            return self._format_response(
                success=False,
                error="세션이 시작되지 않았습니다. start_intake()를 먼저 호출해주세요."
            )

        # 사용자 메시지 추가
        self.current_state["messages"].append({
            "role": "user",
            "content": user_input,
            "stage": self.current_state["current_stage"]
        })

        # 이미지 추가 (있는 경우)
        if image:
            logger.info("🖼️ 이미지 업로드 감지")
            self.current_state["uploaded_image"] = image

        # 사용자 입력에서 정보 추출 및 업데이트
        self._extract_and_update_info(user_input)

        # 워크플로우 다음 단계 진행
        try:
            if not self.current_state.get("conversation_complete"):
                logger.info("🔄 다음 워크플로우 단계 실행 중...")
                self._execute_next_workflow_step()

                # Log current state after workflow step
                logger.info(f"📊 [Admin] 워크플로우 진행 상태:")
                logger.info(f"   - 현재 단계: {self.current_state.get('current_stage')}")
                logger.info(f"   - 완료 여부: {self.current_state.get('conversation_complete')}")
                if self.current_state.get('conversation_complete'):
                    logger.info(f"   - 수집된 데이터: Demographics, History, Symptoms, MedBLIP findings")

            logger.info("✅ 사용자 입력 처리 완료")
            return self._format_response(success=True)
        except Exception as e:
            logger.error(f"❌ 사용자 입력 처리 실패: {str(e)}")
            return self._format_response(success=False, error=str(e))

    def _execute_next_workflow_step(self):
        """현재 상태에 따라 다음 워크플로우 단계만 실행"""
        current_stage = self.current_state.get("current_stage", "demographics")

        logger.info(f"📍 현재 단계: {current_stage}")

        # 현재 단계에 따라 적절한 노드 실행
        if current_stage == "demographics":
            self._check_and_move_to_history()
        elif current_stage == "history":
            self._check_and_move_to_symptoms()
        elif current_stage == "symptoms":
            self._check_and_move_to_medications()
        elif current_stage == "medications":
            self._check_and_move_to_image()
        elif current_stage == "image_request":
            self._check_and_handle_image()
        elif current_stage == "image_analysis":
            self._check_and_handle_image()

        # 이미지가 업로드된 경우 즉시 분석 수행
        if (self.current_state.get("uploaded_image") and
                not self.current_state.get("medblip_findings")):
            logger.info("🖼️ 새로운 이미지 감지 - 즉시 분석 수행")
            self._perform_image_analysis()

    def _check_and_move_to_history(self):
        """인구학적 정보가 수집되었으면 병력 단계로 이동"""
        demographics = self.current_state.get("demographics", {})
        if demographics and demographics.get("raw_input"):
            logger.info("✅ 인구학적 정보 수집 완료 - 병력 단계로 이동")
            result = self.admin_workflow._collect_history_node(self.current_state)
            self.current_state.update(result)

    def _check_and_move_to_symptoms(self):
        """병력 정보가 수집되었으면 증상 단계로 이동"""
        history = self.current_state.get("history", {})
        if history and history.get("raw_input"):
            logger.info("✅ 병력 정보 수집 완료 - 증상 단계로 이동")
            result = self.admin_workflow._collect_symptoms_node(self.current_state)
            self.current_state.update(result)

    def _check_and_move_to_medications(self):
        """증상 정보가 수집되었으면 약물 단계로 이동"""
        symptoms = self.current_state.get("symptoms", {})
        if symptoms and symptoms.get("raw_input"):
            logger.info("✅ 증상 정보 수집 완료 - 약물 단계로 이동")
            result = self.admin_workflow._collect_medications_node(self.current_state)
            self.current_state.update(result)

    def _check_and_move_to_image(self):
        """약물 정보가 수집되었으면 이미지 단계로 이동"""
        meds = self.current_state.get("meds", {})
        if meds and meds.get("raw_input"):
            logger.info("✅ 약물 정보 수집 완료 - 이미지 단계로 이동")
            result = self.admin_workflow._request_image_node(self.current_state)
            self.current_state.update(result)

    def _check_and_handle_image(self):
        """이미지 처리 또는 최종 단계로 이동"""
        if self.current_state.get("uploaded_image"):
            logger.info("🖼️ 이미지 분석 시작")
            result = self.admin_workflow._analyze_image_node(self.current_state)
            self.current_state.update(result)
            # 이미지 분석 후 컨텍스트 준비
            result = self.admin_workflow._prepare_case_context_node(self.current_state)
            self.current_state.update(result)
            # 분석 완료 상태로 업데이트
            self.current_state["current_stage"] = "deliberation"
            logger.info("✅ 이미지 분석 완료 - deliberation 단계로 이동")
        else:
            # 사용자가 이미지 없음을 선택한 경우
            user_messages = [
                msg for msg in self.current_state.get("messages", [])
                if msg.get("role") == "user"
            ]
            if user_messages:
                last_message = user_messages[-1].get("content", "").lower()
                if any(keyword in last_message for keyword in
                       ["이미지 없음", "없음", "없습니다", "skip"]):
                    logger.info("⏭️ 이미지 건너뛰기 - 최종 단계로 이동")
                    result = self.admin_workflow._prepare_case_context_node(self.current_state)
                    self.current_state.update(result)
                    # 분석 완료 상태로 업데이트
                    self.current_state["current_stage"] = "deliberation"
                    logger.info("✅ 컨텍스트 준비 완료 - deliberation 단계로 이동")

    def _perform_image_analysis(self):
        """이미지 분석 수행"""
        logger.info("🔍 MedBLIP 이미지 분석 시작")
        try:
            result = self.admin_workflow._analyze_image_node(self.current_state)
            self.current_state.update(result)

            # 분석 완료 후 컨텍스트 준비
            logger.info("📊 이미지 분석 완료 - 컨텍스트 준비 중")
            result = self.admin_workflow._prepare_case_context_node(self.current_state)
            self.current_state.update(result)

            # 분석 완료 상태로 업데이트
            self.current_state["current_stage"] = "deliberation"
            logger.info("✅ 이미지 분석 및 컨텍스트 준비 완료 - deliberation 단계로 이동")
        except Exception as e:
            logger.error(f"❌ 이미지 분석 중 오류: {str(e)}")
            # 오류 발생 시 이미지 없이 진행
            result = self.admin_workflow._prepare_case_context_node(self.current_state)
            self.current_state.update(result)
            # 분석 완료 상태로 업데이트
            self.current_state["current_stage"] = "deliberation"
            logger.info("✅ 컨텍스트 준비 완료 - deliberation 단계로 이동")

    def _extract_and_update_info(self, user_input: str):
        """사용자 입력에서 의료 정보 추출 및 상태 업데이트"""
        stage = self.current_state["current_stage"]

        if stage == "demographics":
            # 인구학적 정보 추출
            demographics = {"raw_input": user_input}
            if any(keyword in user_input for keyword in
                   ["살", "세", "년생"]):
                demographics["age_mentioned"] = True
            if any(keyword in user_input for keyword in
                   ["남", "여", "남성", "여성"]):
                demographics["gender_mentioned"] = True
            if any(keyword in user_input for keyword in
                   ["직", "업무", "일"]):
                demographics["occupation_mentioned"] = True

            self.current_state["demographics"].update(demographics)

        elif stage == "history":
            # 병력 정보 추출
            history = {"raw_input": user_input}
            if "없" in user_input:
                history["has_history"] = False
            else:
                history["has_history"] = True

            self.current_state["history"].update(history)

        elif stage == "symptoms":
            # 증상 정보 추출
            symptoms = {"raw_input": user_input}
            if any(keyword in user_input for keyword in
                   ["없", "검진"]):
                symptoms["has_symptoms"] = False
            else:
                symptoms["has_symptoms"] = True

            self.current_state["symptoms"].update(symptoms)

        elif stage == "medications":
            # 약물 정보 추출
            meds = {"raw_input": user_input}
            if "없" in user_input:
                meds["has_medications"] = False
            else:
                meds["has_medications"] = True

            self.current_state["meds"].update(meds)

    def _format_response(self, success: bool,
                         error: Optional[str] = None) -> Dict[str, Any]:
        """응답 포맷 표준화"""
        if not success:
            return {
                "success": False,
                "error": error,
                "messages": [{
                    "role": "assistant",
                    "content": f"처리 중 오류가 발생했습니다: {error}"
                }]
            }

        return {
            "success": True,
            "messages": self.current_state.get("messages", []),
            "current_stage": self.current_state.get("current_stage", "unknown"),
            "conversation_complete": self.current_state.get(
                "conversation_complete", False
            ),
            "case_context": (self.get_case_context() if
                            self.current_state.get("conversation_complete")
                            else None)
        }

    def get_case_context(self) -> Optional[CaseContext]:
        """완성된 CaseContext 반환"""
        if (not self.current_state or
                not self.current_state.get("conversation_complete")):
            return None

        return self.admin_workflow.get_case_context(self.current_state)

    def create_patient_summary(self,
                               supervisor_decision: Dict[str, Any]
                               ) -> PatientSummary:
        """
        Supervisor Agent의 합의 결과를 환자 친화적 한국어로 재작성

        AGENTS.md 명세:
        - 합의 후, 최종 의료 출력을 환자 친화적 언어로 재작성하고 UI로 반환
        """
        if self.llm is None:
            # 오프라인 모드: 기본 템플릿 사용
            return self._create_offline_patient_summary(supervisor_decision)

        # LLM을 사용한 환자 친화적 재작성 (영어 -> 한국어 번역 추가)
        translate_prompt = f"""
        Translate the following medical consultation results to Korean in a patient-friendly manner.

        Medical Expert Consensus:
        {supervisor_decision}

        Translation Guidelines:
        1. Use patient-friendly Korean language (avoid complex medical terms)
        2. Maintain all important medical information
        3. Keep the safety warnings and recommendations clear
        4. Format the output in a clear, structured way

        Provide the translation in the following format:

        **상담 결과 요약**
        [Patient-friendly summary in Korean]

        **권장 사항**
        [Recommendations in Korean]

        **주의 사항**
        [Precautions in Korean]

        **다음 단계**
        [Next steps in Korean]
        """

        try:
            # Log admin patient summary generation
            logger.info("🏥 [Admin] 환자 친화적 요약 생성 중...")
            logger.info(f"📝 Supervisor Decision Input: {supervisor_decision}")

            response = self.llm.invoke([HumanMessage(content=translate_prompt)])
            summary_text = response.content

            # Log generated summary
            logger.info("📊 [Admin] 생성된 환자 요약:")
            logger.info(f"📝 Summary Text: {summary_text[:300]}...")

        except Exception as e:
            logger.error(f"❌ LLM 요약 생성 실패: {str(e)}")
            # LLM 오류 시 기본 템플릿 사용
            return self._create_offline_patient_summary(supervisor_decision)

        # 한국어 면책 조항으로 변환
        korean_disclaimers = [
            "이 상담 결과는 교육 및 참고 목적입니다.",
            "확정적 진단이나 치료를 제공하지 않습니다.",
            "반드시 전문의와 상담하시기 바랍니다.",
            "응급상황에서는 즉시 119를 호출하거나 응급실을 방문하세요."
        ]

        return PatientSummary(
            summary_text=summary_text,
            disclaimers=korean_disclaimers
        )

    def _create_offline_patient_summary(
            self,
            supervisor_decision: Dict[str, Any]
    ) -> PatientSummary:
        """오프라인 모드용 기본 환자 요약 생성"""
        consensus_hypotheses = supervisor_decision.get(
            'consensus_hypotheses', ['추가 검토 필요']
        )
        prioritized_tests = supervisor_decision.get(
            'prioritized_tests', ['전문의 상담 권장']
        )

        summary_text = f"""
        📊 상담 결과 요약

        전문 의료진 패널의 검토 결과, 다음과 같은 의견을 제시합니다:

        🔍 검토된 가능성:
        {consensus_hypotheses}

        📋 권장 검사:
        {prioritized_tests}

        이 결과는 참고용이며, 정확한 진단을 위해서는 전문의와의 상담이 필수입니다.
        """

        return PatientSummary(
            summary_text=summary_text.strip(),
            disclaimers=ADMIN_SAFETY_DISCLAIMERS
        )

    def reset(self):
        """에이전트 상태 초기화"""
        logger.info("🔄 AdminAgent 상태 초기화")
        self.current_state = None