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
from langchain_openai import ChatOpenAI

from app.agents.conversation_manager import CaseContext, PatientSummary
from app.agents.admin_workflow import AdminWorkflow, AdminWorkflowState
from app.tools.medblip_tool import MedBLIPTool

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

        # LLM 초기화 (환자 친화적 재작성용)
        if self.api_key:
            logger.info("🔑 OpenAI API 키 발견 - LLM 초기화 중")
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.7
            )
            logger.info("✅ OpenAI LLM 초기화 완료")
        else:
            logger.warning("⚠️ OpenAI API 키 없음 - 오프라인 모드로 동작")
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

            logger.info("✅ 이미지 분석 및 컨텍스트 준비 완료")
        except Exception as e:
            logger.error(f"❌ 이미지 분석 중 오류: {str(e)}")
            # 오류 발생 시 이미지 없이 진행
            result = self.admin_workflow._prepare_case_context_node(self.current_state)
            self.current_state.update(result)

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

        # LLM을 사용한 환자 친화적 재작성
        prompt = f"""
다음 의료 전문가 합의 결과를 환자가 이해하기 쉬운 한국어로 재작성해주세요.

전문가 합의 결과:
{supervisor_decision}

재작성 원칙:
1. 전문 용어를 일반인이 이해할 수 있는 언어로 변경
2. 불확실성과 리스크 프레이밍 포함
3. 전문의 상담 받을 것을 권고
4. 응급상황에서는 즉시 응급실 방문 안내
5. 교육 및 참고 목적임을 명시
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary_text = response.content
        except Exception:
            # LLM 오류 시 기본 템플릿 사용
            return self._create_offline_patient_summary(supervisor_decision)

        return PatientSummary(
            summary_text=summary_text,
            disclaimers=[
                "이 상담 결과는 교육 및 참고 목적입니다.",
                "확정적 진단이나 치료를 제공하지 않습니다.",
                "반드시 전문의와 상담하시기 바랍니다.",
                "응급상황에서는 즉시 119를 호출하거나 응급실을 방문하세요."
            ]
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
            disclaimers=[
                "이 상담 결과는 교육 및 참고 목적입니다.",
                "확정적 진단이나 치료를 제공하지 않습니다.",
                "반드시 전문의와 상담하시기 바랍니다.",
                "응급상황에서는 즉시 119를 호출하거나 응급실을 방문하세요."
            ]
        )

    def reset(self):
        """에이전트 상태 초기화"""
        logger.info("🔄 AdminAgent 상태 초기화")
        self.current_state = None