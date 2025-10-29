#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervisor Agent - Multi-agent system의 감독 에이전트

AGENTS.md 명세에 따른 역할:
- 정확히 3명의 Doctor Agent 패널을 진단과 진단 검사에 초점을 맞춘 반복 라운드 (최대 13라운드)를 통해 조율
- 제안을 평가하고, 격차를 지적하며, 토론을 촉진하고, 합의를 유도
- 합의에 도달하면 조기 종료
"""

import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm_for_agent
from app.core.observability import get_callbacks
from app.agents.conversation_manager import (
    CaseContext, DoctorOpinion, SupervisorDecision,
    ConversationManager, SessionState
)
from app.agents.prompts.supervisor_prompts import (
    SUPERVISOR_ORCHESTRATION_PROMPT,
    SUPERVISOR_CONSENSUS_PROMPT,
    SUPERVISOR_CRITIQUE_PROMPT
)

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Multi-agent system의 Supervisor Agent

    AGENTS.md 명세에 따른 역할:
    - 정확히 3명의 Doctor Agent 패널을 진단과 진단 검사에 초점을 맞춘 반복 라운드 (최대 13라운드)를 통해 조율
    - 제안을 평가하고, 격차를 지적하며, 토론을 촉진하고, 합의를 유도
    - 합의에 도달하면 조기 종료
    """

    def __init__(self, openai_api_key: Optional[str] = None, max_rounds: int = 7):
        logger.info("🎯 SupervisorAgent 초기화 시작")

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.max_rounds = max_rounds

        # LLM 초기화 - vLLM/Langfuse 지원
        logger.info(f"🔑 LLM 초기화 중 (endpoint: {os.getenv('OPENAI_API_BASE', 'OpenAI API')})")
        callbacks = get_callbacks()
        self.llm = get_llm_for_agent(
            agent_type="supervisor",
            api_key=self.api_key,
            callbacks=callbacks
        )
        logger.info(f"✅ LLM 초기화 완료 (Langfuse: {len(callbacks)} callbacks)")

        # ConversationManager 초기화 (사용자 지정 max_rounds 적용)
        self.conversation_manager = ConversationManager(max_rounds=self.max_rounds)

        logger.info(f"✅ SupervisorAgent 초기화 완료 (max_rounds={self.max_rounds})")

    def start_deliberation(self,
                          session_id: str,
                          case_context: CaseContext,
                          doctors: List['DoctorAgent']) -> Dict[str, Any]:
        """
        Doctor Agent들과의 심의 시작

        Args:
            session_id: 세션 식별자
            case_context: 사례 컨텍스트
            doctors: 3명의 Doctor Agent 리스트

        Returns:
            심의 결과 딕셔너리
        """
        logger.info(f"🚀 심의 시작 - 세션: {session_id}")

        if len(doctors) != 3:
            raise ValueError("정확히 3명의 Doctor Agent가 필요합니다")

        # 세션 시작
        session_state = self.conversation_manager.start_session(session_id, case_context)

        try:
            # 정확히 self.max_rounds 라운드까지 반복 (조기 종료 없음)
            while not session_state.terminated and session_state.current_round < session_state.max_rounds:
                logger.info(f"🔄 라운드 {session_state.current_round + 1} 시작")

                # 라운드 시작
                round_number = self.conversation_manager.begin_round(session_id)

                # 각 Doctor Agent로부터 의견 수집
                doctor_opinions = self._collect_doctor_opinions(
                    session_id, case_context, doctors, round_number
                )

                # Supervisor 결정
                supervisor_decision = self._make_supervisor_decision(
                    session_id, case_context, doctor_opinions, round_number
                )

                # 조기 종료 로직 주석 처리 - 무조건 self.max_rounds 라운드 실행
                # if self.conversation_manager.reached_consensus(session_id):
                #     logger.info("✅ 합의 도달 - 심의 종료")
                #     self.conversation_manager.end_session(session_id, "합의 도달")
                #     break

                # 세션 상태 업데이트
                session_state = self.conversation_manager.get_session(session_id)

            if not session_state.terminated:
                logger.info(f"⏰ {self.max_rounds}라운드 완료 - 심의 종료")
                self.conversation_manager.end_session(session_id, f"{self.max_rounds}라운드 완료")

            return self._format_deliberation_result(session_id)

        except Exception as e:
            logger.error(f"❌ 심의 중 오류: {str(e)}")
            self.conversation_manager.end_session(session_id, f"오류: {str(e)}")
            raise

    def _collect_doctor_opinions(self,
                                session_id: str,
                                case_context: CaseContext,
                                doctors: List['DoctorAgent'],
                                round_number: int) -> Dict[str, DoctorOpinion]:
        """각 Doctor Agent로부터 의견 수집"""
        logger.info(f"👥 라운드 {round_number}: Doctor 의견 수집 중")

        doctor_opinions = {}

        # 이전 라운드 의견들 가져오기 (피드백용)
        session_state = self.conversation_manager.get_session(session_id)
        previous_opinions = self._get_previous_round_opinions(session_state, round_number)

        for i, doctor in enumerate(doctors):
            doctor_id = f"doctor_{i+1}"
            logger.info(f"🩺 {doctor_id} 의견 수집 중...")

            # Debug: Log peer_opinions keys being passed to this doctor
            logger.info(f"🔍 [{doctor_id}] peer_opinions keys: {list(previous_opinions.keys())}")
            logger.info(f"🔍 [{doctor_id}] doctor.doctor_id: {doctor.doctor_id}")

            try:
                opinion = doctor.provide_opinion(
                    case_context=case_context,
                    round_number=round_number,
                    peer_opinions=previous_opinions,
                    supervisor_feedback=self._get_supervisor_feedback(session_state, round_number)
                )

                # 의견 검증 및 저장
                self.conversation_manager.add_doctor_opinion(session_id, doctor_id, opinion)
                doctor_opinions[doctor_id] = opinion

                logger.info(f"✅ {doctor_id} 의견 수집 완료")

            except Exception as e:
                logger.error(f"❌ {doctor_id} 의견 수집 실패: {str(e)}")
                # 기본 의견 생성
                fallback_opinion = DoctorOpinion(
                    hypotheses=["추가 검토 필요"],
                    diagnostic_tests=["전문의 상담"],
                    reasoning=f"의견 수집 중 오류 발생: {str(e)}",
                    critique_of_peers=""
                )
                self.conversation_manager.add_doctor_opinion(session_id, doctor_id, fallback_opinion)
                doctor_opinions[doctor_id] = fallback_opinion

        return doctor_opinions

    def _make_supervisor_decision(self,
                                 session_id: str,
                                 case_context: CaseContext,
                                 doctor_opinions: Dict[str, DoctorOpinion],
                                 round_number: int) -> SupervisorDecision:
        """Supervisor 결정 생성"""
        logger.info(f"🎯 라운드 {round_number}: Supervisor 결정 생성 중")

        try:
            # 합의 분석을 위한 프롬프트 생성
            session_state = self.conversation_manager.get_session(session_id)
            previous_context = self._get_previous_rounds_context(session_state, round_number)

            consensus_prompt = self._build_consensus_prompt(
                case_context, doctor_opinions, round_number, previous_context
            )

            # LLM을 통한 합의 분석
            response = self.llm.invoke([
                SystemMessage(content=SUPERVISOR_CONSENSUS_PROMPT),
                HumanMessage(content=consensus_prompt)
            ])

            # Log supervisor output
            logger.info(f"🎯 [Supervisor] 라운드 {round_number} 합의 분석 결과:")
            logger.info(f"📝 Raw LLM Response: {response.content}")

            # 응답 파싱
            decision = self._parse_supervisor_response(response.content, round_number)

            # Log parsed decision
            logger.info(f"📊 [Supervisor] 파싱된 결정:")
            logger.info(f"   - 합의 가설: {decision.get('consensus_hypotheses', [])}")
            logger.info(f"   - 우선 검사: {decision.get('prioritized_tests', [])}")
            logger.info(f"   - 종료 여부: {decision.get('termination_reason', 'None')}")
            logger.info(f"   - 근거: {decision.get('rationale', '')[:200]}...")

            # 결정 기록
            self.conversation_manager.record_supervisor_decision(session_id, decision)

            logger.info("✅ Supervisor 결정 생성 완료")
            return decision

        except Exception as e:
            logger.error(f"❌ Supervisor 결정 생성 실패: {str(e)}")
            # 기본 결정 생성
            fallback_decision = SupervisorDecision(
                consensus_hypotheses=["추가 검토 필요"],
                prioritized_tests=["전문의 상담"],
                rationale=f"결정 생성 중 오류 발생: {str(e)}",
                termination_reason=None
            )
            self.conversation_manager.record_supervisor_decision(session_id, fallback_decision)
            return fallback_decision

    def _build_consensus_prompt(self,
                               case_context: CaseContext,
                               doctor_opinions: Dict[str, DoctorOpinion],
                               round_number: int,
                               previous_context: str = "") -> str:
        """합의 분석을 위한 프롬프트 구성"""

        # 사례 요약
        case_summary = f"""
        환자 정보:
        - 인구학적 정보: {case_context.get('demographics', {})}
        - 증상: {case_context.get('symptoms', {})}
        - 병력: {case_context.get('history', {})}
        - 약물: {case_context.get('meds', {})}
        - MedBLIP 소견: {case_context.get('medblip_findings', {})}
        - 추가 정보: {case_context.get('free_text', '')}
        """

        # Doctor 의견 요약
        opinions_summary = f"\n라운드 {round_number} Doctor 의견들:\n"
        for doctor_id, opinion in doctor_opinions.items():
            opinions_summary += f"""
        {doctor_id}:
        - 가설: {opinion.get('hypotheses', [])}
        - 진단 검사: {opinion.get('diagnostic_tests', [])}
        - 근거: {opinion.get('reasoning', '')}
        - 동료 의견에 대한 비판: {opinion.get('critique_of_peers', '')}
        """

        return f"""
        {case_summary}

        {previous_context}

        {opinions_summary}

        위 정보를 바탕으로 다음을 엄격하게 분석해주세요:
        1. Doctor들 간의 합의 수준 (정확한 일치 여부 확인)
        2. 상충하는 의견들과 그 이유
        3. 통합된 가설 후보들 (공통점 위주로)
        4. 우선순위가 높은 진단 검사들 (중복 제거)
        5. 합의 도달 여부 및 근거 (최소 2명 이상 동의시에만 "명확한 합의" 표기)

        **중요**: 단순히 가설과 검사가 있다고 합의가 아닙니다.
        적어도 3명 중 2명 이상이 동일하거나 매우 유사한 진단 가설과 검사에 동의할 때만 "명확한 합의" 또는 "완전한 합의"라고 표현하세요.
        """

    def _parse_supervisor_response(self, response: str, round_number: int) -> SupervisorDecision:
        """Supervisor LLM 응답 파싱"""
        try:
            # 간단한 파싱 로직 (실제로는 더 정교한 파싱 필요)
            lines = response.strip().split('\n')

            consensus_hypotheses = []
            prioritized_tests = []
            rationale = response
            termination_reason = None

            # "합의" 또는 "종료" 키워드 검사
            if any(keyword in response.lower() for keyword in ['합의', '일치', '동의']):
                termination_reason = "Doctor 패널 합의 도달"

            # 기본값 설정
            if not consensus_hypotheses:
                consensus_hypotheses = [f"라운드 {round_number} 검토 결과"]

            if not prioritized_tests:
                prioritized_tests = ["추가 전문의 상담", "종합적 재검토"]

            return SupervisorDecision(
                consensus_hypotheses=consensus_hypotheses,
                prioritized_tests=prioritized_tests,
                rationale=rationale,
                termination_reason=termination_reason
            )

        except Exception as e:
            logger.error(f"❌ 응답 파싱 실패: {str(e)}")
            return SupervisorDecision(
                consensus_hypotheses=[f"라운드 {round_number} 분석 완료"],
                prioritized_tests=["전문의 최종 상담"],
                rationale="응답 파싱 중 오류 발생",
                termination_reason=None
            )

    def _get_previous_round_opinions(self,
                                   session_state: SessionState,
                                   current_round: int) -> Dict[str, DoctorOpinion]:
        """이전 라운드의 Doctor 의견들 반환"""
        if current_round <= 1 or not session_state.rounds:
            return {}

        try:
            previous_round = session_state.rounds[-2]  # 이전 라운드
            return previous_round.doctor_opinions
        except (IndexError, AttributeError):
            return {}

    def _get_supervisor_feedback(self,
                               session_state: SessionState,
                               current_round: int) -> str:
        """이전 라운드의 Supervisor 피드백 반환"""
        if current_round <= 1 or not session_state.rounds:
            return ""

        try:
            previous_round = session_state.rounds[-2]
            decision = previous_round.supervisor_decision
            return decision.get('rationale', '') if decision else ""
        except (IndexError, AttributeError):
            return ""

    def _get_previous_rounds_context(self, session_state: SessionState, current_round: int) -> str:
        """이전 라운드들의 컨텍스트 요약"""
        if current_round <= 1 or not session_state or not session_state.rounds:
            return ""

        context = f"\n이전 라운드들 요약 ({len(session_state.rounds)-1}라운드까지):\n"

        for round_record in session_state.rounds[:-1]:  # 현재 라운드 제외
            context += f"\n라운드 {round_record.round_index}:\n"

            # Doctor 의견들
            for doctor_id, opinion in round_record.doctor_opinions.items():
                context += f"  {doctor_id}: {opinion.get('hypotheses', [])} | {opinion.get('diagnostic_tests', [])}\n"

            # Supervisor 결정
            if round_record.supervisor_decision:
                decision = round_record.supervisor_decision
                context += f"  Supervisor: {decision.get('consensus_hypotheses', [])} | 합의여부: {bool(decision.get('termination_reason'))}\n"

        return context

    def _format_deliberation_result(self, session_id: str) -> Dict[str, Any]:
        """심의 결과 포맷팅"""
        session_state = self.conversation_manager.get_session(session_id)

        if not session_state or not session_state.rounds:
            return {
                "success": False,
                "error": "심의 결과를 찾을 수 없습니다",
                "session_id": session_id
            }

        # 최종 라운드의 결정 가져오기
        final_round = session_state.rounds[-1]
        final_decision = final_round.supervisor_decision

        return {
            "success": True,
            "session_id": session_id,
            "total_rounds": len(session_state.rounds),
            "termination_reason": session_state.termination_reason,
            "final_decision": final_decision,
            "consensus_reached": session_state.termination_reason == "합의 도달",
            "all_rounds": [
                {
                    "round_number": r.round_index,
                    "doctor_opinions": r.doctor_opinions,
                    "supervisor_decision": r.supervisor_decision
                }
                for r in session_state.rounds
            ]
        }

    def get_final_consensus(self, session_id: str) -> Optional[SupervisorDecision]:
        """최종 합의 결과 반환"""
        session_state = self.conversation_manager.get_session(session_id)

        if not session_state or not session_state.rounds:
            return None

        final_round = session_state.rounds[-1]
        return final_round.supervisor_decision
