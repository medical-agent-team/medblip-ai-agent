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

        # ConversationManager 초기화 ({self.max_rounds}라운드로 변경)
        self.conversation_manager = ConversationManager(max_rounds=self.max_rounds)

        logger.info("✅ SupervisorAgent 초기화 완료")

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
            # 정확히 {self.max_rounds}라운드까지 반복 (조기 종료 없음)
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

                # 조기 종료 로직 주석 처리 - 무조건 {self.max_rounds}라운드 실행
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

            # Log before LLM invocation
            logger.info("=" * 80)
            logger.info(f"🎯 [Supervisor] 라운드 {round_number} - LLM 호출 준비")
            logger.info(f"🔍 LLM 인스턴스: {self.llm}")
            logger.info(f"🔍 LLM 타입: {type(self.llm)}")
            if hasattr(self.llm, 'model_name'):
                logger.info(f"🔍 모델명: {self.llm.model_name}")
            if hasattr(self.llm, 'openai_api_base'):
                logger.info(f"🔍 Base URL: {self.llm.openai_api_base}")
            if hasattr(self.llm, 'callbacks'):
                logger.info(f"🔍 Callbacks: {[type(cb).__name__ for cb in self.llm.callbacks]}")
            logger.info(f"📏 System prompt 길이: {len(SUPERVISOR_CONSENSUS_PROMPT)} 문자")
            logger.info(f"📏 Human prompt 길이: {len(consensus_prompt)} 문자")
            logger.info(f"📏 총 입력 길이: ~{len(SUPERVISOR_CONSENSUS_PROMPT) + len(consensus_prompt)} 문자")

            # LLM을 통한 합의 분석
            logger.info("🚀 [Supervisor] LLM invoke 시작...")
            base_messages = [
                SystemMessage(content=SUPERVISOR_CONSENSUS_PROMPT),
                HumanMessage(content=consensus_prompt)
            ]
            try:
                response = self.llm.invoke(base_messages)
                logger.info("✅ [Supervisor] LLM invoke 완료")
            except Exception as e:
                logger.error(f"❌ [Supervisor] LLM invoke 실패: {type(e).__name__}: {str(e)}")
                logger.error(f"   Exception details: {repr(e)}")
                raise

            response = self._ensure_valid_response(
                base_messages=base_messages,
                response=response,
                context=f"supervisor-round{round_number}"
            )

            # Log supervisor output with diagnostics
            logger.info(f"🔍 [Supervisor] 응답 검증:")
            logger.info(f"   Response 타입: {type(response)}")
            logger.info(f"   Has 'content': {hasattr(response, 'content')}")
            if hasattr(response, 'content'):
                logger.info(f"   Content 타입: {type(response.content)}")
                logger.info(f"   Content is None: {response.content is None}")
                if response.content is not None:
                    logger.info(f"   Content 길이: {len(response.content)} 문자")
                    logger.info(f"   Content is empty string: {response.content == ''}")
                else:
                    logger.error(f"❌ [Supervisor] response.content is None!")
            else:
                logger.error(f"❌ [Supervisor] response has no 'content' attribute!")

            logger.info(f"📏 응답 길이: {len(response.content) if response and hasattr(response, 'content') and response.content else 0} 문자")

            # Check for expected sections
            expected_sections = ["**Integrated Hypothesis**", "**Priority Tests**", "**Consensus Status**"]
            found_sections = [s for s in expected_sections if s in response.content]
            logger.info(f"✓ 발견된 섹션: {found_sections} ({len(found_sections)}/{len(expected_sections)})")

            # Log first 500 chars for debugging
            logger.info(f"📝 응답 미리보기 (처음 500자):")
            logger.info(f"{response.content[:500]}...")

            # Full response for detailed debugging
            logger.debug(f"📝 전체 Raw LLM Response: {response.content}")

            # 응답 파싱
            decision = self._parse_supervisor_response(response.content, round_number)

            # Log parsed decision with more detail
            logger.info(f"📊 [Supervisor] 파싱된 결정:")
            logger.info(f"   - 합의 가설 ({len(decision.get('consensus_hypotheses', []))}개): {decision.get('consensus_hypotheses', [])}")
            logger.info(f"   - 우선 검사 ({len(decision.get('prioritized_tests', []))}개): {decision.get('prioritized_tests', [])}")
            logger.info(f"   - 종료 여부: {decision.get('termination_reason', 'None')}")
            rationale = decision.get('rationale', '')
            logger.info(f"   - 근거 길이: {len(rationale)} 문자")
            logger.info(f"   - 근거 미리보기: {rationale[:200]}...")

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

    def _ensure_valid_response(self,
                               base_messages: List[SystemMessage | HumanMessage],
                               response: Any,
                               context: str):
        """Ensure supervisor receives non-empty content; retry once if truncated."""
        if getattr(response, "content", None):
            return response

        self._log_empty_response_debug(response, context=context)

        metadata = getattr(response, "response_metadata", {}) or {}
        finish_reason = metadata.get("finish_reason")
        if finish_reason == "length":
            logger.warning(f"🔁 [{context}] Retrying supervisor LLM call due to truncation")
            continuation_messages = base_messages + [
                HumanMessage(content=(
                    "Continue from the prior response and deliver the full consensus analysis. "
                    "Stay within the specified length constraints."
                ))
            ]
            retry_response = self.llm.invoke(continuation_messages)
            if getattr(retry_response, "content", None):
                logger.info(f"✅ [{context}] Retry succeeded")
                return retry_response
            self._log_empty_response_debug(retry_response, context=f"{context}-retry")
            raise RuntimeError("Supervisor LLM returned empty content after retry.")

        raise RuntimeError("Supervisor LLM returned empty content without truncation metadata.")

    def _log_empty_response_debug(self, response: Any, context: str) -> None:
        """빈 Supervisor 응답 시 디버그 정보 출력"""
        try:
            message_dict = None
            try:
                from langchain_core.messages import message_to_dict  # type: ignore
            except Exception:  # pragma: no cover - best effort import
                message_to_dict = None  # type: ignore

            logger.warning(f"⚠️ [{context}] Empty LLM content detected")
            if hasattr(response, "response_metadata"):
                logger.warning(f"⚙️ [{context}] response_metadata: {getattr(response, 'response_metadata', {})}")
            if hasattr(response, "additional_kwargs"):
                logger.warning(f"⚙️ [{context}] additional_kwargs: {getattr(response, 'additional_kwargs', {})}")
            if hasattr(response, "usage_metadata"):
                logger.warning(f"⚙️ [{context}] usage_metadata: {getattr(response, 'usage_metadata', {})}")
            if message_to_dict:
                message_dict = message_to_dict(response)
            if message_dict:
                logger.warning(f"📦 [{context}] message_dict: {message_dict}")
        except Exception as debug_error:
            logger.error(f"🛑 Failed to log empty LLM response for {context}: {debug_error}")

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
        """Supervisor LLM 응답 파싱 - 구조화된 섹션 추출"""
        try:
            if not response or not response.strip():
                logger.warning(f"⚠️ 빈 LLM 응답 수신 (라운드 {round_number})")
                return self._create_fallback_decision(round_number, "빈 응답")

            lines = response.strip().split('\n')

            consensus_hypotheses = []
            prioritized_tests = []
            rationale = ""
            termination_reason = None

            # 섹션별 파싱
            current_section = None
            current_subsection = None

            for line in lines:
                line_stripped = line.strip()

                # 메인 섹션 감지
                if "**Integrated Hypothesis**" in line or "**통합 가설**" in line:
                    current_section = "hypothesis"
                    current_subsection = None
                elif "**Priority Tests**" in line or "**우선 검사**" in line:
                    current_section = "tests"
                    current_subsection = None
                elif "**Consensus Status**" in line or "**합의 상태**" in line:
                    current_section = "consensus"
                    current_subsection = None
                elif "**Consensus Analysis**" in line or "**합의 분석**" in line:
                    current_section = "analysis"
                    current_subsection = None
                elif "**Safety Considerations**" in line or "**안전 고려사항**" in line:
                    current_section = "safety"
                    current_subsection = None

                # 서브섹션 감지
                elif current_section == "hypothesis":
                    if "Main Candidates:" in line or "주요 후보:" in line or "- Main Candidates:" in line:
                        current_subsection = "main_candidates"
                    elif line_stripped.startswith('-') and current_subsection == "main_candidates":
                        # 가설 항목 추출
                        hypothesis = line_stripped.lstrip('- ').strip()
                        if hypothesis and len(hypothesis) > 3:
                            consensus_hypotheses.append(hypothesis)

                elif current_section == "tests":
                    if "Immediately Needed:" in line or "즉시 필요:" in line or "- Immediately Needed:" in line:
                        current_subsection = "immediately_needed"
                    elif line_stripped.startswith('-') and current_subsection == "immediately_needed":
                        # 검사 항목 추출
                        test = line_stripped.lstrip('- ').strip()
                        if test and len(test) > 3:
                            prioritized_tests.append(test)

                elif current_section == "consensus":
                    if "Consensus Rationale:" in line or "합의 근거:" in line or "- Consensus Rationale:" in line:
                        current_subsection = "rationale"
                        # 같은 줄에 내용이 있으면 추출
                        rationale_start = line.split(":", 1)
                        if len(rationale_start) > 1:
                            rationale_text = rationale_start[1].strip()
                            if rationale_text:
                                rationale = rationale_text
                    elif current_subsection == "rationale" and line_stripped:
                        # 다음 줄들도 rationale에 포함
                        if not line_stripped.startswith('-') and not line_stripped.startswith('**'):
                            rationale += " " + line_stripped

                    # 합의 도달 여부 확인
                    if "Clear consensus" in line or "Complete consensus" in line or \
                       "명확한 합의" in line or "완전한 합의" in line or \
                       "Consensus Reached: Yes" in line or "합의 도달: 예" in line:
                        termination_reason = "Doctor 패널 합의 도달"

            # 추출된 데이터 검증 및 기본값 설정
            if not consensus_hypotheses:
                logger.warning(f"⚠️ 가설 추출 실패 - 기본값 사용 (라운드 {round_number})")
                consensus_hypotheses = [f"라운드 {round_number} 검토 결과"]

            if not prioritized_tests:
                logger.warning(f"⚠️ 검사 추출 실패 - 기본값 사용 (라운드 {round_number})")
                prioritized_tests = ["추가 전문의 상담", "종합적 재검토"]

            if not rationale or len(rationale.strip()) < 10:
                logger.warning(f"⚠️ 근거 추출 실패 - 전체 응답 사용 (라운드 {round_number})")
                rationale = response[:500]  # 처음 500자만 저장

            # 추출 성공 로그
            logger.info(f"✅ 파싱 성공: {len(consensus_hypotheses)}개 가설, {len(prioritized_tests)}개 검사")

            return SupervisorDecision(
                consensus_hypotheses=consensus_hypotheses,
                prioritized_tests=prioritized_tests,
                rationale=rationale.strip(),
                termination_reason=termination_reason
            )

        except Exception as e:
            logger.error(f"❌ 응답 파싱 실패: {str(e)}")
            return self._create_fallback_decision(round_number, f"파싱 오류: {str(e)}")

    def _create_fallback_decision(self, round_number: int, reason: str) -> SupervisorDecision:
        """파싱 실패 시 대체 결정 생성"""
        return SupervisorDecision(
            consensus_hypotheses=[f"라운드 {round_number} 분석 완료"],
            prioritized_tests=["전문의 최종 상담"],
            rationale=f"응답 처리 중 문제 발생: {reason}",
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
