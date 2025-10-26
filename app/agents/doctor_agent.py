#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Doctor Agent - Multi-agent system의 의사 에이전트

AGENTS.md 명세에 따른 역할:
- 진단 추론과 권장사항 제공
- 동료 의견 평가 및 피드백에 대한 논평
- 반복적으로 출력 개선
"""

import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.conversation_manager import CaseContext, DoctorOpinion
from app.agents.prompts.doctor_prompts import (
    DOCTOR_ANALYSIS_PROMPT,
    DOCTOR_CRITIQUE_PROMPT,
    DOCTOR_REASONING_PROMPT
)

logger = logging.getLogger(__name__)


class DoctorAgent:
    """
    Multi-agent system의 Doctor Agent

    AGENTS.md 명세에 따른 역할:
    - 진단 추론과 권장사항 제공
    - 동료 의견 평가 및 피드백에 대한 논평
    - 반복적으로 출력 개선
    """

    def __init__(self,
                 doctor_id: str,
                 openai_api_key: Optional[str] = None):
        logger.info(f"👨‍⚕️ DoctorAgent {doctor_id} 초기화 시작")

        self.doctor_id = doctor_id
        self.specialty = "일반의"  # 모든 Doctor는 일반의
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다")

        # LLM 초기화
        logger.info(f"🔑 {doctor_id} OpenAI LLM 초기화 중")
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7  # 창의적 추론을 위해 적당한 온도
        )

        # 의사별 히스토리 (라운드별 의견 기록)
        self.opinion_history: List[Dict[str, Any]] = []

        logger.info(f"✅ DoctorAgent {doctor_id} ({self.specialty}) 초기화 완료")

    def provide_opinion(self,
                       case_context: CaseContext,
                       round_number: int,
                       peer_opinions: Optional[Dict[str, DoctorOpinion]] = None,
                       supervisor_feedback: Optional[str] = None) -> DoctorOpinion:
        """
        사례에 대한 의학적 의견 제공

        Args:
            case_context: 환자 사례 컨텍스트
            round_number: 현재 라운드 번호
            peer_opinions: 이전 라운드의 동료 의견들
            supervisor_feedback: Supervisor의 피드백

        Returns:
            의사의 의견 (DoctorOpinion)
        """
        logger.info(f"🩺 {self.doctor_id} 라운드 {round_number} 의견 제공 시작")

        try:
            # 첫 번째 라운드인지 확인
            is_first_round = round_number == 1 or not peer_opinions

            if is_first_round:
                # 첫 번째 라운드: 초기 분석
                opinion = self._provide_initial_opinion(case_context, round_number)
            else:
                # 후속 라운드: 동료 의견과 피드백을 고려한 업데이트
                opinion = self._provide_updated_opinion(
                    case_context, round_number, peer_opinions, supervisor_feedback
                )

            # 히스토리에 기록
            self.opinion_history.append({
                "round": round_number,
                "opinion": opinion,
                "timestamp": self._get_timestamp()
            })

            logger.info(f"✅ {self.doctor_id} 라운드 {round_number} 의견 제공 완료")
            return opinion

        except Exception as e:
            logger.error(f"❌ {self.doctor_id} 의견 제공 실패: {str(e)}")

            # 기본 의견 반환
            fallback_opinion = DoctorOpinion(
                hypotheses=["추가 검토 필요"],
                diagnostic_tests=["전문의 상담"],
                reasoning=f"의견 생성 중 오류 발생: {str(e)}",
                critique_of_peers=""
            )

            self.opinion_history.append({
                "round": round_number,
                "opinion": fallback_opinion,
                "error": str(e),
                "timestamp": self._get_timestamp()
            })

            return fallback_opinion

    def _provide_initial_opinion(self,
                                case_context: CaseContext,
                                round_number: int) -> DoctorOpinion:
        """첫 번째 라운드 초기 의견 제공"""
        logger.info(f"🔍 {self.doctor_id} 초기 의견 분석 중")

        # 사례 분석을 위한 프롬프트 구성
        analysis_prompt = self._build_initial_analysis_prompt(case_context)

        # LLM을 통한 의견 생성
        response = self.llm.invoke([
            SystemMessage(content=DOCTOR_ANALYSIS_PROMPT.format(
                doctor_id=self.doctor_id,
                specialty=self.specialty
            )),
            HumanMessage(content=analysis_prompt)
        ])

        # Log doctor output
        logger.info(f"🩺 [{self.doctor_id}] 라운드 {round_number} 초기 분석 결과:")
        logger.info(f"📝 Raw LLM Response: {response.content}")

        # 응답 파싱
        parsed_opinion = self._parse_doctor_response(response.content, round_number)

        # Log parsed opinion
        logger.info(f"📊 [{self.doctor_id}] 파싱된 의견:")
        logger.info(f"   - 가설: {parsed_opinion.get('hypotheses', [])}")
        logger.info(f"   - 진단검사: {parsed_opinion.get('diagnostic_tests', [])}")
        logger.info(f"   - 추론: {parsed_opinion.get('reasoning', '')[:200]}...")

        return parsed_opinion

    def _provide_updated_opinion(self,
                                case_context: CaseContext,
                                round_number: int,
                                peer_opinions: Dict[str, DoctorOpinion],
                                supervisor_feedback: Optional[str]) -> DoctorOpinion:
        """후속 라운드 업데이트된 의견 제공"""
        logger.info(f"🔄 {self.doctor_id} 의견 업데이트 중 (라운드 {round_number})")

        # 이전 의견 가져오기
        previous_opinion = self._get_previous_opinion()

        # 업데이트된 분석을 위한 프롬프트 구성
        update_prompt = self._build_update_analysis_prompt(
            case_context, previous_opinion, peer_opinions, supervisor_feedback, round_number
        )

        # LLM을 통한 업데이트된 의견 생성
        response = self.llm.invoke([
            SystemMessage(content=DOCTOR_CRITIQUE_PROMPT.format(
                doctor_id=self.doctor_id,
                specialty=self.specialty,
                round_number=round_number
            )),
            HumanMessage(content=update_prompt)
        ])

        # Log doctor output for update
        logger.info(f"🔄 [{self.doctor_id}] 라운드 {round_number} 업데이트 분석 결과:")
        logger.info(f"📝 Raw LLM Response: {response.content}")

        # 응답 파싱
        parsed_opinion = self._parse_doctor_response(response.content, round_number, is_update=True)

        # Log parsed updated opinion
        logger.info(f"📊 [{self.doctor_id}] 업데이트된 의견:")
        logger.info(f"   - 가설: {parsed_opinion.get('hypotheses', [])}")
        logger.info(f"   - 진단검사: {parsed_opinion.get('diagnostic_tests', [])}")
        logger.info(f"   - 동료 비판: {parsed_opinion.get('critique_of_peers', '')[:100]}...")

        return parsed_opinion

    def _build_initial_analysis_prompt(self, case_context: CaseContext) -> str:
        """초기 분석을 위한 프롬프트 구성"""

        prompt = f"""
                Patient Case Analysis

        **Patient Information:**
        - Demographics: {case_context.get('demographics', {})}
        - Current Symptoms: {case_context.get('symptoms', {})}
        - Past Medical History: {case_context.get('history', {})}
        - Medications: {case_context.get('meds', {})}
        - Vital Signs: {case_context.get('vitals', {})}

        **Imaging Findings (MedBLIP Analysis):**
        {case_context.get('medblip_findings', {})}

        **Additional Information:**
        {case_context.get('free_text', '')}

        Based on the above information, please provide:
        1. Possible diagnostic hypotheses (in order of priority)
        2. Recommended diagnostic tests (in order of priority)
        3. Clinical reasoning process
        4. Key considerations and differential diagnoses

        **Important:**
        - Provide hypotheses, not definitive diagnoses
        - Prioritize patient safety
        - Emphasize the need for additional testing
        """

        return prompt

    def _build_update_analysis_prompt(self,
                                     case_context: CaseContext,
                                     previous_opinion: Optional[DoctorOpinion],
                                     peer_opinions: Dict[str, DoctorOpinion],
                                     supervisor_feedback: Optional[str],
                                     round_number: int) -> str:
        """업데이트 분석을 위한 프롬프트 구성"""

        # 기본 사례 정보
        base_info = f"""
        Patient Case (round {round_number})
        **Patient Information:**
        - Demographics: {case_context.get('demographics', {})}
        - Current Symptoms: {case_context.get('symptoms', {})}
        - Past Medical History: {case_context.get('history', {})}
        - Medications: {case_context.get('meds', {})}
        - MedBLIP Findings : {case_context.get('medblip_findings', {})}


        """

        # 이전 의견
        previous_section = ""
        if previous_opinion:
            previous_section = f"""
        **Previous Round's My Opinion:**
        - hypotheses: {previous_opinion.get('hypotheses', [])}
        - diagnostic_tests: {previous_opinion.get('diagnostic_tests', [])}
        - reasoning: {previous_opinion.get('reasoning', '')}
        """

        # 동료 의견들
        peer_section = "\n**Other Doctor's Opinion:**\n"
        logger.info(f"🔍 [{self.doctor_id}] 동료 의견 필터링 중...")
        logger.info(f"🔍 [{self.doctor_id}] self.doctor_id: '{self.doctor_id}'")
        logger.info(f"🔍 [{self.doctor_id}] peer_opinions.keys(): {list(peer_opinions.keys())}")

        for doctor_id, opinion in peer_opinions.items():
            logger.info(f"🔍 [{self.doctor_id}] 비교 중: doctor_id='{doctor_id}' vs self.doctor_id='{self.doctor_id}' -> 같음={doctor_id == self.doctor_id}")
            if doctor_id != self.doctor_id:  # 자신의 의견 제외
                peer_section += f"""
        {doctor_id}:
        - hypotheses: {opinion.get('hypotheses', [])}
        - diagnostic_tests: {opinion.get('diagnostic_tests', [])}
        - reasoning: {opinion.get('reasoning', '')}
        """
            else:
                logger.info(f"🔍 [{self.doctor_id}] ✅ 자신의 의견 제외됨: {doctor_id}")

        # Supervisor 피드백
        feedback_section = ""
        if supervisor_feedback:
            feedback_section = f"""
        **Supervisor feedback:**
        {supervisor_feedback}
        """

        prompt = f"""
        {base_info}
        {previous_section}
        {peer_section}
        {feedback_section}

        
        Based on the above information, please **perform the following steps in order**:

        1. **Self-Evaluation of Previous Opinion** (Required):
           - Analyze the strengths and weaknesses of your previous diagnostic hypotheses and test recommendations.
           - Identify any missing points or areas for improvement compared with your peers’ opinions.
           - Acknowledge the limitations and uncertainties of your own reasoning.

        2. **Peer Opinion Analysis:** Evaluate and critique the opinions of other physicians (excluding your own).

        3. **Opinion Update:** Provide an updated opinion that integrates your self-evaluation and feedback from peers.

        4. **Evidence Reinforcement:** Strengthen the clinical reasoning and evidence supporting your updated opinion.

        5. **Consensus Assessment:** Assess the likelihood of reaching consensus with your peers.

        **Update Priorities:**
        - **Priority 1:** Address weaknesses identified in your own previous opinion (from self-evaluation)
        - **Priority 2:** Incorporate valid points from peer opinions
        - **Priority 3:** Reflect supervisor feedback
        - **Priority 4:** Include additional relevant considerations
        - Always prioritize **patient safety** in every judgment
        
        """

        return prompt

    def _parse_doctor_response(self,
                              response: str,
                              round_number: int,
                              is_update: bool = False) -> DoctorOpinion:
        """Doctor LLM 응답 파싱"""
        try:
            # 간단한 파싱 로직 (실제로는 더 정교한 파싱 필요)
            lines = [line.strip() for line in response.split('\n') if line.strip()]

            hypotheses = []
            diagnostic_tests = []
            reasoning = response  # 전체 응답을 reasoning으로 사용
            critique_of_peers = ""

            # 키워드 기반 정보 추출
            current_section = None
            for line in lines:
                line_lower = line.lower()

                if any(keyword in line_lower for keyword in ['가설', '진단', '후보', '가능성']):
                    current_section = 'hypotheses'
                elif any(keyword in line_lower for keyword in ['검사', '진단검사', '추가검사']):
                    current_section = 'tests'
                elif any(keyword in line_lower for keyword in ['동료', '의견', '비판', '평가']):
                    current_section = 'critique'

                # 목록 항목 추출 (-, *, 1., 2. 등)
                if line.startswith(('-', '*', '•')) or (len(line) > 2 and line[1] == '.'):
                    item = line.lstrip('-*•0123456789. ').strip()
                    if current_section == 'hypotheses' and item:
                        hypotheses.append(item)
                    elif current_section == 'tests' and item:
                        diagnostic_tests.append(item)

            # 동료 비판 추출 (업데이트 라운드인 경우)
            if is_update:
                critique_keywords = ['동료', '의견', '평가', '비판', '분석']
                critique_lines = []
                for line in lines:
                    if any(keyword in line for keyword in critique_keywords):
                        critique_lines.append(line)
                critique_of_peers = ' '.join(critique_lines)

            # 기본값 설정
            if not hypotheses:
                hypotheses = [f"{self.specialty} 관점에서의 종합적 검토 필요"]

            if not diagnostic_tests:
                diagnostic_tests = ["추가 전문의 상담", "종합적 진단 검사"]

            return DoctorOpinion(
                hypotheses=hypotheses[:5],  # 최대 5개로 제한
                diagnostic_tests=diagnostic_tests[:5],  # 최대 5개로 제한
                reasoning=reasoning,
                critique_of_peers=critique_of_peers
            )

        except Exception as e:
            logger.error(f"❌ 응답 파싱 실패: {str(e)}")
            return DoctorOpinion(
                hypotheses=[f"{self.specialty} 관점 분석 중 오류"],
                diagnostic_tests=["전문의 재상담"],
                reasoning=f"응답 파싱 중 오류 발생: {str(e)}",
                critique_of_peers=""
            )

    def _get_previous_opinion(self) -> Optional[DoctorOpinion]:
        """이전 라운드의 의견 반환"""
        if not self.opinion_history:
            return None

        return self.opinion_history[-1].get('opinion')

    def _get_timestamp(self) -> str:
        """현재 시간 문자열 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_opinion_history(self) -> List[Dict[str, Any]]:
        """의견 히스토리 반환"""
        return self.opinion_history.copy()

    def reset(self):
        """에이전트 상태 초기화"""
        logger.info(f"🔄 {self.doctor_id} 상태 초기화")
        self.opinion_history.clear()


def create_doctor_panel(openai_api_key: Optional[str] = None) -> List[DoctorAgent]:
    """
    AGENTS.md 명세에 따라 정확히 3명의 Doctor Agent 패널 생성
    모든 Doctor는 일반의 (non-specialized)

    Returns:
        3명의 DoctorAgent 리스트
    """
    logger.info("👥 Doctor Agent 패널 생성 중...")

    doctors = [
        DoctorAgent("doctor_1", openai_api_key),
        DoctorAgent("doctor_2", openai_api_key),
        DoctorAgent("doctor_3", openai_api_key)
    ]

    logger.info("✅ 3명의 일반의 Doctor Agent 패널 생성 완료")
    return doctors