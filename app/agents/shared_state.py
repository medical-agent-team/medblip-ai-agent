#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent Medical Consultation Shared State

LangGraph를 통해 모든 에이전트가 공유하는 상태 객체를 정의합니다.
Admin, Doctor, Supervisor Agent가 모두 이 상태를 읽고 쓸 수 있습니다.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, TypedDict
from PIL import Image
from dataclasses import dataclass, field


# 기본 정보 구조
class Demographics(TypedDict, total=False):
    """환자 기본 정보"""
    age: Optional[str]
    gender: Optional[str]
    occupation: Optional[str]
    residence: Optional[str]
    raw_input: str
    processed: bool


# 과거 병력 구조
class MedicalHistory(TypedDict, total=False):
    """과거 병력 및 가족력"""
    past_diseases: List[str]
    surgeries: List[str]
    allergies: List[str]
    family_history: List[str]
    raw_input: str
    processed: bool


# 현재 증상 구조
class CurrentSymptoms(TypedDict, total=False):
    """현재 증상 정보"""
    main_symptoms: List[str]
    onset_time: Optional[str]
    severity: Optional[str]
    pattern: Optional[str]  # 지속적/간헐적/악화/호전
    aggravating_factors: List[str]
    associated_symptoms: List[str]
    raw_input: str
    processed: bool


# 복용 약물 구조
class Medications(TypedDict, total=False):
    """복용 중인 약물 정보"""
    prescription_drugs: List[str]
    over_the_counter: List[str]
    supplements: List[str]
    traditional_medicine: List[str]
    raw_input: str
    processed: bool


# MedBLIP 분석 결과 구조
class MedBLIPAnalysis(TypedDict, total=False):
    """MedBLIP 의료 이미지 분석 결과"""
    caption: str
    confidence: Optional[float]
    entities: List[Dict[str, Any]]
    findings: List[str]
    impression: Optional[str]
    processed: bool


# Doctor 의견 구조
class DoctorOpinion(TypedDict, total=False):
    """개별 Doctor Agent의 의견"""
    doctor_id: str
    round_number: int
    hypotheses: List[str]
    diagnostic_tests: List[str]
    reasoning: str
    critique_of_peers: str
    confidence_level: Optional[str]


# Supervisor 결정 구조
class SupervisorDecision(TypedDict, total=False):
    """Supervisor Agent의 합의 결정"""
    round_number: int
    consensus_reached: bool
    consensus_hypotheses: List[str]
    prioritized_tests: List[str]
    rationale: str
    termination_reason: Optional[str]
    next_round_needed: bool


# 최종 환자 요약 구조
class PatientSummary(TypedDict, total=False):
    """환자 친화적 최종 요약"""
    summary_text: str
    key_findings: List[str]
    recommended_actions: List[str]
    safety_warnings: List[str]
    disclaimers: List[str]


# 공유 상태 메인 스키마
class SharedMedicalState(TypedDict, total=False):
    """모든 에이전트가 공유하는 메인 상태 객체"""

    # 1. 기본정보
    demographics: Demographics

    # 2. 과거병력
    history: MedicalHistory

    # 3. 현재증상
    symptoms: CurrentSymptoms

    # 4. 복용약물
    medications: Medications

    # 5. MedBLIP 캡션
    medblip_analysis: MedBLIPAnalysis

    # 워크플로우 제어
    current_stage: str
    stage_completed: Dict[str, bool]

    # 이미지 관련
    uploaded_image: Optional[Image.Image]
    image_processed: bool

    # 멀티 에이전트 라운드 관리
    current_round: int
    max_rounds: int
    doctor_opinions: Dict[str, DoctorOpinion]  # doctor_id -> opinion
    supervisor_decisions: List[SupervisorDecision]

    # 최종 결과
    consultation_complete: bool
    patient_summary: Optional[PatientSummary]

    # 메타데이터
    session_id: str
    created_at: Optional[str]
    updated_at: Optional[str]

    # 오류 및 메시지
    messages: List[Dict[str, Any]]
    error_messages: List[str]


@dataclass
class StateManager:
    """공유 상태를 관리하는 헬퍼 클래스"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    @staticmethod
    def create_initial_state(session_id: str) -> SharedMedicalState:
        """초기 상태 객체 생성"""
        return SharedMedicalState(
            # 기본 데이터 구조 초기화
            demographics=Demographics(processed=False, raw_input=""),
            history=MedicalHistory(processed=False, raw_input=""),
            symptoms=CurrentSymptoms(processed=False, raw_input=""),
            medications=Medications(processed=False, raw_input=""),
            medblip_analysis=MedBLIPAnalysis(processed=False, caption=""),

            # 워크플로우 상태
            current_stage="greeting",
            stage_completed={
                "demographics": False,
                "history": False,
                "symptoms": False,
                "medications": False,
                "image_analysis": False
            },

            # 이미지 관련
            uploaded_image=None,
            image_processed=False,

            # 멀티 에이전트 관리
            current_round=0,
            max_rounds=13,
            doctor_opinions={},
            supervisor_decisions=[],

            # 완료 상태
            consultation_complete=False,
            patient_summary=None,

            # 메타데이터
            session_id=session_id,
            created_at=None,
            updated_at=None,

            # 메시지 및 오류
            messages=[],
            error_messages=[]
        )

    @staticmethod
    def update_demographics(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
        """기본정보 업데이트"""
        demographics = state["demographics"]
        demographics["raw_input"] = raw_input
        demographics["processed"] = True

        # 키워드에서 정보 추출
        for key, value in kwargs.items():
            if key in ["age", "gender", "occupation", "residence"]:
                demographics[key] = value

        state["stage_completed"]["demographics"] = True
        return state

    @staticmethod
    def update_history(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
        """병력 정보 업데이트"""
        history = state["history"]
        history["raw_input"] = raw_input
        history["processed"] = True

        # 리스트 필드 업데이트
        for key in ["past_diseases", "surgeries", "allergies", "family_history"]:
            if key in kwargs and isinstance(kwargs[key], list):
                history[key] = kwargs[key]

        state["stage_completed"]["history"] = True
        return state

    @staticmethod
    def update_symptoms(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
        """증상 정보 업데이트"""
        symptoms = state["symptoms"]
        symptoms["raw_input"] = raw_input
        symptoms["processed"] = True

        # 개별 필드 업데이트
        for key in ["onset_time", "severity", "pattern"]:
            if key in kwargs:
                symptoms[key] = kwargs[key]

        # 리스트 필드 업데이트
        for key in ["main_symptoms", "aggravating_factors", "associated_symptoms"]:
            if key in kwargs and isinstance(kwargs[key], list):
                symptoms[key] = kwargs[key]

        state["stage_completed"]["symptoms"] = True
        return state

    @staticmethod
    def update_medications(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
        """약물 정보 업데이트"""
        medications = state["medications"]
        medications["raw_input"] = raw_input
        medications["processed"] = True

        # 리스트 필드 업데이트
        for key in ["prescription_drugs", "over_the_counter", "supplements", "traditional_medicine"]:
            if key in kwargs and isinstance(kwargs[key], list):
                medications[key] = kwargs[key]

        state["stage_completed"]["medications"] = True
        return state

    @staticmethod
    def update_medblip_analysis(state: SharedMedicalState, caption: str, **kwargs) -> SharedMedicalState:
        """MedBLIP 분석 결과 업데이트"""
        analysis = state["medblip_analysis"]
        analysis["caption"] = caption
        analysis["processed"] = True

        # 추가 필드 업데이트
        for key in ["confidence", "entities", "findings", "impression"]:
            if key in kwargs:
                analysis[key] = kwargs[key]

        state["stage_completed"]["image_analysis"] = True
        state["image_processed"] = True
        return state

    @staticmethod
    def add_doctor_opinion(state: SharedMedicalState, doctor_id: str, opinion: DoctorOpinion) -> SharedMedicalState:
        """Doctor 의견 추가"""
        opinion["doctor_id"] = doctor_id
        opinion["round_number"] = state["current_round"]
        state["doctor_opinions"][doctor_id] = opinion
        return state

    @staticmethod
    def add_supervisor_decision(state: SharedMedicalState, decision: SupervisorDecision) -> SharedMedicalState:
        """Supervisor 결정 추가"""
        decision["round_number"] = state["current_round"]
        state["supervisor_decisions"].append(decision)
        return state

    @staticmethod
    def is_intake_complete(state: SharedMedicalState) -> bool:
        """정보 수집이 완료되었는지 확인"""
        required_stages = ["demographics", "history", "symptoms", "medications"]
        return all(state["stage_completed"].get(stage, False) for stage in required_stages)

    @staticmethod
    def can_start_consultation(state: SharedMedicalState) -> bool:
        """상담 시작 가능한지 확인"""
        return StateManager.is_intake_complete(state)


# 상태 업데이트 헬퍼 함수들
def update_state_demographics(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
    """상태의 기본정보 업데이트 (함수형 인터페이스)"""
    return StateManager.update_demographics(state, raw_input, **kwargs)

def update_state_history(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
    """상태의 병력정보 업데이트 (함수형 인터페이스)"""
    return StateManager.update_history(state, raw_input, **kwargs)

def update_state_symptoms(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
    """상태의 증상정보 업데이트 (함수형 인터페이스)"""
    return StateManager.update_symptoms(state, raw_input, **kwargs)

def update_state_medications(state: SharedMedicalState, raw_input: str, **kwargs) -> SharedMedicalState:
    """상태의 약물정보 업데이트 (함수형 인터페이스)"""
    return StateManager.update_medications(state, raw_input, **kwargs)

def update_state_medblip(state: SharedMedicalState, caption: str, **kwargs) -> SharedMedicalState:
    """상태의 MedBLIP 분석결과 업데이트 (함수형 인터페이스)"""
    return StateManager.update_medblip_analysis(state, caption, **kwargs)


__all__ = [
    'SharedMedicalState',
    'Demographics',
    'MedicalHistory',
    'CurrentSymptoms',
    'Medications',
    'MedBLIPAnalysis',
    'DoctorOpinion',
    'SupervisorDecision',
    'PatientSummary',
    'StateManager',
    'update_state_demographics',
    'update_state_history',
    'update_state_symptoms',
    'update_state_medications',
    'update_state_medblip'
]