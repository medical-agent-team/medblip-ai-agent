#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-agent system for medical consultation

AGENTS.md 명세에 따른 구성:
- AdminAgent: 사용자 입력 수집, MedBLIP 분석, 컨텍스트 패키징
- SupervisorAgent: Doctor Agent 간 조율, 합의 도출
- DoctorAgent: 전문의 관점, 다양한 의견 제공
- ConversationManager: 대화 흐름, 상태 관리
"""

from app.agents.admin_agent import AdminAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.doctor_agent import DoctorAgent, create_doctor_panel
from app.agents.conversation_manager import (
    ConversationManager,
    CaseContext,
    DoctorOpinion,
    SupervisorDecision,
    PatientSummary,
    SessionState,
    RoundRecord
)

# Legacy agents for backward compatibility
try:
    from app.agents.agent import Agent
except ImportError:
    Agent = None

try:
    from app.agents.radiology_agent import RadiologyAgent
except ImportError:
    RadiologyAgent = None

__all__ = [
    # Core multi-agent system
    'AdminAgent',
    'SupervisorAgent',
    'DoctorAgent',
    'create_doctor_panel',

    # Conversation management
    'ConversationManager',
    'CaseContext',
    'DoctorOpinion',
    'SupervisorDecision',
    'PatientSummary',
    'SessionState',
    'RoundRecord',

    # Legacy agents (if available)
    'Agent',
    'RadiologyAgent'
]