#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conversation manager for multi-agent orchestration.

Responsibilities
- In-memory session state (ephemeral): rounds, messages, decisions
- Data contracts: CaseContext, DoctorOpinion, SupervisorDecision, PatientSummary
- Validation and safety caps (max rounds, no treatment advice flags)
- Optional LangGraph network: builds a StateGraph to orchestrate Admin/Supervisor/Doctors

Notes
- This module is side-effect free at import time and can be used without
  LangGraph; graph building is conditional and guarded.
- All user-facing text remains Korean at the UI layer; this module is backend
  orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TypedDict


# ---------- Data Contracts (lightweight) ----------

class CaseContext(TypedDict, total=False):
    demographics: Dict[str, Any]
    symptoms: Any
    history: Any
    meds: Any
    vitals: Any
    medblip_findings: Dict[str, Any]
    free_text: str


class DoctorOpinion(TypedDict, total=False):
    hypotheses: List[str]
    diagnostic_tests: List[str]
    reasoning: str
    critique_of_peers: str


class SupervisorDecision(TypedDict, total=False):
    consensus_hypotheses: List[str]
    prioritized_tests: List[str]
    rationale: str
    termination_reason: Optional[str]


class PatientSummary(TypedDict, total=False):
    summary_text: str
    disclaimers: List[str]


# ---------- Round/Session State ----------

@dataclass
class RoundRecord:
    round_index: int
    doctor_opinions: Dict[str, DoctorOpinion] = field(default_factory=dict)
    supervisor_decision: Optional[SupervisorDecision] = None


@dataclass
class SessionState:
    session_id: str
    context: CaseContext
    current_round: int = 0
    max_rounds: int = 13
    rounds: List[RoundRecord] = field(default_factory=list)
    terminated: bool = False
    termination_reason: Optional[str] = None


# ---------- Node Protocols (LangGraph-compatible) ----------

class AdminNode(Protocol):
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...


class SupervisorNode(Protocol):
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...


class DoctorNode(Protocol):
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...


# ---------- Conversation Manager ----------

class ConversationManager:
    """Centralizes session lifecycle, validation, and round bookkeeping."""

    def __init__(self, *, max_rounds: int = 13) -> None:
        self.max_rounds = max_rounds
        self._sessions: Dict[str, SessionState] = {}

    # -- Session lifecycle --
    def start_session(self, session_id: str, context: CaseContext) -> SessionState:
        if session_id in self._sessions:
            return self._sessions[session_id]
        state = SessionState(session_id=session_id, context=context, max_rounds=self.max_rounds)
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str, reason: str | None = None) -> None:
        st = self._sessions.get(session_id)
        if st is None:
            return
        st.terminated = True
        st.termination_reason = reason

    # -- Round operations --
    def begin_round(self, session_id: str) -> int:
        st = self._require_session(session_id)
        if st.terminated:
            raise RuntimeError("세션이 종료되었습니다.")
        if st.current_round >= st.max_rounds:
            st.terminated = True
            st.termination_reason = "라운드 한도 도달"
            raise RuntimeError("최대 라운드 수에 도달했습니다.")
        st.current_round += 1
        st.rounds.append(RoundRecord(round_index=st.current_round))
        return st.current_round

    def add_doctor_opinion(self, session_id: str, doctor_id: str, opinion: DoctorOpinion) -> None:
        st = self._require_session(session_id)
        rr = self._require_current_round(st)
        self._validate_doctor_opinion(opinion)
        rr.doctor_opinions[doctor_id] = opinion

    def record_supervisor_decision(self, session_id: str, decision: SupervisorDecision) -> None:
        st = self._require_session(session_id)
        rr = self._require_current_round(st)
        self._validate_supervisor_decision(decision)
        rr.supervisor_decision = decision

    def reached_consensus(self, session_id: str) -> bool:
        st = self._require_session(session_id)
        rr = self._require_current_round(st)
        d = rr.supervisor_decision
        if not d:
            return False
        # Minimal heuristic placeholder: consensus when >=1 hypothesis and >=1 prioritized test
        return bool(d.get("consensus_hypotheses")) and bool(d.get("prioritized_tests"))

    # -- Lightweight validators / safety --
    def _validate_doctor_opinion(self, opinion: DoctorOpinion) -> None:
        # Basic shape checks; expand with stricter rules as needed
        if not isinstance(opinion.get("hypotheses", []), list):
            raise ValueError("hypotheses는 리스트여야 합니다.")
        if not isinstance(opinion.get("diagnostic_tests", []), list):
            raise ValueError("diagnostic_tests는 리스트여야 합니다.")

    def _validate_supervisor_decision(self, decision: SupervisorDecision) -> None:
        if not isinstance(decision.get("consensus_hypotheses", []), list):
            raise ValueError("consensus_hypotheses는 리스트여야 합니다.")
        if not isinstance(decision.get("prioritized_tests", []), list):
            raise ValueError("prioritized_tests는 리스트여야 합니다.")

    def _require_session(self, session_id: str) -> SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            raise KeyError(f"세션을 찾을 수 없습니다: {session_id}")
        return st

    def _require_current_round(self, st: SessionState) -> RoundRecord:
        if not st.rounds:
            raise RuntimeError("시작된 라운드가 없습니다. begin_round를 먼저 호출하세요.")
        return st.rounds[-1]

    # -- PHI-minimizing trace (placeholder) --
    def redact_for_log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Implement minimal redaction; callers can add more filters upstack
        redacted = dict(payload)
        redacted.pop("free_text", None)
        return redacted

    # ---------- LangGraph integration ----------
    def build_graph(
        self,
        *,
        admin: Optional[AdminNode] = None,
        supervisor: Optional[SupervisorNode] = None,
        doctors: Optional[List[DoctorNode]] = None,
    ):
        """Return a compiled LangGraph StateGraph when langgraph is installed.

        Nodes are simple callables(state) -> state. This method wires them as:
        START -> admin -> doctor[0..2] -> supervisor -> (END or loop)

        Loop termination is expected to be handled by supervisor via state flags,
        and by the caller managing per-round execution.
        """
        try:
            from langgraph.graph import StateGraph, START, END  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "langgraph가 설치되어 있지 않습니다. pyproject에 langgraph를 추가하세요."
            ) from e

        sg = StateGraph(dict)

        if admin:
            sg.add_node("admin", admin)
        if doctors:
            for i, d in enumerate(doctors):
                sg.add_node(f"doctor_{i}", d)
        if supervisor:
            sg.add_node("supervisor", supervisor)

        # Edges: START -> admin -> doctors -> supervisor
        if admin:
            sg.add_edge(START, "admin")
            prev = "admin"
        else:
            sg.add_edge(START, "supervisor" if supervisor else END)
            prev = None

        if doctors:
            for i in range(len(doctors)):
                cur = f"doctor_{i}"
                if prev:
                    sg.add_edge(prev, cur)
                prev = cur

        if supervisor:
            if prev:
                sg.add_edge(prev, "supervisor")
            # By default go to END; callers can re-enter per round externally
            sg.add_edge("supervisor", END)

        return sg.compile()


__all__ = [
    "ConversationManager",
    "CaseContext",
    "DoctorOpinion",
    "SupervisorDecision",
    "PatientSummary",
    "SessionState",
    "RoundRecord",
]

