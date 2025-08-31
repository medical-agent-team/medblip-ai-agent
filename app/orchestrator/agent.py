import os
from typing import Dict, Any, Optional, List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .prompts.prompt import ORCHESTRATOR_AGENT_PROMPT, RADIOLOGY_ANALYSIS_PROMPT
from .models import ConversationState


class OrchestratorAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        if llm is None:
            # Load OpenAI API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
        else:
            self.llm = llm
            
        self.orchestrator_prompt = ORCHESTRATOR_AGENT_PROMPT
        self.radiology_prompt = RADIOLOGY_ANALYSIS_PROMPT
        self.conversation_state = {
            "conversation_stage": "greeting",
            "collected_info": {},
            "has_image": False,
            "conversation_history": "",
            "decision": None,
            "reason": None,
            "context": None
        }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """AI 응답을 파싱하여 결정사항과 메시지를 추출"""
        lines = response.strip().split('\n')
        parsed = {
            "decision": None,
            "reason": None,
            "message": response,
            "context": None
        }
        
        for line in lines:
            if line.startswith("DECISION:"):
                parsed["decision"] = line.replace("DECISION:", "").strip()
            elif line.startswith("REASON:"):
                parsed["reason"] = line.replace("REASON:", "").strip()
            elif line.startswith("MESSAGE:"):
                parsed["message"] = line.replace("MESSAGE:", "").strip()
            elif line.startswith("CONTEXT:"):
                parsed["context"] = line.replace("CONTEXT:", "").strip()
        
        return parsed
    
    def _update_conversation_stage(self, decision: str) -> str:
        """Update conversation stage based on decision"""
        stage_mapping = {
            "COLLECT_BASIC_INFO": "basic_info",
            "REQUEST_IMAGE": "image_upload",
            "ANALYZE_IMAGE": "analysis",
            "PROVIDE_EXPLANATION": "explanation"
        }
        return stage_mapping.get(decision, "basic_info")
    
    def _extract_medical_info(self, user_input: str, current_info: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 입력에서 의료 정보 추출 및 업데이트"""
        # 실제 구현에서는 더 정교한 NLP 기법을 사용할 수 있음
        updated_info = current_info.copy()
        
        # 나이 정보 추출
        if any(keyword in user_input for keyword in ["살", "세", "년생", "나이"]):
            updated_info["age_mentioned"] = True
        
        # 증상 정보 추출
        if any(keyword in user_input for keyword in ["아프", "통증", "불편", "증상"]):
            if "symptoms" not in updated_info:
                updated_info["symptoms"] = []
            updated_info["symptoms"].append(user_input)
        
        # 과거 병력 추출
        if any(keyword in user_input for keyword in ["병원", "치료", "수술", "진단"]):
            updated_info["medical_history_mentioned"] = True
        
        return updated_info
    
    def process_conversation(self, user_input: str, has_image: bool = False) -> Dict[str, Any]:
        """Main conversation processing function"""
        # Update conversation history
        self.conversation_state["conversation_history"] += f"\n사용자: {user_input}"
        
        # Extract medical information
        self.conversation_state["collected_info"] = self._extract_medical_info(
            user_input, self.conversation_state["collected_info"]
        )
        
        # Update image information
        if has_image:
            self.conversation_state["has_image"] = True
        
        # Execute prompt
        chain = self.orchestrator_prompt | self.llm
        response = chain.invoke({
            "user_input": user_input,
            "conversation_stage": self.conversation_state["conversation_stage"],
            "collected_info": str(self.conversation_state["collected_info"]),
            "has_image": self.conversation_state["has_image"],
            "conversation_history": self.conversation_state["conversation_history"]
        })
        
        # Parse response
        parsed_response = self._parse_response(response.content)
        
        # Update state
        if parsed_response["decision"]:
            self.conversation_state["decision"] = parsed_response["decision"]
            self.conversation_state["reason"] = parsed_response["reason"]
            self.conversation_state["context"] = parsed_response["context"]
            self.conversation_state["conversation_stage"] = self._update_conversation_stage(
                parsed_response["decision"]
            )
        
        # Add AI response to conversation history
        self.conversation_state["conversation_history"] += f"\n의료진: {parsed_response['message']}"
        
        return {
            "message": parsed_response["message"],
            "decision": parsed_response["decision"],
            "reason": parsed_response["reason"],
            "context": parsed_response["context"],
            "conversation_stage": self.conversation_state["conversation_stage"],
            "collected_info": self.conversation_state["collected_info"]
        }
    
    def analyze_radiology_image(self, image_analysis: str, symptoms: str = "", basic_info: str = "", medical_history: str = "") -> str:
        """Analyze radiological image and provide patient-friendly explanation"""
        chain = self.radiology_prompt | self.llm
        response = chain.invoke({
            "image_analysis": image_analysis,
            "symptoms": symptoms,
            "basic_info": basic_info,
            "medical_history": medical_history
        })
        
        return response.content
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_state = {
            "conversation_stage": "greeting",
            "collected_info": {},
            "has_image": False,
            "conversation_history": "",
            "decision": None,
            "reason": None,
            "context": None
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Return current conversation summary"""
        return {
            "stage": self.conversation_state["conversation_stage"],
            "collected_info": self.conversation_state["collected_info"],
            "has_image": self.conversation_state["has_image"],
            "last_decision": self.conversation_state["decision"]
        }


class OrchestratorWorkflow:
    """LangGraph를 이용한 워크플로우 클래스"""
    
    def __init__(self, orchestrator_agent: OrchestratorAgent):
        self.orchestrator = orchestrator_agent
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(ConversationState)
        
        # 노드 추가
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("medblip_agent", self._medblip_node)
        workflow.add_node("end_conversation", self._end_node)
        
        # 시작점 설정
        workflow.set_entry_point("orchestrator")
        
        # 조건부 분기 설정
        workflow.add_conditional_edges(
            "orchestrator",
            self._decide_next_step,
            {
                "continue": "orchestrator",
                "route_to_medblip": "medblip_agent",
                "end": "end_conversation"
            }
        )
        
        # 엔드포인트 설정
        workflow.add_edge("medblip_agent", END)
        workflow.add_edge("end_conversation", END)
        
        return workflow.compile()
    
    def _orchestrator_node(self, state: ConversationState) -> ConversationState:
        """오케스트레이터 노드"""
        result = self.orchestrator.process_conversation(
            state["user_input"], 
            state.get("has_image", False)
        )
        
        state.update({
            "decision": result["decision"],
            "reason": result["reason"],
            "context": result["context"],
            "conversation_stage": result["conversation_stage"],
            "collected_info": result["collected_info"]
        })
        
        # 메시지 추가
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(AIMessage(content=result["message"]))
        
        return state
    
    def _medblip_node(self, state: ConversationState) -> ConversationState:
        """MedBLIP 에이전트 노드 (실제 구현 필요)"""
        # TODO: MedBLIP 에이전트 구현
        state["messages"].append(AIMessage(content="MedBLIP 에이전트로 전달되었습니다."))
        return state
    
    def _end_node(self, state: ConversationState) -> ConversationState:
        """대화 종료 노드"""
        state["messages"].append(AIMessage(content="건강검진 상담이 완료되었습니다."))
        return state
    
    def _decide_next_step(self, state: ConversationState) -> str:
        """다음 단계 결정"""
        decision = state.get("decision")
        
        if decision == "ROUTE_TO_MEDBLIP":
            return "route_to_medblip"
        elif decision == "END_CONSULTATION":
            return "end"
        else:
            return "continue"
    
    def run(self, user_input: str, has_image: bool = False) -> Dict[str, Any]:
        """워크플로우 실행"""
        initial_state = ConversationState(
            messages=[HumanMessage(content=user_input)],
            user_input=user_input,
            conversation_stage="greeting",
            collected_info={},
            has_image=has_image,
            conversation_history="",
            decision=None,
            reason=None,
            context=None
        )
        
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "messages": final_state["messages"],
            "decision": final_state.get("decision"),
            "context": final_state.get("context"),
            "conversation_stage": final_state.get("conversation_stage"),
            "collected_info": final_state.get("collected_info")
        }