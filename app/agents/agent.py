import os
from typing import Dict, Any, Optional
from langchain.chat_models.base import BaseChatModel

from app.core.llm_factory import get_llm_for_agent
from app.core.observability import get_callbacks
from .prompts.prompt import ORCHESTRATOR_AGENT_PROMPT, RADIOLOGY_ANALYSIS_PROMPT


class OrchestratorAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            try:
                callbacks = get_callbacks()
                self.llm = get_llm_for_agent(
                    agent_type="generic",
                    api_key=api_key,
                    callbacks=callbacks
                )
            except Exception:
                self.llm = None
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
        
        # Execute prompt (online) or use offline guidance
        if self.llm is not None:
            chain = self.orchestrator_prompt | self.llm
            response = chain.invoke({
                "user_input": user_input,
                "conversation_stage": self.conversation_state["conversation_stage"],
                "collected_info": str(self.conversation_state["collected_info"]),
                "has_image": self.conversation_state["has_image"],
                "conversation_history": self.conversation_state["conversation_history"],
            })
            parsed_response = self._parse_response(response.content)
        else:
            # Minimal offline flow
            stage = self.conversation_state["conversation_stage"]
            if stage == "greeting":
                parsed_response = {
                    "decision": "COLLECT_BASIC_INFO",
                    "reason": "Collect minimal info",
                    "message": (
                        "안녕하세요! MedBLIP 기반 의료 상담 서비스입니다. "
                        "간단한 정보를 알려주시고 이미지를 업로드해 주세요."
                    ),
                    "context": None,
                }
            elif stage == "basic_info":
                parsed_response = {
                    "decision": "REQUEST_IMAGE",
                    "reason": "Proceed to image upload",
                    "message": "감사합니다. 이제 방사선 이미지를 업로드해주세요.",
                    "context": None,
                }
            elif stage in ("image_upload", "analysis") and self.conversation_state["has_image"]:
                parsed_response = {
                    "decision": "PROVIDE_EXPLANATION",
                    "reason": "Explain findings",
                    "message": "이미지 분석을 완료하고 설명을 제공하겠습니다.",
                    "context": None,
                }
            else:
                parsed_response = {
                    "decision": "REQUEST_IMAGE",
                    "reason": "Await image",
                    "message": "이미지를 업로드해 주세요.",
                    "context": None,
                }
        
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
            "collected_info": self.conversation_state["collected_info"],
        }
    
    def analyze_radiology_image(self, image_analysis: str, symptoms: str = "", basic_info: str = "", medical_history: str = "") -> str:
        """Analyze radiological image and provide patient-friendly explanation"""
        if self.llm is None:
            return (
                f"이미지 분석 결과: {image_analysis}\n"
                "자세한 설명은 의료진 상담을 권장드립니다."
            )
        chain = self.radiology_prompt | self.llm
        response = chain.invoke(
            {
                "image_analysis": image_analysis,
                "symptoms": symptoms,
                "basic_info": basic_info,
                "medical_history": medical_history,
            }
        )
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
            "last_decision": self.conversation_state["decision"],
        }

"""
Note: Removed experimental LangGraph workflow and related model types to keep the project slim.
"""
