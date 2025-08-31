from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
from langchain.schema import BaseMessage

class ConversationState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    conversation_stage: str
    collected_info: Dict[str, Any]
    has_image: bool
    conversation_history: str
    decision: Optional[str]
    reason: Optional[str]
    context: Optional[Dict[str, Any]]