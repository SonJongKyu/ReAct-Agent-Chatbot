"""
(260120) State 정의
"""


"""AgentState 정의 - 새 필드 추가"""
from typing import TypedDict, List, Dict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """Agent의 상태 정의"""
    # 기본 필드
    question: str
    messages: Annotated[List[BaseMessage], operator.add] # operator.add: 새로운 메시지를 자동으로 추가함
    user_id: str
    
    # Query rewriting
    queries: List[str]
    
    # Tool execution
    tool_results: Dict[str, str]
    
    # Final answer
    final_answer: str
    
    # === 새로 추가된 필드 ===
    question_type: str  # "direct", "context_search", "new_search"
    classification_reason: str  # 분류 이유