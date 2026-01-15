from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal


class AgentAction(BaseModel):
    action: Literal["response", "tool"]
    tool_name: Optional[str] = None
    args: Optional[Dict] = None
    content: Optional[str] = None
