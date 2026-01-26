from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class AgentRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = None  # Conversation history

class AgentResponse(BaseModel):
    answer: str
