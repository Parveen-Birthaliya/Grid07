from pydantic import BaseModel, validator, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class ConfidenceLevel(str, Enum):
    """Confidence level for route matching and Jailbreak detection"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BotPersona(BaseModel):
    """Define each bot's personality"""
    bot_id: str
    name: str
    description: str

class RoutingResult(BaseModel):
    bot_id: str 
    similarity_score: float = Field(..., gt=0.0, le=1.0, description="Similarity score always between 0 and 1")
    confidence: ConfidenceLevel
    persona: str
    

class SearchResult(BaseModel):
    query: str
    results: str
    source: Optional[str] = None
    retry_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationMessage(BaseModel):
    role: str = Field(..., description="user or bot")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationState(BaseModel):
    history: List[ConversationMessage] = Field(default_factory=list)
    max_history: int = 5

    def add_message(self, role: str, content: str):
        self.history.append(ConversationMessage(role=role, content=content))

        # FIFO logic
        if len(self.history) > self.max_history:
            self.history.pop(0)

class AgentState(BaseModel):
    """State machine for each query """
    query_id: str
    query: str
    matched_bots: List[RoutingResult]
    context: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    search_query: Optional[str] = None
    search_results: Optional[SearchResult] = None
    generated_post: Optional[str] = None
    conversation: ConversationState = Field(default_factory=ConversationState)
    error: Optional[str] = None
    error_node: Optional[str] = None  # which node failed (routing/search/generation)



class JailbreakDetectionResult(BaseModel):
    is_jailbreak: bool
    confidence: ConfidenceLevel
    patterns_detected: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class EngineResponse(BaseModel):
    """Response from the Grid07 engine"""
    success: bool
    query_id: str
    original_query: str
    matched_bots: List[RoutingResult] = Field(default_factory=list)
    generated_post: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)




