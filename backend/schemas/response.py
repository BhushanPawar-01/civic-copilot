from typing import List, Optional
from pydantic import BaseModel, Field
from .intent import IntentResponse
from .policy import PolicyResponse
from .action import ActionResponse

class FinalResponse(BaseModel):
    """The complete payload sent to the Streamlit frontend."""
    session_id: str = Field(..., description="Unique identifier for the current conversation")
    
    # Core Content
    answer_text: str = Field(..., description="The final, user-friendly natural language response")
    
    # Structured Data for UI Components
    intent_data: IntentResponse = Field(..., description="The analyzed intent and confidence score")
    policy_data: Optional[PolicyResponse] = Field(None, description="Retrieved facts and source citations")
    action_data: Optional[ActionResponse] = Field(None, description="Recommended next steps and drafts")
    
    # Trust & Auditability Layer
    confidence_level: str = Field(..., description="Final confidence rating: 'High', 'Medium', or 'Low'")
    is_verified: bool = Field(default=False, description="Whether the Verification Agent has cleared this response")
    risk_disclaimer: Optional[str] = Field(None, description="Safety notes or legal disclaimers for uncertain responses")
    
    # Internal Audit (Hidden by default in UI)
    trace_id: str = Field(..., description="ID for looking up the full execution trace in audit logs")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123_abc",
                "answer_text": "Based on official guidelines, your passport delay can be addressed by...",
                "confidence_level": "High",
                "is_verified": True
            }
        }