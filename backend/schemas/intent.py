from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator

class IntentResponse(BaseModel):
    """Output schema for the Intent Agent to categorize user needs."""
    
    detected_domain: str = Field(
        ..., 
        description="The specific civic area identified (e.g., 'passport', 'voter_id', 'taxation')."
    )
    
    task_type: str = Field(
        ..., 
        description="The type of action required: 'information_retrieval', 'status_tracking', 'grievance', or 'eligibility'."
    )
    
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="The agent's certainty in the detected intent."
    )
    
    entities: Dict[str, str] = Field(
        default_factory=dict, 
        description="Extracted key-value pairs like {'application_id': '123', 'state': 'Karnataka'}."
    )
    
    requires_clarification: bool = Field(
        default=False, 
        description="Flag set to True if the query is too ambiguous to proceed to RAG."
    )
    
    clarifying_question: Optional[str] = Field(
        None, 
        description="The specific question to ask the user if confidence is low."
    )

    @field_validator('confidence_score')
    @classmethod
    def set_clarification_flag(cls, v: float, info):
        """Logic can be added here to force clarification if confidence is below a threshold."""
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "detected_domain": "passport",
                "task_type": "status_tracking",
                "confidence_score": 0.92,
                "entities": {"location": "Delhi", "service_type": "Fresh Passport"},
                "requires_clarification": False
            }
        }