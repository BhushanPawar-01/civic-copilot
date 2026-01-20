from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ActionStep(BaseModel):
    """A discrete step the user needs to take."""
    order: int = Field(..., description="The sequence number of the step")
    action: str = Field(..., description="Short, imperative description of the task (e.g., 'Submit Form 4')")
    details: str = Field(..., description="Detailed instructions on how to complete this step")
    link: Optional[str] = Field(None, description="Direct URL to a form or portal relevant to this step")

class EscalationPath(BaseModel):
    """Guidance for the user if the standard process fails or is delayed."""
    authority: str = Field(..., description="The office or officer to contact (e.g., 'Regional Passport Officer')")
    method: str = Field(..., description="Preferred contact method: 'Email', 'Grievance Portal', or 'In-person'")
    template_draft: Optional[str] = Field(None, description="A pre-written draft message the user can use")

class ActionResponse(BaseModel):
    """The structured output for the Action Agent."""
    summary: str = Field(..., description="A brief overview of the recommended course of action")
    immediate_steps: List[ActionStep] = Field(..., description="Chronological list of actions for the user")
    escalation_guidance: Optional[EscalationPath] = Field(None, description="Steps to take if the user is facing delays")
    estimated_timeline: Optional[str] = Field(None, description="Expected time for completion based on policy facts")