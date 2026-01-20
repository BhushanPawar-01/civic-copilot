from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Import the Orchestrator and the FinalResponse schema
from orchestrator.orchestrator import AgentOrchestrator
from schemas.response import FinalResponse
from api.dependencies import get_orchestrator

router = APIRouter(prefix="/api/v1", tags=["civic-chat"])

class ChatRequest(BaseModel):
    """Schema for the incoming user message."""
    query: str = Field(..., min_length=1, example="Why is my passport delayed?")
    session_id: Optional[str] = Field(None, description="Provide to continue a conversation.")

@router.post("/chat", response_model=FinalResponse)
async def chat_endpoint(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Primary endpoint to interact with the Civic Copilot.
    Orchestrates the multi-agent workflow and returns a verified response.
    """
    try:
        # 1. Generate a session_id if it's a new conversation
        session_id = request.session_id or "session_" + str(uuid.uuid4())[:8]
        
        # 2. Trigger the multi-agent workflow
        response = await orchestrator.run_workflow(
            user_query=request.query,
            session_id=session_id
        )
        
        return response

    except Exception as e:
        # Logging here via your trace_logger would be ideal
        print(f"Workflow Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your civic inquiry."
        )

@router.get("/health")
async def health_check():
    """Simple health check for the API."""
    return {"status": "healthy", "service": "civic-copilot-backend"}