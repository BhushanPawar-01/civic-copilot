import uuid
from typing import Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

class SessionState(BaseModel):
    """Internal model for the persisted state of a user session."""
    session_id: str
    last_updated: datetime = Field(default_factory=datetime.now)
    summary_memory: str = ""  # The running condensed history
    active_domain: Optional[str] = None
    pending_entities: Dict[str, Any] = {} # Entities like application_id
    turn_count: int = 0

class SessionManager:
    """Handles loading and updating session state and memory."""

    def __init__(self):
        # In a real production system, this would be a Redis or DB call.
        # For the PoC, we use an in-memory dictionary.
        self._storage: Dict[str, SessionState] = {}

    async def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        """Retrieves an existing session or initializes a new one."""
        if not session_id or session_id not in self._storage:
            new_id = session_id or str(uuid.uuid4())
            state = SessionState(session_id=new_id)
            self._storage[new_id] = state
            return state
        
        return self._storage[session_id]

    async def get_context_for_orchestrator(self, session_id: str) -> str:
        """
        Fetches the 'Conversation Context'.
        Combines the summary memory with any relevant pending entities.
        """
        state = await self.get_or_create_session(session_id)
        
        context_parts = []
        if state.summary_memory:
            context_parts.append(f"Summary of previous interaction: {state.summary_memory}")
        
        if state.pending_entities:
            context_parts.append(f"Known context: {state.pending_entities}")
            
        return "\n".join(context_parts) if context_parts else "No previous context."

    async def update_session_after_turn(
        self, 
        session_id: str, 
        new_summary: str, 
        domain: Optional[str] = None,
        extracted_entities: Optional[Dict] = None
    ):
        """
        Updates the session state after the Memory Summarizer step.
        """
        state = await self.get_or_create_session(session_id)
        
        state.summary_memory = new_summary
        state.turn_count += 1
        state.last_updated = datetime.now()
        
        if domain:
            state.active_domain = domain
            
        if extracted_entities:
            state.pending_entities.update(extracted_entities)
        
        self._storage[session_id] = state