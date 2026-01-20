from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized response format for all providers."""
    content: str
    raw_response: Any
    model_name: str
    usage: Dict[str, int] = {}


class LLMClient(ABC):
    """Abstract Base Class for LLM providers[cite: 83]."""

    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Main generation method for agents to call[cite: 84, 85]."""
        pass