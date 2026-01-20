from typing import Optional
from llm.base import LLMClient
from schemas.response import FinalResponse

class MemorySummarizer:
    """Condenses conversation history into structured, production-grade memory."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a Civic Memory Specialist. Your job is to update a running "
            "summary of a citizen's inquiry. Focus on: 1) Key personal facts/entities "
            "2) The current status of their request 3) Unresolved questions. "
            "Keep the summary concise and professional. Avoid repeating everything."
        )

    async def summarize_turn(
        self, 
        current_summary: str, 
        user_query: str, 
        final_response: FinalResponse
    ) -> str:
        """
        Processes the latest turn to generate an updated 'Summary Memory'.
        Matches the 'Summarize Conversation' step in the flowchart.
        """
        
        prompt = f"""
        PREVIOUS SUMMARY:
        {current_summary if current_summary else "No previous history."}

        LATEST USER QUERY:
        {user_query}

        LATEST ASSISTANT RESPONSE:
        {final_response.answer_text}

        TASK:
        Generate an updated, single-paragraph summary that incorporates any new 
        facts (like application IDs or specific locations) and tracks the 
        current progress of the civic task.
        """

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=256,
                temperature=0.3  # Lower temperature for consistency
            )
            return response.content.strip()
        except Exception as e:
            # Fallback: return the existing summary so the session doesn't break
            print(f"Summarization error: {e}")
            return current_summary