import re
import json
from typing import Optional
from llm.base import LLMClient
from schemas.intent import IntentResponse

class IntentAgent:
    """
    Agent responsible for classifying civic domain, task type, and 
    extracting key entities from user queries with robust JSON parsing.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a Civic Intent Specialist. Your goal is to map user queries to specific "
            "government service domains. \n\n"
            "ALLOWED DOMAINS:\n"
            "- 'passport': Use for any queries regarding application, renewal, delays, or police verification regarding passport.\n"
            "- 'voter_id': Use for registration for voter ID, changes related to voter ID, or election-related queries.\n"
            "- 'general': Use only if the query does not fit the above categories.\n\n"
            "TASK TYPES:\n"
            "- 'information_retrieval', 'status_tracking', 'grievance', 'eligibility'.\n\n"
            "CRITICAL: You must output a valid JSON object with these EXACT keys:\n"
            "{\n"
            "  \"detected_domain\": \"string\",\n"
            "  \"task_type\": \"string\",\n"
            "  \"confidence_score\": float (0.0 to 1.0),\n"
            "  \"entities\": {},\n"
            "  \"requires_clarification\": boolean,\n"
            "  \"clarifying_question\": \"string or null\"\n"
            "}"
        )

    async def process(self, query: str, context: str) -> IntentResponse:
        """
        Processes a user query and conversation context into a structured IntentResponse.
        """
        prompt = f"""
        CONVERSATION CONTEXT: {context}
        USER QUERY: {query}

        INSTRUCTIONS:
        1. Classify the query into one of the ALLOWED DOMAINS. If it's about a passport, you MUST use 'passport'.
        2. Identify the specific task type.
        3. Extract entities like application numbers, locations, or names.
        4. If the query is vague (e.g., just saying 'my document is late'), set requires_clarification to true.
        5. Output ONLY the JSON object. No preamble.
        """

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> IntentResponse:
        """Helper to safely extract and map JSON from the Intent Agent output."""
        try:
            content_cleaned = content.strip()
            # Handle markdown blocks
            if "```json" in content_cleaned:
                content_cleaned = content_cleaned.split("```json")[1].split("```")[0].strip()
            
            # Robust extraction of the JSON object
            match = re.search(r"(\{.*\})", content_cleaned, re.DOTALL)
            if not match:
                raise ValueError("No JSON found")

            data = json.loads(match.group(1))

            # Adaptive Mapping Logic
            domain = data.get("detected_domain", data.get("domain", "general"))
            
            # Ensure the domain is normalized to your index tags
            if "passport" in domain.lower():
                domain = "passport"
            elif "voter" in domain.lower():
                domain = "voter_id"

            conf = data.get("confidence_score", data.get("confidence"))
            
            # Confidence Gate Reinforcement
            if conf is None:
                conf = 0.0 if data.get("requires_clarification") else 0.8
            
            mapped_data = {
                "detected_domain": domain,
                "task_type": data.get("task_type", "information_retrieval"),
                "confidence_score": float(conf),
                "entities": data.get("entities", {}),
                "requires_clarification": bool(data.get("requires_clarification", False)),
                "clarifying_question": data.get("clarifying_question")
            }

            return IntentResponse(**mapped_data)

        except Exception:
            return IntentResponse(
                detected_domain="general",
                task_type="error_fallback",
                confidence_score=0.0,
                requires_clarification=True,
                clarifying_question="I'm having trouble categorizing your request. Could you please provide more details?"
            )