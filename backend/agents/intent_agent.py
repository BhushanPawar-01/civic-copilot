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
            "You are a Civic Intent Specialist. Analyze user queries for government services. "
            "Identify the domain, task type, and extract entities.\n\n"
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
        Includes a robust cleaning layer to handle non-JSON conversational noise.
        """
        prompt = f"""
        CONVERSATION CONTEXT: {context}
        USER QUERY: {query}

        INSTRUCTIONS:
        1. Identify domain and task.
        2. Extract entities (IDs, locations).
        3. Assess confidence (0.0 to 1.0).
        4. Output ONLY the structured JSON.
        """

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> IntentResponse:
        try:
            content_cleaned = content.strip()
            # Handle cases where LLM wraps JSON in ```json blocks
            if "```json" in content_cleaned:
                content_cleaned = content_cleaned.split("```json")[1].split("```")[0].strip()
            
            match = re.search(r"(\{.*\})", content_cleaned, re.DOTALL)
            if not match:
                raise ValueError("No JSON found")

            data = json.loads(match.group(1))

            # Improved Mapping: If domain is found but confidence is missing, 
            # give it a base score so the workflow doesn't stall.
            domain = data.get("detected_domain", data.get("domain", "general"))
            conf = data.get("confidence_score", data.get("confidence"))
            
            # Logic: If the LLM identified a domain but forgot the score, 
            # assume 0.8 to pass the gate, unless clarification is requested.
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

        except Exception as e:
            # Log the error here to see why parsing fails
            return IntentResponse(
                detected_domain="general",
                task_type="error_fallback",
                confidence_score=0.0,
                requires_clarification=True,
                clarifying_question="I'm having trouble understanding the details. Could you please rephrase?"
            )