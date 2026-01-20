import re
import json
from typing import Optional
from pydantic import BaseModel, Field
from llm.base import LLMClient
from schemas.policy import PolicyResponse
from schemas.action import ActionResponse

class VerificationResult(BaseModel):
    """Internal schema for the verification agent's output."""
    is_validated: bool = Field(..., description="True if the response is fully grounded in policy facts.")
    reasoning: str = Field(..., description="Internal logic for the validation decision.")
    disclaimer: Optional[str] = Field(None, description="A safety warning if uncertainty is detected.")

class VerificationAgent:
    """
    Agent responsible for hallucination detection and grounding verification.
    Acts as the final safety gate before the user sees the response.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a Civic Safety Auditor. Your sole job is to cross-check "
            "suggested actions against official policy facts. \n\n"
            "CRITERIA FOR FAILURE:\n"
            "1. The Action Agent suggests a step not supported by Policy Facts.\n"
            "2. The Action Agent ignores a critical warning or timeline in the Policy.\n"
            "3. The response contains 'hallucinated' contact details or URLs.\n"
            "Output strictly in JSON matching the VerificationResult schema."
        )

    async def process(
        self, 
        query: str, 
        policy: PolicyResponse, 
        action: ActionResponse
    ) -> VerificationResult:
        """
        Performs the final validation check.
        Matches the 'Verification Agent' step in the flowchart.
        """
        
        # If no policy facts were found, we automatically flag for uncertainty
        if not policy.verified_facts:
            return VerificationResult(
                is_validated=False,
                reasoning="No policy facts available to ground the response.",
                disclaimer="Warning: This guidance is general and not linked to official documentation."
            )

        facts_text = "\n".join([f"- {f.fact}" for f in policy.verified_facts])
        steps_text = "\n".join([f"- {s.action}: {s.details}" for s in action.immediate_steps])

        prompt = f"""
        USER QUERY: {query}
        
        OFFICIAL POLICY FACTS:
        {facts_text}

        PROPOSED ACTION STEPS:
        {steps_text}

        TASK:
        Check if the 'Proposed Action Steps' are 100% grounded in the 'Official Policy Facts'. 
        If there is any fabrication, set is_validated to False.
        """

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.0  # Absolute zero for maximum consistency
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> VerificationResult:
        """Helper to safely extract and map JSON from the Verification Agent output."""
        try:
            # 1. Regex to isolate the JSON block
            content_cleaned = content.strip()
            match = re.search(r"(\{.*\})", content_cleaned, re.DOTALL)
            
            if not match:
                raise ValueError("No JSON block found in output")

            data = json.loads(match.group(1))

            # 2. Key mapping and validation
            # Remap variations like 'validated' or 'valid' to 'is_validated'
            is_validated = data.get("is_validated", data.get("validated", data.get("valid", False)))
            reasoning = data.get("reasoning", "No specific reasoning provided by auditor.")
            disclaimer = data.get("disclaimer")

            # If validation failed but no disclaimer was provided, generate a fallback disclaimer
            if not is_validated and not disclaimer:
                disclaimer = "Please cross-verify this information with official government portals before proceeding."

            return VerificationResult(
                is_validated=bool(is_validated),
                reasoning=str(reasoning),
                disclaimer=disclaimer
            )

        except Exception as e:
            # 3. Safe Failure Fallback
            # If the auditor fails to communicate properly, we assume the response is NOT safe.
            return VerificationResult(
                is_validated=False,
                reasoning=f"Internal Auditor Parsing Error: {str(e)}",
                disclaimer="Note: Please verify these steps on the official government portal as internal verification failed."
            )