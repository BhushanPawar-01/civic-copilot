import re
import json
from typing import List, Dict, Any
from llm.base import LLMClient
from schemas.policy import PolicyResponse
from schemas.action import ActionResponse, ActionStep, EscalationPath

class ActionAgent:
    """
    Agent responsible for converting policy facts into actionable steps 
    and drafts with robust JSON parsing and error recovery.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "You are a Civic Guide. Your job is to take verified policy facts "
            "and turn them into clear, numbered steps for a citizen. \n\n"
            "Tone should be helpful, clear, and professional. If the policy "
            "allows for escalation, provide a draft template. \n\n"
            "CRITICAL: Output ONLY a valid JSON object matching the ActionResponse schema."
        )

    async def process(self, policy_data: PolicyResponse) -> ActionResponse:
        """
        Translates policy facts into a structured ActionResponse.
        """
        if not policy_data.verified_facts:
            return ActionResponse(
                summary="I couldn't find specific official steps for this request in our current documents.",
                immediate_steps=[]
            )

        facts_summary = "\n".join([f"- {f.fact}" for f in policy_data.verified_facts])

        prompt = f"""
        DOMAIN: {policy_data.domain}
        VERIFIED FACTS:
        {facts_summary}

        TASK:
        1. Summarize the situation for the user.
        2. Provide a chronological list of 'immediate_steps'.
        3. Create an 'escalation_guidance' section if the facts mention delays or grievances.
        """

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.4 
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> ActionResponse:
        """Helper to safely extract and map JSON from the Action Agent output."""
        try:
            # 1. Regex to isolate the JSON block
            content_cleaned = content.strip()
            match = re.search(r"(\{.*\})", content_cleaned, re.DOTALL)
            
            if not match:
                raise ValueError("No JSON block found in output")

            data = json.loads(match.group(1))

            # 2. Key mapping and validation
            # Ensures lists and nested objects exist even if the LLM skips them
            summary = data.get("summary", "Based on the official facts, please follow these steps.")
            steps_raw = data.get("immediate_steps", [])
            
            immediate_steps = [
                ActionStep(
                    order=s.get("order", i+1),
                    action=s.get("action", "General Step"),
                    details=s.get("details", ""),
                    link=s.get("link")
                ) for i, s in enumerate(steps_raw)
            ]

            escalation_raw = data.get("escalation_guidance")
            escalation_guidance = None
            if escalation_raw:
                escalation_guidance = EscalationPath(
                    authority=escalation_raw.get("authority", "Relevant Authority"),
                    method=escalation_raw.get("method", "Official Portal"),
                    template_draft=escalation_raw.get("template_draft")
                )

            return ActionResponse(
                summary=summary,
                immediate_steps=immediate_steps,
                escalation_guidance=escalation_guidance,
                estimated_timeline=data.get("estimated_timeline")
            )

        except Exception as e:
            # 3. Fallback for failed parsing
            return ActionResponse(
                summary="We found relevant policy facts, but encountered an error while formatting the steps. Please refer to official government portal guidelines.",
                immediate_steps=[]
            )