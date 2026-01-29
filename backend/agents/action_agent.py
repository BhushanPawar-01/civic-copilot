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
            "You are a helpful and clear Civic Guide. Your primary role is to synthesize verified policy facts "
            "into a user-friendly, actionable plan. You must adhere to the following rules:\n"
            "1.  **Summarize Key Information:** Your main 'summary' must be a concise, easy-to-understand paragraph that directly answers the user's original question using the provided facts.\n"
            "2.  **Actionable Steps:** Convert the facts into a clear, chronological list of 'immediate_steps' a citizen should follow.\n"
            "3.  **Escalation and Timelines:** If the facts mention grievances, delays, or specific timelines, populate the 'escalation_guidance' and 'estimated_timeline' fields.\n"
            "4.  **Strict JSON Output:** You must output a single, valid JSON object that strictly adheres to the `ActionResponse` schema. Do not add any text before or after the JSON object."
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
        **User's Goal:** To understand the process and timeline based on official rules.
        **Domain:** {policy_data.domain}

        **Verified Facts from Official Documents:**
        ```
        {facts_summary}
        ```
        **Your Task:**
        Based *only* on the verified facts, generate a comprehensive action plan for the user. The summary should be a direct and helpful answer to their question.

        **Output Format (JSON):**
        ```json
        {{
          "summary": "A clear, paragraph-style summary that directly answers the user's question about processing time, mentioning the different timelines (e.g., normal vs. Tatkaal vs. complex cases).",
          "immediate_steps": [
            {{
              "order": 1,
              "action": "A clear action for the user to take (e.g., 'Submit Application').",
              "details": "Provide context or details for this step, derived from the facts.",
              "link": "A URL if mentioned in the facts, otherwise null."
            }}
          ],
          "escalation_guidance": {{
            "authority": "The name of the authority to contact for grievances (e.g., 'CPGRAMS').",
            "method": "The method of contact (e.g., 'Official Portal').",
            "template_draft": "A pre-drafted message for the user if applicable, otherwise null."
          }},
          "estimated_timeline": "A concise summary of the timeline expectations (e.g., '1-3 working days for normal applications, up to 30 days for complex cases')."
        }}
        ```
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