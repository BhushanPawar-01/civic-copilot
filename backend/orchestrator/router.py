from enum import Enum
from typing import Literal
from schemas.intent import IntentResponse

class WorkflowRoute(str, Enum):
    POLICY_KNOWLEDGE = "policy_knowledge"
    CLARIFICATION = "clarification"
    ERROR = "error"

class OrchestratorRouter:
    """
    Handles the logical branching of the civic workflow based on 
    agent outputs and confidence scores.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def determine_next_step(self, intent: IntentResponse) -> WorkflowRoute:
        """
        Evaluates the Intent Agent output to decide the next node in the flowchart.
        Matches the 'Confidence OK?' diamond in your architecture.
        """
        
        # 1. Explicit Clarification Flag
        if intent.requires_clarification:
            return WorkflowRoute.CLARIFICATION

        # 2. Confidence Score Threshold
        if intent.confidence_score < self.confidence_threshold:
            return WorkflowRoute.CLARIFICATION

        # 3. Domain Validation
        # If the domain is unknown or not supported yet, route to clarification/error
        if not intent.detected_domain or intent.detected_domain == "unknown":
            return WorkflowRoute.CLARIFICATION

        # 4. Default Success Route
        return WorkflowRoute.POLICY_KNOWLEDGE

    def should_escalate(self, verification_status: bool, uncertainty_flag: bool) -> bool:
        """
        Logic for the Verification Agent branch in the flowchart.
        Determines if a 'Risk & Disclaimer Layer' needs to be injected.
        """
        return not verification_status or uncertainty_flag