import uuid
import time
from typing import Optional, Dict, Any
from llm.base import LLMClient
from memory.session_manager import SessionManager
from memory.summarizer import MemorySummarizer
from schemas.response import FinalResponse
from schemas.intent import IntentResponse
from agents.intent_agent import IntentAgent
from agents.policy_agent import PolicyAgent
from agents.action_agent import ActionAgent
from agents.verification_agent import VerificationAgent
from audit.trace_logger import audit_logger

class AgentOrchestrator:
    """The central brain coordinating the multi-agent civic workflow."""

    def __init__(
        self, 
        llm_client: LLMClient, 
        session_manager: SessionManager,
        summarizer: MemorySummarizer,
        retriever
    ):
        self.llm = llm_client
        self.session_manager = session_manager
        self.summarizer = summarizer
        self.retriever = retriever
        # Initialize specialized agents
        self.intent_agent = IntentAgent(llm_client)
        self.policy_agent = PolicyAgent(llm_client, retriever)
        self.action_agent = ActionAgent(llm_client)
        self.verification_agent = VerificationAgent(llm_client)

    async def run_workflow(self, user_query: str, session_id: str) -> FinalResponse:
        """
        Executes the full agentic loop as defined in the flowchart.
        """
        trace_id = str(uuid.uuid4())
        
        # 1. Fetch Conversation Context & Summary Memory
        context = await self.session_manager.get_context_for_orchestrator(session_id)
        
        # 2. Intent Agent: Interpret user need
        start_time = time.time()
        intent: IntentResponse = await self.intent_agent.process(user_query, context)
        audit_logger.log_agent_step(
            trace_id=trace_id,
            agent_name="IntentAgent",
            input_data=user_query,
            output_data=intent,
            duration=time.time() - start_time
        )
        
        # 3. Confidence Gate: Branching logic
        if intent.requires_clarification or intent.confidence_score < 0.7:
            return await self._handle_clarification(session_id, intent, trace_id)

        # 4. Policy / Knowledge Agent: RAG-based retrieval
        start_time = time.time()
        policy_data = await self.policy_agent.process(intent, user_query)
        audit_logger.log_agent_step(
            trace_id=trace_id,
            agent_name="PolicyAgent",
            input_data=intent,
            output_data=policy_data,
            duration=time.time() - start_time
        )
        
        # 5. Action Agent: Draft recommended steps
        start_time = time.time()
        action_data = await self.action_agent.process(policy_data)
        audit_logger.log_agent_step(
            trace_id=trace_id,
            agent_name="ActionAgent",
            input_data=policy_data,
            output_data=action_data,
            duration=time.time() - start_time
        )
        
        # 6. Verification Agent: Final safety & grounding check
        verification = await self.verification_agent.process(
            query=user_query,
            policy=policy_data,
            action=action_data
        )
        audit_logger.log_agent_step(
            trace_id=trace_id,
            agent_name="VerificationAgent",
            input_data=[user_query, policy_data, action_data],
            output_data=verification,
            duration=time.time() - start_time
        )

        # 7. Final Response Builder
        final_response = FinalResponse(
            session_id=session_id,
            answer_text=action_data.summary,
            intent_data=intent,
            policy_data=policy_data,
            action_data=action_data,
            confidence_level="High" if verification.is_validated else "Medium",
            is_verified=verification.is_validated,
            risk_disclaimer=verification.disclaimer if not verification.is_validated else None,
            trace_id=trace_id
        )

        # 8. Memory Summarizer: Update session state for next turn
        new_summary = await self.summarizer.summarize_turn(
            current_summary=context,
            user_query=user_query,
            final_response=final_response
        )
        
        await self.session_manager.update_session_after_turn(
            session_id=session_id,
            new_summary=new_summary,
            domain=intent.detected_domain,
            extracted_entities=intent.entities
        )

        return final_response

    async def _handle_clarification(self, session_id: str, intent: IntentResponse, trace_id: str) -> FinalResponse:
        """Helper to return a clarification payload when intent is ambiguous."""
        return FinalResponse(
            session_id=session_id,
            answer_text=intent.clarifying_question or "Could you provide more details about your request?",
            intent_data=intent,
            confidence_level="Low",
            is_verified=True,
            trace_id=trace_id
        )