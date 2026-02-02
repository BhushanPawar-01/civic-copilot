from llm.hf_client import HFClient
from llm.gemini_client import GeminiClient
from llm.openai_client import OpenAIClient
from memory.session_manager import SessionManager
from memory.summarizer import MemorySummarizer
from orchestrator.orchestrator import AgentOrchestrator
from rag.retriever import CivicRetriever
from config import settings

def get_llm_client():
    """Factory function to get the LLM client based on the provider."""
    if settings.LLM_PROVIDER == "hf":
        return HFClient()
    elif settings.LLM_PROVIDER == "gemini":
        return GeminiClient()
    elif settings.LLM_PROVIDER == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

# Singleton instances for the PoC
llm_client = get_llm_client()
session_manager = SessionManager()
retriever = CivicRetriever(index_path=settings.FAISS_INDEX_PATH)
summarizer = MemorySummarizer(llm_client)

def get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_client=llm_client,
        session_manager=session_manager,
        summarizer=summarizer,
        retriever=retriever
    )