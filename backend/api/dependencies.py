from llm.hf_client import HFClient
from memory.session_manager import SessionManager
from memory.summarizer import MemorySummarizer
from orchestrator.orchestrator import AgentOrchestrator
from rag.retriever import CivicRetriever
from config import settings

# Singleton instances for the PoC
llm_client = HFClient()
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