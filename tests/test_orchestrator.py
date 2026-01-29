import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from llm.hf_client import HFClient
from memory.session_manager import SessionManager
from memory.summarizer import MemorySummarizer
from orchestrator.orchestrator import AgentOrchestrator
from rag.retriever import CivicRetriever
from config import settings

async def test_full_workflow():
    print("Testing Full Orchestrator Workflow...")
    
    # Setup dependencies
    llm = HFClient()
    sm = SessionManager()
    # Ensure this path points to your actual dummy index
    retriever = CivicRetriever(index_path=settings.FAISS_INDEX_PATH)
    summarizer = MemorySummarizer(llm)
    
    orchestrator = AgentOrchestrator(llm, sm, summarizer, retriever)
    
    session_id = "test_user_001"
    query = "How long does passport processing take?"
    
    print(f"\nRunning workflow for: '{query}'")
    
    try:
        response = await orchestrator.run_workflow(query, session_id)
        print("\nFINAL RESPONSE:")
        print(f"Text: {response.answer_text}")
        print(f"Verified: {response.is_verified}")
        print(f"Trace ID: {response.trace_id}")
    except Exception as e:
        print(f"\n Workflow Crashed: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_workflow())