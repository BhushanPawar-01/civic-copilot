import asyncio
import sys
import os

# Add the backend directory to sys.path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from llm.hf_client import HFClient
from agents.intent_agent import IntentAgent

async def debug_intent_agent():
    print("🔍 Testing Intent Agent Isolation...")
    
    # 1. Initialize Client
    client = HFClient()
    agent = IntentAgent(client)
    
    # 2. Test Query
    test_query = "I'm applying for my first passport but my address proof is in my father's name. Can I still register and what documents will they accept?"
    context = "No previous history."
    
    print(f"\n👉 User Query: {test_query}")
    
    # We'll bypass the agent's .process() briefly to see the raw output
    prompt = f"CONTEXT: {context}\nQUERY: {test_query}\nOutput ONLY JSON."
    raw_response = await client.generate(prompt=prompt, system_prompt=agent.system_prompt)
    
    print("\n--- RAW LLM OUTPUT ---")
    print(raw_response.content)
    print("----------------------")
    
    # 3. Test Actual Processing
    try:
        structured_intent = await agent.process(test_query, context)
        print("\n✅ Parsed Result:")
        print(f"Domain: {structured_intent.detected_domain}")
        print(f"Confidence: {structured_intent.confidence_score}")
        print(f"Requires Clarification: {structured_intent.requires_clarification}")
    except Exception as e:
        print(f"\n❌ Parsing Failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_intent_agent())