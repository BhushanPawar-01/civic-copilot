import re
import json
from typing import List, Dict, Any
from llm.base import LLMClient
from rag.retriever import CivicRetriever
from schemas.policy import PolicyResponse, PolicyFact, SourceMetadata
from schemas.intent import IntentResponse

class PolicyAgent:
    """
    Agent responsible for synthesizing RAG results into grounded civic facts.
    Includes a robust cleaning layer to handle model-specific JSON variations.
    """

    def __init__(self, llm_client: LLMClient, retriever: CivicRetriever):
        self.llm = llm_client
        self.retriever = retriever
        self.system_prompt = (
            "You are a Civic Policy Analyst. Your task is to extract verified facts "
            "from provided government document snippets. \n\n"
            "RULES:\n"
            "1. Only include information present in the snippets.\n"
            "2. If snippets conflict, note this in 'uncertainty_notes'.\n"
            "3. Assign each fact to its specific 'source_id'.\n"
            "4. Output strictly in JSON matching the PolicyResponse schema."
        )

    async def process(self, intent: IntentResponse, original_query: str) -> PolicyResponse:
        """
        Retrieves raw data and synthesizes it into grounded facts.
        Matches the 'Policy Agent' step in the flowchart.
        """
        # 1. Fetch relevant chunks from the retriever
        raw_results = await self.retriever.retrieve(
            query=original_query, 
            domain=intent.detected_domain,
            top_k=4
        )

        if not raw_results:
            return PolicyResponse(
                domain=intent.detected_domain,
                verified_facts=[],
                sources=[],
                uncertainty_notes="No official documents found for this query."
            )

        # 2. Prepare context for the LLM
        context_block = "\n\n".join([
            f"SOURCE_ID: {r['metadata'].source_id}\nCONTENT: {r['content']}" 
            for r in raw_results
        ])

        prompt = f"""
        USER QUERY: {original_query}
        IDENTIFIED DOMAIN: {intent.detected_domain}

        OFFICIAL DOCUMENT SNIPPETS:
        {context_block}

        TASK:
        Extract the specific rules, timelines, and requirements that answer the user query.
        """

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1
        )

        return self._parse_response(response.content, intent.detected_domain, raw_results)

    def _parse_response(self, content: str, domain: str, raw_results: List[Dict]) -> PolicyResponse:
        """Helper to safely extract and map JSON from the Policy Agent output."""
        try:
            # Step 1: Regex Extraction
            content_cleaned = content.strip()
            match = re.search(r"(\{.*\})", content_cleaned, re.DOTALL)
            
            if not match:
                raise ValueError("No JSON found in response")

            data = json.loads(match.group(1))

            # Step 2: Remap Facts
            facts_raw = data.get("verified_facts", [])
            verified_facts = [
                PolicyFact(
                    fact=f.get("fact", ""),
                    relevance_score=float(f.get("relevance_score", 1.0)),
                    source_refs=f.get("source_refs", f.get("references", []))
                ) for f in facts_raw if f.get("fact")
            ]

            # Step 3: Remap Sources 
            # We use the raw_results metadata as the "Source of Truth" for IDs and URLs
            # but allow the LLM to decide which ones are actually relevant.
            source_ids_used = {ref for f in verified_facts for ref in f.source_refs}
            sources = [
                r['metadata'] for r in raw_results 
                if r['metadata'].source_id in source_ids_used
            ]

            # If the LLM failed to provide sources, we default to all retrieved sources as a fallback
            if not sources and raw_results:
                sources = [r['metadata'] for r in raw_results]

            return PolicyResponse(
                domain=domain,
                verified_facts=verified_facts,
                sources=sources,
                uncertainty_notes=data.get("uncertainty_notes")
            )

        except Exception as e:
            # Step 4: Graceful Error Handling
            return PolicyResponse(
                domain=domain,
                verified_facts=[],
                sources=[r['metadata'] for r in raw_results],
                uncertainty_notes=f"Internal Policy Parsing Error: {str(e)}"
            )