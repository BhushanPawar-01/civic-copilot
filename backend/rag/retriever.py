import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from schemas.policy import SourceMetadata

class CivicRetriever:
    """Production-grade RAG retriever using FAISS and local embeddings."""

    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        # Load a local lightweight model for embeddings
        self.encoder = SentenceTransformer(model_name)
        
        # Load the pre-built FAISS index
        self.index = faiss.read_index(index_path)
        
        # Mock mapping of index IDs to actual source metadata
        # In production, this would be a lookup in a metadata DB or JSON file
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    async def retrieve(self, query: str, top_k: int = 5, domain: str = None) -> List[Dict[str, Any]]:
        """
        Performs vector search to find relevant civic policy chunks.
        """
        # 1. Embed the user query
        query_vector = self.encoder.encode([query])
        
        # 2. Search the FAISS index
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # No match found
            
            # Fetch metadata for the matched chunk
            meta = self.metadata_store.get(idx, {})
            
            # Filter by domain if the Intent Agent provided one
            if domain and meta.get("domain") != domain:
                continue
                
            results.append({
                "content": meta.get("text", "Content not found"),
                "score": float(distances[0][i]),
                "metadata": SourceMetadata(
                    source_id=meta.get("id", f"SRC_{idx}"),
                    title=meta.get("title", "Official Policy Document"),
                    url=meta.get("url"),
                    page_number=meta.get("page")
                )
            })
            
        return results

    def add_documents(self, texts: List[str], metadata: List[Dict]):
        """Helper to populate the local index for the PoC."""
        embeddings = self.encoder.encode(texts)
        start_idx = self.index.ntotal
        self.index.add(np.array(embeddings).astype("float32"))
        
        for i, meta in enumerate(metadata):
            self.metadata_store[start_idx + i] = meta