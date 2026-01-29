import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from schemas.policy import SourceMetadata

class CivicRetriever:
    """Production-grade RAG retriever using FAISS and local embeddings."""

    def __init__(self, index_path: str, metadata_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        # Load a local lightweight model for embeddings
        self.encoder = SentenceTransformer(model_name)
        
        # Load the pre-built FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata map
        if metadata_path is None:
            metadata_path = index_path.replace('.faiss', '_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
            # Ensure keys are integers (robust to string keys)
            self.metadata_store: Dict[int, Dict[str, Any]] = {int(k): v for k, v in loaded_metadata.items()}
        print(f"Loaded metadata for {len(self.metadata_store)} chunks")

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
            if idx == -1: 
                continue  # No match found
            
            # Fetch metadata for the matched chunk
            meta = self.metadata_store.get(int(idx), None)
            
            if meta is None:
                print(f"No metadata found for index {idx}")
                continue
            
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
        
        print(f"Retrieved {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results

    def add_documents(self, texts: List[str], metadata: List[Dict]):
        """Helper to populate the local index for the PoC."""
        embeddings = self.encoder.encode(texts)
        start_idx = self.index.ntotal
        self.index.add(np.array(embeddings).astype("float32"))
        
        for i, meta in enumerate(metadata):
            self.metadata_store[start_idx + i] = meta