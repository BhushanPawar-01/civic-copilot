import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .sources import CIVIC_SOURCES

class CivicIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_map: Dict[int, Dict] = {}

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple sliding window chunking for the PoC."""
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - 50)]

    def process_and_index(self):
        """Iterates through sources, embeds content, and builds the FAISS index."""
        current_id = 0
        
        for source in CIVIC_SOURCES:
            if not source.local_path or not os.path.exists(source.local_path):
                continue
                
            # Basic text loader (extend this for PDF parsing in production)
            with open(source.local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self._chunk_text(content)
            embeddings = self.encoder.encode(chunks)
            
            # Add to FAISS
            self.index.add(np.array(embeddings).astype("float32"))
            
            # Map index IDs back to metadata for retrieval
            for i, chunk in enumerate(chunks):
                self.metadata_map[current_id] = {
                    "id": source.id,
                    "title": source.title,
                    "url": str(source.url) if source.url else None,
                    "text": chunk,
                    "domain": source.domain
                }
                current_id += 1

    def save(self, index_path: str):
        faiss.write_index(self.index, index_path)
        # Note: In a full build, you'd also pickle the metadata_map here.