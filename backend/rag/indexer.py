import faiss
import numpy as np
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime
import fitz  # PyMuPDF
from markdownify import markdownify as md
from typing import List, Dict
from .sources import CIVIC_SOURCES

class CivicIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_map: Dict[int, Dict] = {}
        self.build_timestamp = None

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple sliding window chunking for the PoC."""
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - 50)]

    def _extract_text(self, file_path: str) -> str:
        """Extracts text from .txt or .pdf files."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == ".html":
            if not md:
                raise ImportError("markdownify is required for HTML processing. Run 'pip install markdownify'.")
            with open(file_path, 'r', encoding='utf-8') as f:
                return md(f.read(), strip=['script', 'style', 'nav', 'footer', 'header'])
        elif ext == ".pdf":
            if not fitz:
                raise ImportError("PyMuPDF (fitz) is required for PDF processing. Run 'pip install pymupdf'.")
            doc = fitz.open(file_path)
            return "\n".join([page.get_text() for page in doc])
        return ""

    def process_and_index(self):
        """Iterates through sources, embeds content, and builds the FAISS index."""
        current_id = 0
        self.build_timestamp = datetime.now().isoformat()
        
        for source in CIVIC_SOURCES:
            if not source.local_path or not os.path.exists(source.local_path):
                print(f"Skipping {source.id}: File not found at {source.local_path}")
                continue
                
            try:
                print(f"Reading {source.id} ({source.local_path})...")
                content = self._extract_text(source.local_path)
            except Exception as e:
                print(f"Error reading {source.local_path}: {e}")
                continue
            
            chunks = self._chunk_text(content)
            
            if not chunks:
                print(f"No chunks generated for {source.id}")
                continue
            
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
                    "domain": source.domain,
                    "indexed_at": self.build_timestamp
                }
                current_id += 1
            
            print(f"Indexed {len(chunks)} chunks from {source.id}")
        
        print(f"\nTotal indexed chunks: {len(self.metadata_map)}")

    def save(self, index_path: str, metadata_path: str = None):
        """Save both FAISS index and metadata map."""
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to {index_path}")
        
        # Save metadata map as JSON
        if metadata_path is None:
            metadata_path = index_path.replace('.faiss', '_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata_map, f, indent=4)
        print(f"Metadata saved to {metadata_path}")