"""
Script to build the FAISS index and metadata store.
Run this ONCE before starting your application.

Usage:
    python setup_data.py
"""

import os
import sys

# Add backend to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.rag.indexer import CivicIndexer

def main():
    print("Starting FAISS Index Build Process\n")
    
    # Create data directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialize indexer
    print("Loading embedding model...")
    indexer = CivicIndexer(model_name="all-MiniLM-L6-v2")
    
    # Process and index all sources
    print("\nProcessing civic sources...")
    indexer.process_and_index()
    
    # Save index and metadata
    print("\nSaving index and metadata...")
    index_path = "data/processed/civic_index.faiss"
    metadata_path = "data/processed/civic_index_metadata.json"
    indexer.save(index_path, metadata_path)

    print("\nIndex build complete!")
    print(f"   - Index: {index_path}")
    print(f"   - Metadata: {metadata_path}")
    print(f"   - Total chunks: {len(indexer.metadata_map)}")

if __name__ == "__main__":
    main()