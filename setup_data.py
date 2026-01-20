import os
from backend.rag.indexer import CivicIndexer

def initialize_project():
    print("🚀 Initializing Civic Copilot Data...")
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Create a dummy policy file if none exists for testing
    dummy_path = "data/raw/passport_sla.txt"
    if not os.path.exists(dummy_path):
        with open(dummy_path, "w") as f:
            f.write("Passport processing time is 30 days. Expedited takes 7 days.")
        print(f"✅ Created dummy data at {dummy_path}")

    # Run the Indexer
    indexer = CivicIndexer()
    print("📂 Chunking and embedding documents...")
    indexer.process_and_index()
    
    # Save the index
    save_path = "data/processed/civic_index.faiss"
    indexer.save(save_path)
    print(f"📦 FAISS index saved to {save_path}")

if __name__ == "__main__":
    initialize_project()