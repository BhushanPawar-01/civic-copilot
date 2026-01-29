import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes import router as api_router
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    # This runs once when the server starts
    print("Initializing Civic Copilot Agents...")
    
    # Check if FAISS index exists
    import os
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        print(f"WARNING: FAISS index not found at {settings.FAISS_INDEX_PATH}. "
              "Please run setup_data.py first.")
    
    yield  # The application runs while this is suspended

    # --- Shutdown Logic ---
    # This runs when the server stops
    print("Shutting down Civic Copilot...")

app = FastAPI(
    title="Civic Copilot API",
    lifespan=lifespan, 
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router)

# NOTE: Removed @app.on_event("startup") to avoid duplication and warnings.

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)