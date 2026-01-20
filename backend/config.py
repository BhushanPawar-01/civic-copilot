from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Civic Copilot"
    DEBUG: bool = True
    
    # API Keys (Loaded from .env)
    HF_TOKEN: str = Field(..., env="HF_TOKEN")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "../data")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "processed/civic_index.faiss")
    
    # Model Settings
    DEFAULT_LLM_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=os.path.join(BASE_DIR, "..", ".env"), extra="ignore")

settings = Settings()

# If the FAISS index path was provided as a relative path (for example from .env),
# resolve it relative to the project root so faiss can open the file from the
# backend working directory.
if not os.path.isabs(settings.FAISS_INDEX_PATH):
    settings.FAISS_INDEX_PATH = os.path.normpath(
        os.path.join(settings.BASE_DIR, "..", settings.FAISS_INDEX_PATH)
    )