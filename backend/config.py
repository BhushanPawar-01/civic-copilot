from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Civic Copilot"
    DEBUG: bool = False  # Set to False for production
    ENVIRONMENT: str = "production"
    
    # API Keys (Optional for HuggingFace Spaces - can be set via Secrets)
    HF_TOKEN: Optional[str] = Field(default=None, env="HF_TOKEN")
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "../data")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "processed/civic_index.faiss")
    
    # Model Settings
    DEFAULT_LLM_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # LLM Generation defaults
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_TOKENS: int = 1024
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, "..", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

# If the FAISS index path was provided as a relative path (for example from .env),
# resolve it relative to the project root so faiss can open the file from the
# backend working directory.
if not os.path.isabs(settings.FAISS_INDEX_PATH):
    settings.FAISS_INDEX_PATH = os.path.normpath(
        os.path.join(settings.BASE_DIR, "..", settings.FAISS_INDEX_PATH)
    )

# Validation helper functions
def validate_hf_token():
    """Check if HF token is available for HuggingFace client."""
    if not settings.HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not found. Please set it in Hugging Face Space Secrets "
            "or in your .env file for local development."
        )
    return settings.HF_TOKEN

def validate_gemini_key():
    """Check if Gemini API key is available."""
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in Hugging Face Space Secrets "
            "or in your .env file for local development."
        )
    return settings.GEMINI_API_KEY