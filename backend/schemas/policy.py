from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class SourceMetadata(BaseModel):
    """Metadata for a single source chunk retrieved via RAG."""
    source_id: str = Field(..., description="Unique identifier for the document or portal entry")
    title: str = Field(..., description="Title of the official document or webpage")
    url: Optional[HttpUrl] = Field(None, description="Direct link to the official government source")
    page_number: Optional[int] = Field(None, description="Page number if the source is a PDF")
    last_updated: Optional[str] = Field(None, description="Date the information was last verified or published")

class PolicyFact(BaseModel):
    """A specific factual claim retrieved from official documents."""
    fact: str = Field(..., description="The verified rule, timeline, or requirement")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="How relevant this fact is to the user query")
    source_refs: List[str] = Field(..., description="List of source_ids supporting this specific fact")

class PolicyResponse(BaseModel):
    """The structured output schema for the Policy Knowledge Agent."""
    domain: str = Field(..., description="The civic domain identified (e.g., 'passport_services')")
    verified_facts: List[PolicyFact] = Field(..., description="List of grounded facts retrieved from the RAG layer")
    sources: List[SourceMetadata] = Field(..., description="Detailed metadata for all cited sources")
    uncertainty_notes: Optional[str] = Field(
        None, 
        description="Internal notes on gaps in information to be handled by the Verification Agent"
    )

    class Config:
        """Pydantic configuration for cleaner schema generation."""
        json_schema_extra = {
            "example": {
                "domain": "passport_services",
                "verified_facts": [
                    {
                        "fact": "Standard passport processing takes 30-45 days.",
                        "relevance_score": 0.95,
                        "source_refs": ["DOC_001"]
                    }
                ],
                "sources": [
                    {
                        "source_id": "DOC_001",
                        "title": "Passport Seva Service Level Agreement",
                        "url": "https://www.passportindia.gov.in/",
                        "last_updated": "2024-01-10"
                    }
                ]
            }
        }