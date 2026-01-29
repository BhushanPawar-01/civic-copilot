import json
import os
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any

class CivicSource(BaseModel):
    id: str
    title: str
    url: Optional[HttpUrl] = None
    local_path: Optional[str] = None
    domain: str

def load_sources_from_json(json_path: str) -> List[CivicSource]:
    """Loads the source registry from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Source mapping not found at {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        return [CivicSource(**item) for item in data]

# Registry of official data sources loaded from the mapping file
MAPPING_FILE = os.path.join("data", "raw", "passport", "sources.json")
CIVIC_SOURCES = load_sources_from_json(MAPPING_FILE)

# Fallback for other domains if needed