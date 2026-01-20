from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class CivicSource(BaseModel):
    id: str
    title: str
    url: Optional[HttpUrl] = None
    local_path: Optional[str] = None
    domain: str

# registry of official data sources for the PoC
CIVIC_SOURCES = [
    CivicSource(
        id="PASSPORT_SLA_2024",
        title="Passport Seva Service Level Agreement",
        url="https://www.passportindia.gov.in/",
        local_path="data/raw/passport_sla.pdf",
        domain="passport"
    ),
    CivicSource(
        id="VOTER_REG_GUIDE",
        title="Voter Registration Handbook",
        local_path="data/raw/voter_guide.txt",
        domain="voter_id"
    )
]