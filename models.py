from pydantic import BaseModel
from typing import Dict, Any

class SREAction(BaseModel):
    service: str
    command: str

class SREObservation(BaseModel):
    state: Dict[str, Any]  # The validator usually looks for 'state'
    logs: str
