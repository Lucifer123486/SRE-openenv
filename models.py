from pydantic import BaseModel
from typing import Literal, Dict

class SREAction(BaseModel):
    service: str
    command: Literal["restart", "scale_up", "rollback", "no_op"]

class SREObservation(BaseModel):
    logs: str
    metrics: Dict[str, Dict[str, float]]