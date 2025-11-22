from pydantic import BaseModel, Field, constr
from typing import Optional


class EvalRequest(BaseModel):
    criteria_description: constr(min_length=1, max_length=2000)
    applicant_cv: constr(min_length=1, max_length=20000)


class EvalResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Suitability score between 0 and 1")
    model_version: Optional[str] = None
    elapsed_seconds: float
