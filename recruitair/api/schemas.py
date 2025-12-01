from typing import Optional

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    criteria_description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Job criteria description",
        examples=["Experience with Python", "Knowledge of ML algorithms."],
    )
    applicant_cv: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Applicant's CV text",
        examples=["Skilled in Python and data analysis."],
    )


class EvalResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Suitability score between 0 and 1")
    model_version: Optional[str] = Field(None, description="Version of the model used for evaluation")
    elapsed_seconds: float = Field(..., ge=0.0, description="Time taken to process the request in seconds")
