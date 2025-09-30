from typing import List

from pydantic import BaseModel, Field


class KeyCriterion(BaseModel):
    title: str = Field(description="The title of the key requirement or criterion in one or two words at most.")
    description: str = Field(description="A brief but exhaustive description of the key requirement.")
    importance: int = Field(
        ge=1,
        le=5,
        description="An integer from 1 to 5 indicating the importance of "
        "an applicant to this job offer to fulfill this key requirement.",
    )


class KeyCriteriaResponse(BaseModel):
    key_criteria: List[KeyCriterion] = Field(
        description="An exhaustive list of all the key requirements or criteria that are explicit or implicit in the job offer."
    )
