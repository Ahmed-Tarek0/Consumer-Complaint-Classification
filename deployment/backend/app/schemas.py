"""
Request/response contracts for the API.

Keeping these separate from the inference logic means the API shape
can be reviewed (and the OpenAPI docs generated) without reading any
model code.
"""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The customer complaint narrative to classify.",
        examples=[
            "I was charged twice for the same transaction and the "
            "bank won't refund the duplicate charge."
        ],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be empty or whitespace only")
        return value.strip()


class CategoryScore(BaseModel):
    category: str
    probability: float


class PredictResponse(BaseModel):
    category: str
    confidence: float
    top_predictions: list[CategoryScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
