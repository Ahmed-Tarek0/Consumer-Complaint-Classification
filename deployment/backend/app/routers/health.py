from fastapi import APIRouter

from app.model_loader import model_bundle
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_bundle.is_loaded else "model_not_loaded",
        model_loaded=model_bundle.is_loaded,
        device=model_bundle.device,
    )
