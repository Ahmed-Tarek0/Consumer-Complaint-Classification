from fastapi import APIRouter, HTTPException

from app.inference import predict_complaint
from app.model_loader import model_bundle
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not model_bundle.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        return predict_complaint(request.text)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc
