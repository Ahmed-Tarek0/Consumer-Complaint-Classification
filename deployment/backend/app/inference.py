"""
Turns raw text into a PredictResponse.

Deliberately has no knowledge of FastAPI, HTTP status codes, or
routing — it takes a string and returns a typed result, which makes it
trivial to unit test or reuse from a CLI / batch script later.
"""

import torch

from app.config import settings
from app.model_loader import model_bundle
from app.schemas import CategoryScore, PredictResponse


@torch.no_grad()
def predict_complaint(text: str) -> PredictResponse:
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=settings.MAX_LENGTH,
        return_tensors="pt",
    ).to(model_bundle.device)

    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(settings.TOP_K, probabilities.shape[0])
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions = [
        CategoryScore(
            category=model_bundle.labels[index.item()],
            probability=round(prob.item(), 4),
        )
        for prob, index in zip(top_probs, top_indices)
    ]

    return PredictResponse(
        category=top_predictions[0].category,
        confidence=top_predictions[0].probability,
        top_predictions=top_predictions,
    )
