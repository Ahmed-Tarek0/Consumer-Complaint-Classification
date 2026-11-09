"""
App factory / entrypoint.

This file only wires things together — settings, the model's
lifecycle, CORS, and routers. Business logic lives in inference.py;
HTTP concerns live in routers/.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_loader import model_bundle
from app.routers import health, predict

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once when the server starts, not on the first request —
    # so the first real user never pays the model-loading latency.
    model_bundle.load()
    yield


app = FastAPI(
    title="Consumer Complaint Classification API",
    description=(
        "Classifies a consumer complaint narrative into a category "
        "using a fine-tuned transformer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
