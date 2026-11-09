"""
Owns the model's lifecycle.

Loading a transformer is expensive (disk + GPU/CPU memory), so it must
happen exactly once, at process startup, and be reused by every
request. This module is the single place that does that loading;
nothing else imports torch or transformers directly.

The model, tokenizer and labels are pulled from the Hugging Face Hub
(see settings.MODEL_DIR), not from a local folder, so the repo never
needs to ship the ~250MB weight file.
"""

import json
import logging

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings

logger = logging.getLogger("complaint_classifier")


class ModelBundle:
    """Holds the tokenizer, model and label list once loaded."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.labels: list[str] = []
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        model_source = settings.MODEL_DIR  # e.g. "username/complaint-classifier"

        logger.info("Loading model from %s onto %s ...", model_source, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()

        labels_path = hf_hub_download(repo_id=model_source, filename="labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self._loaded = True
        logger.info("Model ready (%d categories).", len(self.labels))

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# One instance, shared by the whole app.
model_bundle = ModelBundle()