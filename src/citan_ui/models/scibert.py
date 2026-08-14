"""Loading and inference for the CitAn SciBERT-MT checkpoint."""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import Settings
from ..domain import (
    CITATION_MARKER,
    POLARITY_LABELS,
    SCIBERT_INTENT_LABELS,
    SCIBERT_SEMANTICS_LABELS,
    Prediction,
)


LOGGER = logging.getLogger(__name__)


class SciBertPredictor:
    """Thread-safe, lazy SciBERT-MT inference backend."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self.is_loaded:
            return self._model, self._tokenizer, self._device

        with self._lock:
            if self.is_loaded:
                return self._model, self._tokenizer, self._device

            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            from .scibert_architecture import SciBertMultiTaskClassifier

            weights_dir = self._settings.scibert_weights_dir
            LOGGER.info("Loading SciBERT-MT weights from %s", weights_dir)
            tokenizer = AutoTokenizer.from_pretrained(weights_dir, use_fast=True)
            if tokenizer.mask_token_id is None:
                raise RuntimeError("The SciBERT tokenizer does not define [MASK].")

            bert = AutoModel.from_config(AutoConfig.from_pretrained(weights_dir))
            model = SciBertMultiTaskClassifier(
                bert,
                mask_token_id=tokenizer.mask_token_id,
            )
            state = torch.load(
                weights_dir / "model_weights.pth",
                weights_only=True,
                map_location="cpu",
            )
            model.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()

            self._model = model
            self._tokenizer = tokenizer
            self._device = device
            LOGGER.info("SciBERT-MT is ready on %s", device)
            return model, tokenizer, device

    def predict(self, citance: str) -> Prediction:
        import torch

        model, tokenizer, device = self._load()
        masked_citance = citance.replace(CITATION_MARKER, tokenizer.mask_token, 1)
        inputs = tokenizer(
            masked_citance,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.scibert_max_tokens,
        ).to(device)

        with self._lock, torch.inference_mode():
            logits = model(**inputs)

        label_encodings = {
            "semantics": SCIBERT_SEMANTICS_LABELS,
            "intent": SCIBERT_INTENT_LABELS,
            "polarity": POLARITY_LABELS,
        }
        labels: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for task, label_order in label_encodings.items():
            probabilities = torch.softmax(logits[f"{task}_logits"], dim=-1)[0]
            predicted_index = int(probabilities.argmax().item())
            labels[task] = label_order[predicted_index]
            confidence[task] = round(float(probabilities[predicted_index].item()), 4)

        return Prediction(
            model="SciBERT-MT",
            semantics=labels["semantics"],
            intent=labels["intent"],
            polarity=labels["polarity"],
            confidence=confidence,
        )
