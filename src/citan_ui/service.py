"""Application service coordinating validation and lazy model inference."""

from __future__ import annotations

import threading
import time
from typing import Any

from .config import Settings
from .domain import CITATION_MARKER
from .models.protocol import Predictor


VALID_MODEL_SELECTIONS = frozenset({"scibert", "llama", "both"})


def validate_citance(text: str, *, max_characters: int = 10_000) -> str:
    """Validate and normalize one user-supplied citance."""

    normalized = (text or "").strip()
    marker_count = normalized.count(CITATION_MARKER)
    if marker_count != 1:
        raise ValueError(
            f"Use exactly one {CITATION_MARKER} marker for the target citation "
            f"(found {marker_count}). Leave non-target citation markers unchanged."
        )
    if len(normalized) > max_characters:
        raise ValueError(
            "The input is too long; provide one citance sentence or a short context window."
        )
    return normalized


class AnalysisService:
    """Own and invoke model backends without loading them at server startup."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._predictors: dict[str, Predictor] = {}
        self._registry_lock = threading.Lock()

    def _predictor(self, name: str) -> Predictor:
        existing = self._predictors.get(name)
        if existing is not None:
            return existing

        with self._registry_lock:
            existing = self._predictors.get(name)
            if existing is not None:
                return existing

            if name == "llama":
                from .models.llama import LlamaPredictor

                predictor: Predictor = LlamaPredictor(self.settings)
            elif name == "scibert":
                from .models.scibert import SciBertPredictor

                predictor = SciBertPredictor(self.settings)
            else:
                raise ValueError(f"Unknown model: {name}")

            self._predictors[name] = predictor
            return predictor

    def health(self) -> dict[str, Any]:
        """Report server health without triggering model loading."""

        return {
            "ok": True,
            "llama_loaded": bool(
                self._predictors.get("llama")
                and self._predictors["llama"].is_loaded
            ),
            "scibert_loaded": bool(
                self._predictors.get("scibert")
                and self._predictors["scibert"].is_loaded
            ),
        }

    def analyze(self, citance: str, selection: str) -> dict[str, Any]:
        """Validate a citance and classify it with the selected backend(s)."""

        normalized = validate_citance(
            citance,
            max_characters=self.settings.max_citance_characters,
        )
        if selection not in VALID_MODEL_SELECTIONS:
            raise ValueError(
                f"Unknown model selection {selection!r}; expected llama, scibert, or both."
            )

        started = time.perf_counter()
        model_order = ("llama", "scibert") if selection == "both" else (selection,)
        predictions = [
            self._predictor(model_name).predict(normalized).to_dict()
            for model_name in model_order
        ]
        return {
            "citance": normalized,
            "predictions": predictions,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
