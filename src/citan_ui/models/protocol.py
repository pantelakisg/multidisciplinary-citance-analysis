"""Interface implemented by every CitAn inference backend."""

from __future__ import annotations

from typing import Protocol

from ..domain import Prediction


class Predictor(Protocol):
    """A lazily loaded model capable of classifying one citance."""

    @property
    def is_loaded(self) -> bool: ...

    def predict(self, citance: str) -> Prediction: ...
