"""Shared domain types and label definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CITATION_MARKER = "<CIT>"

SEMANTICS_LABELS = ("Claim", "Methodology", "Artifact", "Results")
INTENT_LABELS = ("Reuse", "Comparison", "Extension", "Generic")
POLARITY_LABELS = ("Supporting", "Neutral", "Refuting")

# These orders reproduce the integer encodings used to train SciBERT-MT.
SCIBERT_SEMANTICS_LABELS = ("Claim", "Results", "Methodology", "Artifact")
SCIBERT_INTENT_LABELS = ("Reuse", "Extension", "Comparison", "Generic")


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model's classification of a citance."""

    model: str
    semantics: str
    intent: str
    polarity: str
    confidence: dict[str, float] | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary without absent optional fields."""

        return {key: value for key, value in asdict(self).items() if value is not None}
