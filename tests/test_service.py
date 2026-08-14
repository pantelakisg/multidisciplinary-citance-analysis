from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citan_ui.models.llama import parse_llama_output
from citan_ui.service import validate_citance


class CitanceValidationTests(unittest.TestCase):
    def test_accepts_exactly_one_target_marker(self) -> None:
        text = "We reuse the dataset released by <CIT>."
        self.assertEqual(validate_citance(text), text)

    def test_rejects_missing_target_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            validate_citance("We reuse the cited dataset.")

    def test_rejects_multiple_target_markers(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 2"):
            validate_citance("We compare <CIT> with <CIT>.")


class LlamaOutputParserTests(unittest.TestCase):
    def test_parses_case_insensitive_multiline_output(self) -> None:
        parsed = parse_llama_output(
            "semantics: methodology\nintent: reuse\npolarity: neutral"
        )
        self.assertEqual(
            parsed,
            {
                "semantics": "Methodology",
                "intent": "Reuse",
                "polarity": "Neutral",
            },
        )

    def test_rejects_unknown_labels(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unrecognized polarity"):
            parse_llama_output(
                "Semantics: Claim\nIntent: Generic\nPolarity: Positive"
            )


if __name__ == "__main__":
    unittest.main()
