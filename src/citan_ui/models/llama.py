"""Loading and inference for the CitAn Llama 3.1 LoRA adapter."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from ..config import Settings
from ..domain import (
    INTENT_LABELS,
    POLARITY_LABELS,
    SEMANTICS_LABELS,
    Prediction,
)


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """### Instructions:
You are a citance analysis annotator. Your task is to analyze sentences from scientific publications containing the token <CIT>, which is the citance mark that refers to a specific cited work.
Focus only on the part of the sentence directly associated with <CIT>.
Ignore other citance marks unless they directly affect the interpretation of <CIT>.
Do not focus on keywords as they may provide bias on the category, but focus on why the author cites <CIT>, how the author feels towards <CIT>, and how the author utilizes the work <CIT>.
If there are many citance marks, ignore the meaning towards the other marks and focus only on the words that refer to <CIT>.

### Classification Categories:
For each sentence, select one option from each of the following categories based on the portion related to <CIT>:
1. Semantics: What is being cited?
Claim: Specific statements, theories, or ideas. This is a very high overview of <CIT> work or a theoretic concept.
Methodology: Procedures, techniques, or methods used. This is a high level overview and no specific or distinct mentions to <CIT> methodology.
Artifact: Tools, datasets, software, or other resources. This is a more detailed level, and there are specific mentions to <CIT> methodology with explicit detail.
Results: Findings, data analyses, or conclusions from <CIT>.
Choose the category that best represents the primary focus of the citance.
2. Intent: Why is the author citing it?
Reuse: Direct use of elements from the cited work.
Comparison: Highlighting similarities or differences between the author's current work and <CIT>.
Extension: Building upon and enhancing <CIT> work.
Generic: Providing general information from <CIT>.
Select the option that most closely reflects the relationship between the author's work and the cited work.
3. Polarity: What is the author's sentiment?
Supporting: Positive agreement or endorsement towards <CIT>. Can also be expressed through technical acknowledgment (state of the art, groundbreaking etc.).
Neutral: No clear stance from the author towards <CIT>.
Refuting: Disagreement or challenge towards <CIT>. Can also be expressed through technical critique.

Output format:
Semantics: <Claim|Methodology|Artifact|Results>
Intent: <Reuse|Comparison|Extension|Generic>
Polarity: <Supporting|Neutral|Refuting>"""


def parse_llama_output(text: str) -> dict[str, str]:
    """Parse the constrained three-label response produced by Llama."""

    choices = {
        "semantics": SEMANTICS_LABELS,
        "intent": INTENT_LABELS,
        "polarity": POLARITY_LABELS,
    }
    parsed: dict[str, str] = {}
    for field, labels in choices.items():
        label_pattern = "|".join(map(re.escape, labels))
        match = re.search(
            rf"\b{field}\s*:\s*({label_pattern})\b",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise RuntimeError(
                f"Llama returned an unrecognized {field} label: {text.strip()}"
            )
        parsed[field] = next(
            label for label in labels if label.lower() == match.group(1).lower()
        )
    return parsed


class LlamaPredictor:
    """Thread-safe inference backend with an active CitAn LoRA adapter."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any]:
        if self.is_loaded:
            return self._model, self._tokenizer

        with self._lock:
            if self.is_loaded:
                return self._model, self._tokenizer

            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Llama 3.1-FT requires a CUDA-capable GPU for the supplied "
                    "4-bit base model. SciBERT-MT can run on CPU."
                )

            adapter_dir = self._settings.llama_adapter_dir
            LOGGER.info("Loading Llama tokenizer and LoRA adapter from %s", adapter_dir)
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                self._settings.llama_base_model,
                device_map="cuda:0",
            )

            # The task tokenizer adds <CIT>; its saved embedding and lm_head rows
            # are part of the PEFT artifact, so the base vocabulary must be resized
            # before PEFT restores the adapter checkpoint.
            base_model.resize_token_embeddings(len(tokenizer))

            # This is the LoRA inference step. PEFT wraps the base model, loads the
            # trained adapter weights, and keeps the default adapter active for
            # every subsequent call to model.generate(). No training code is run.
            model = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                adapter_name="default",
                device_map="cuda:0",
                is_trainable=False,
            )
            model.set_adapter("default")
            model.eval()

            if "default" not in model.peft_config:
                raise RuntimeError("The CitAn LoRA adapter was not loaded by PEFT.")

            self._model = model
            self._tokenizer = tokenizer
            LOGGER.info("Llama 3.1-FT is ready with the CitAn LoRA adapter active")
            return model, tokenizer

    def predict(self, citance: str) -> Prediction:
        import torch

        model, tokenizer = self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": citance},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with self._lock, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self._settings.llama_max_new_tokens,
                do_sample=False,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        labels = parse_llama_output(generated)
        return Prediction(
            model="Llama 3.1-FT",
            semantics=labels["semantics"],
            intent=labels["intent"],
            polarity=labels["polarity"],
            raw_output=generated,
        )
