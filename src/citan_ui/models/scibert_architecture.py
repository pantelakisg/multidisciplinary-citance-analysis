"""PyTorch architecture used by the CitAn SciBERT multi-task classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SciBertMultiTaskClassifier(nn.Module):
    """Shared SciBERT encoder with one MLP classification head per task.

    The module and parameter names intentionally match the training checkpoint.
    The target citation is represented by the contextual embedding at ``[MASK]``.
    """

    def __init__(
        self,
        bert_model: nn.Module,
        *,
        mask_token_id: int,
        hidden_dim: int = 128,
        num_semantics_classes: int = 4,
        num_intent_classes: int = 4,
        num_polarity_classes: int = 3,
    ) -> None:
        super().__init__()
        self.bert = bert_model
        self.mask_token_id = mask_token_id

        self.shared_mlp = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()

        self.semantics_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.intent_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.polarity_mlp = nn.Linear(hidden_dim, hidden_dim)

        self.semantics_head = nn.Linear(hidden_dim, num_semantics_classes)
        self.intent_head = nn.Linear(hidden_dim, num_intent_classes)
        self.polarity_head = nn.Linear(hidden_dim, num_polarity_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_matches = input_ids.eq(self.mask_token_id)
        mask_count = mask_matches.sum(dim=1)
        if not torch.all(mask_count == 1):
            raise ValueError("Each SciBERT input must contain exactly one [MASK] token.")

        encoded = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        mask_positions = mask_matches.to(torch.int64).argmax(dim=1)
        batch_positions = torch.arange(input_ids.size(0), device=input_ids.device)
        target_embedding = encoded.last_hidden_state[
            batch_positions,
            mask_positions,
            :,
        ]

        shared = self.shared_mlp(target_embedding)
        return {
            "semantics_logits": self.semantics_head(
                self.relu(self.semantics_mlp(shared))
            ),
            "intent_logits": self.intent_head(self.relu(self.intent_mlp(shared))),
            "polarity_logits": self.polarity_head(
                self.relu(self.polarity_mlp(shared))
            ),
        }
