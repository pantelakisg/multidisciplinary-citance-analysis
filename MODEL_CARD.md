# CitAn model card

## Task

Given a scientific citance containing one target marker (`<CIT>`), predict:

- **Semantics:** Claim, Methodology, Artifact, or Results;
- **Intent:** Reuse, Comparison, Extension, or Generic;
- **Polarity:** Supporting, Neutral, or Refuting.

## Models

### Llama 3.1-FT

- Base: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- Adaptation: PEFT LoRA supervised fine-tuning
- Rank: 4
- Alpha: 16
- Task tokenizer: one added `<CIT>` token
- Distributed artifact: adapter weights and tokenizer only; base weights are not included

### SciBERT-MT

- Base: `allenai/scibert_scivocab_cased`
- Architecture: shared 128-dimensional MLP followed by three task-specific MLPs and classification heads
- Target representation: hidden state at the internally substituted `[MASK]` token
- Distributed artifact: complete fine-tuned state dictionary plus the minimal SciBERT config and vocabulary

## Paper-reported test performance

| Model | Semantics macro F1 | Intent macro F1 | Polarity macro F1 |
|---|---:|---:|---:|
| Llama 3.1-FT | 80.66 | 74.55 | 63.89 |
| SciBERT-MT | 77.39 | 79.41 | 58.59 |

The reported test split contains 244 citances. Refer to the paper for data collection, annotation agreement, per-class metrics, and confidence intervals.
