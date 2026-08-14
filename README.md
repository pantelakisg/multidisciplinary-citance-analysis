# Towards Multidisciplinary Citance Analysis

This repository contains the dataset, trained model artifacts, and local
inference interface for the paper *Towards Multidisciplinary Citance Analysis:
A Novel Approach for Intent, Polarity, and Semantics Classification*.
The paper was accepted to the main conference of
[EMNLP 2026](https://2026.emnlp.org/) in Budapest, Hungary.

The two released classifiers predict three properties of a scientific
citance:

- semantics: Claim, Methodology, Artifact, or Results;
- intent: Reuse, Comparison, Extension, or Generic; and
- polarity: Supporting, Neutral, or Refuting.

| Model | Included artifact | Runtime |
|---|---|---|
| Llama 3.1-FT | PEFT/LoRA adapter and task tokenizer | NVIDIA CUDA GPU |
| SciBERT-MT | Complete multi-task model state dictionary | CPU or GPU |

## Dataset

[`data/CitAn.xlsx`](data/CitAn.xlsx) contains the 1,165 annotated citances.
The fixed train, development, and test partitions are in
[`data/CitAn_splits.xlsx`](data/CitAn_splits.xlsx).

| Dimension | Label distribution |
|---|---|
| Semantics | Methodology: 361; Artifact: 346; Claim: 259; Results: 199 |
| Intent | Generic: 603; Reuse: 237; Comparison: 225; Extension: 100 |
| Polarity | Neutral: 826; Supporting: 183; Refuting: 156 |

The dataset covers Health Sciences and Bioinformatics (456 citances), Computer
Science (430), and Social Sciences and Humanities (279). The `synthetic` column
marks 1,040 original (`N`) and 125 synthetic (`Y`) instances.

The main columns are the citance text and target marker
(`Citance Text_GOLD`, `Citation Mark_GOLD`), the three gold labels, source
identifiers, and disciplinary metadata. To classify a row in the interface,
replace its target citation marker with `<CIT>` and leave other citation markers
unchanged.

## Installation

The model files are stored with [Git LFS](https://git-lfs.com/). This project
uses [uv](https://docs.astral.sh/uv/) for a reproducible Python 3.11 environment.

```console
git lfs install
git clone https://github.com/pantelakisg/multidisciplinary-citance-analysis.git
cd multidisciplinary-citance-analysis
git lfs pull
uv sync --locked
uv run citan-ui
```

Open <http://127.0.0.1:7860> and use `Ctrl+C` to stop the server. The first
Llama request downloads the `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` base
model from Hugging Face; the SciBERT model is self-contained.

On Windows and Linux, uv installs the CUDA 13.0 PyTorch build from PyTorch's
official package index so both models are available. macOS uses the standard
CPU build and can run SciBERT-MT only.

`uv sync` is the recommended setup. It creates a local `.venv`, installs the
project in editable mode, and uses the committed `uv.lock`. A standard editable
install also works if uv is not available:

```console
python -m pip install -e .
python -m citan_ui
```

## Using the interface

Replace only the citation you want to classify with exactly one `<CIT>` marker:

```text
Unlike <CIT>, the method introduced by [27] does not require external supervision.
```

Choose SciBERT-MT, Llama 3.1-FT, or Compare both, then select **Analyze
citance**. SciBERT-MT also reports its maximum softmax score; this is an
uncalibrated model confidence indicator, not a probability guarantee.

Llama inference requires a CUDA-capable NVIDIA GPU. The application loads the
released LoRA adapter for every Llama prediction; it does not merge the adapter
or run training. SciBERT-MT can run on CPU, although its first prediction is
slower while the checkpoint loads.

Optional server settings are available through the entry point:

```console
uv run python app.py --host 127.0.0.1 --port 7860 --log-level INFO
```

Keep the default loopback host unless you have deliberately secured the server
for network access.

## Repository layout

```text
app.py                  Local development entry point
data/                   Annotated dataset and fixed splits
src/citan_ui/           Inference service and browser interface
tests/                  Dependency-free unit tests
weights/                Llama adapter and SciBERT checkpoint
MODEL_CARD.md           Model details, results, and limitations
MODEL_MANIFEST.json     Artifact sizes and SHA-256 checksums
LICENSE                 Consolidated license terms
NOTICE                  Required notices and attribution
```

## Development and integrity checks

```console
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The service loads both models lazily. `/api/health` reports their state without
allocating model memory.

The repository contains roughly 3 GB of weights. If a model fails to load after
cloning, confirm that Git LFS downloaded the actual files:

```console
git lfs pull
git lfs ls-files
```

Sizes and SHA-256 hashes for the released weights are recorded in
[`MODEL_MANIFEST.json`](MODEL_MANIFEST.json). Paper-reported test results and
known limitations are in [`MODEL_CARD.md`](MODEL_CARD.md).

## Citation

If you use the CitAn dataset or the released model weights and adapters, you
must cite:

> Georgios Pantelakis, Petros Stavropoulos, and Haris Papageorgiou. 2026.
> *Towards Multidisciplinary Citance Analysis: A Novel Approach for Intent,
> Polarity, and Semantics Classification.*

```bibtex
@misc{pantelakis2026multidisciplinary,
  title  = {Towards Multidisciplinary Citance Analysis: A Novel Approach for
            Intent, Polarity, and Semantics Classification},
  author = {Pantelakis, Georgios and Stavropoulos, Petros and
            Papageorgiou, Haris},
  year   = {2026},
  url    = {https://github.com/pantelakisg/multidisciplinary-citance-analysis}
}
```

## Licenses

The consolidated [`LICENSE`](LICENSE) file records the terms for each part of
the repository:

- original application source: MIT License;
- CitAn dataset annotations and project documentation: CC BY 4.0;
- SciBERT-derived components: Apache License 2.0; and
- Llama adapter and tokenizer artifacts: Llama 3.1 Community License and its
  Acceptable Use Policy.

Copyright in the original CitAn code, annotations, documentation, and
fine-tuning contributions remains with the paper's authors. Required third-party
attribution is collected in [`NOTICE`](NOTICE). Citation is required when using
the dataset or released model artifacts. Use of the software is governed by the
MIT License and does not require a paper citation.
