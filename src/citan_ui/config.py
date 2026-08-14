"""Application paths and runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable settings shared by the inference and Web layers."""

    llama_adapter_dir: Path = REPOSITORY_ROOT / "weights" / "llama-3.1-ft"
    scibert_weights_dir: Path = REPOSITORY_ROOT / "weights" / "scibert-mt"
    frontend_dir: Path = PACKAGE_DIR / "frontend"
    llama_base_model: str = os.environ.get(
        "CITAN_LLAMA_BASE",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    )
    max_citance_characters: int = 10_000
    scibert_max_tokens: int = 512
    llama_max_new_tokens: int = 64

    def validate_artifacts(self) -> None:
        """Fail early when the repository was cloned without its LFS objects."""

        required = (
            self.llama_adapter_dir / "adapter_model.safetensors",
            self.scibert_weights_dir / "model_weights.pth",
            self.frontend_dir / "index.html",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Required CitAn artifacts are missing: {rendered}. "
                "If this is a Git clone, run `git lfs pull`."
            )
