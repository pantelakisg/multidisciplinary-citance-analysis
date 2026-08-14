"""Development entry point for running the CitAn Web UI from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from citan_ui.main import main  # noqa: E402  (src path is configured above)


if __name__ == "__main__":
    main()
