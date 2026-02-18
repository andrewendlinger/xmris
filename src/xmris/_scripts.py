# myproject/scripts.py
import os
import subprocess
from pathlib import Path


def docs_main():
    """Run `jupyter book start` from the project's docs/ directory."""
    # Compute project root (adjust if needed)
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"
    if not docs_dir.exists():
        raise SystemExit(f"docs directory not found: {docs_dir!s}")

    # Change into docs and run the jupyter book CLI
    os.chdir(docs_dir)
    # Use the `jupyter-book` (or `jupyter`) CLI available in the project env.
    # This mirrors what you ran manually: `uv run jupyter book start`
    subprocess.run(["jupyter", "book", "start"], check=True)
