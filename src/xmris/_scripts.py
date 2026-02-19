import os
import subprocess
from pathlib import Path


def docs_main():
    """Run MyST from the uv environment."""
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        raise SystemExit(f"Error: 'docs' directory not found at: {docs_dir!s}")

    os.chdir(docs_dir)
    print("🚀 Launching MyST preview...")

    try:
        # We drop npx. uv automatically exposes 'myst' on the PATH here.
        subprocess.run(["myst", "start"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Preview server stopped.")
