import os
import shutil
import subprocess
from pathlib import Path


def docs_main():
    """Run MyST via npx to ensure it works even if not installed globally."""
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        raise SystemExit(f"Error: 'docs' directory not found at: {docs_dir!s}")

    # Check for npm instead of myst
    if not shutil.which("npm"):
        raise SystemExit(
            "Error: Node.js/npm not found. MyST requires Node.js.\n"
            "Install it from https://nodejs.org/"
        )

    os.chdir(docs_dir)
    print("🚀 Launching MyST preview via npx...")

    try:
        # 'npx myst' will find it locally, globally, or fetch it on the fly
        subprocess.run(["npx", "-p", "mystmd", "myst", "start"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Preview server stopped.")
