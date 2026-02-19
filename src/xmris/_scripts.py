import os
import re
import subprocess
from pathlib import Path


def _get_docs_dir() -> Path:
    """Find the docs directory relative to this script."""
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        raise SystemExit(f"❌ Error: 'docs' directory not found at: {docs_dir!s}")

    return docs_dir


def docs_api():
    """Only generate the API documentation (Markdown) from Python source."""
    docs_dir = _get_docs_dir()
    os.chdir(docs_dir)

    print("🧬 Extracting Python docstrings via quartodoc...")
    try:
        subprocess.run(["quartodoc", "build", "--config", "quartodoc.yml"], check=True)

        print("🔧 Translating Quarto Markdown to MyST Markdown...")
        api_dir = docs_dir / "api_reference"

        for qmd_file in api_dir.rglob("*.qmd"):
            content = qmd_file.read_text(encoding="utf-8")

            # 1. Fix internal file extensions
            content = content.replace(".qmd", ".md")

            # 2. Scrub Quarto CSS classes
            content = re.sub(r"\{\.doc.*?\}", "", content)

            # 3. Translate Quarto Anchors to MyST Targets
            content = re.sub(
                r"^(#+)\s+(.*?)\s*\{\s*#([\w\.\-]+)[^\}]*\}",
                r"(\3)=\n\1 \2",
                content,
                flags=re.MULTILINE,
            )

            # 👇 NEW: 4. Auto-link Xarray type hints using MyST xref syntax
            content = re.sub(
                r"\b(xr\.DataArray|xarray\.DataArray)\b",
                r"[\1](xref:xarray#xarray.DataArray)",
                content,
            )
            content = re.sub(
                r"\b(xr\.Dataset|xarray\.Dataset)\b",
                r"[\1](xref:xarray#xarray.Dataset)",
                content,
            )

            md_file = qmd_file.with_suffix(".md")
            md_file.write_text(content, encoding="utf-8")
            qmd_file.unlink()

        print("✅ API Markdown generation complete!")
    except subprocess.CalledProcessError:
        raise SystemExit("❌ Error: Failed to build API documentation.")


def docs_notebooks():
    """Only launch the MyST preview server (processes notebooks and Markdown)."""
    docs_dir = _get_docs_dir()
    os.chdir(docs_dir)

    print("🚀 Launching MyST preview server...")
    try:
        subprocess.run(["myst", "start"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Preview server stopped.")


def docs_all():
    """Run the full pipeline: generate API docs, then launch the preview."""
    print("🔄 Running full documentation pipeline...")
    # These functions handle their own chdir and error handling seamlessly
    docs_api()
    docs_notebooks()
