# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nbformat",
# ]
# ///

from pathlib import Path

import nbformat


def clean_notebooks():
    # Target the notebooks directory based on your warnings
    notebook_dir = Path("./docs/notebooks")
    keys_to_remove = ["default_lexer", "notebook_metadata_filter"]

    # Track if we actually did anything
    files_cleaned = 0

    print(f"🔍 Scanning {notebook_dir} for leftover Jupytext metadata...")

    for nb_path in notebook_dir.rglob("*.ipynb"):
        try:
            nb = nbformat.read(nb_path, as_version=4)
            jupytext_meta = nb.metadata.get("jupytext", {})

            modified = False
            for key in keys_to_remove:
                if key in jupytext_meta:
                    del jupytext_meta[key]
                    modified = True

            # If we emptied the jupytext dictionary completely, just delete it
            if modified and not jupytext_meta:
                del nb.metadata["jupytext"]

            if modified:
                nbformat.write(nb, nb_path)
                print(f"✅ Cleaned: {nb_path}")
                files_cleaned += 1

        except Exception as e:
            print(f"❌ Error processing {nb_path}: {e}")

    if files_cleaned == 0:
        print("✨ No files needed cleaning!")
    else:
        print(f"🎉 Successfully cleaned {files_cleaned} notebooks.")


if __name__ == "__main__":
    clean_notebooks()
