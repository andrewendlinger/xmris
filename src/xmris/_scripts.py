import dataclasses
import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# IMPORT YOUR CONFIG CLASSES HERE
from xmris.visualization import (
    PlotHeatmapConfig,
    PlotQCGridConfig,
    PlotRidgeConfig,
    PlotTrajectoryConfig,
)

# Map configuration classes to their exact Quartodoc function anchors
CONFIG_MAP: dict[Any, dict[str, str]] = {
    PlotRidgeConfig: {
        "func_name": "plot_ridge",
        "anchor": "xmris.visualization.plot.plot_ridge",
    },
    PlotQCGridConfig: {
        "func_name": "plot_qc_grid",
        "anchor": "xmris.visualization.plot.plot_qc_grid",
    },
    PlotTrajectoryConfig: {
        "func_name": "plot_trajectory",
        "anchor": "xmris.visualization.plot.plot_trajectory",
    },
    PlotHeatmapConfig: {
        "func_name": "plot_heatmap",
        "anchor": "xmris.visualization.plot.plot_heatmap",
    },
}


def _get_docs_dir() -> Path:
    """
    Locate the root documentation directory relative to this script.

    Returns
    -------
    Path
        The absolute path to the 'docs' directory.

    Raises
    ------
    SystemExit
        If the 'docs' directory cannot be found at the expected location.
    """
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        print(f"❌ Error: 'docs' directory not found at: {docs_dir!s}")
        sys.exit(1)

    return docs_dir


def docs_config_classes(api_dir: Path) -> None:
    """
    Dynamically generate MyST Markdown for dataclass configs, grouped by metadata.

    Iterates through the mapped configuration classes, extracts field metadata,
    and constructs formatted Markdown tables for a condensed, readable layout.

    Parameters
    ----------
    api_dir : Path
        The directory where the generated API Markdown files should be saved.
    """
    print("🎨 Generating custom grouped docs for Config Dataclasses...")

    for cls, meta in CONFIG_MAP.items():
        docstring = (
            inspect.cleandoc(cls.__doc__) if cls.__doc__ else "Configuration object."
        )
        func_name = meta["func_name"]
        func_anchor = meta["anchor"]

        # Build the header and admonition block
        lines = [
            f"({cls.__name__})=",
            f"# {cls.__name__}",
            "",
            "```{admonition} Associated Plotting Function",
            ":class: seealso",
            f"This object configures the aesthetics of the **[`{func_name}()`](#{func_anchor})** function.",  # noqa: E501
            "```",
            "",
            docstring,
            "",
            "## Parameters",
            "",
        ]

        # Organize dataclass fields into groups based on their metadata
        groups: dict[str, list[dataclasses.Field]] = {}
        for f in dataclasses.fields(cls):
            group_name = f.metadata.get("group", "Other")
            groups.setdefault(group_name, []).append(f)

        # Append fields to the markdown output using Tables
        for group_name, f_list in groups.items():
            lines.extend(
                [
                    f"### {group_name}",
                    "",
                    "| Parameter | Type | Default | Description |",
                    "| :--- | :---: | :---: | :--- |",
                ]
            )

            for f in f_list:
                # 1. Clean description (remove newlines so it doesn't break the table row)
                raw_desc = f.metadata.get("description", "No description provided.")
                desc = raw_desc.replace("\n", " ")

                # 2. Extract and format the type string
                raw_type = getattr(f.type, "__name__", str(f.type).replace("typing.", ""))
                # CRITICAL: Escape pipe characters for Union types (str | None)
                # in Markdown tables
                safe_type = raw_type.replace("|", "&#124;")

                # 3. Determine the default value string
                if f.default is not dataclasses.MISSING:
                    default_str = f"`{f.default}`"
                elif f.default_factory is not dataclasses.MISSING:
                    default_str = f"`{f.default_factory()}`"
                else:
                    default_str = "*Required*"

                # 4. Add the formatted table row
                lines.append(
                    f"| **`{f.name}`** | *`{safe_type}`* | {default_str} | {desc} |"
                )

            # Add a blank line after the table for standard Markdown spacing
            lines.append("")

        # Write the resulting markdown to disk
        md_file = api_dir / f"{cls.__name__}.md"
        md_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"   ✅ Created grouped API page for {cls.__name__}")


def docs_api() -> None:
    """
    Generate the API documentation from Python source code.

    This function clears the old API reference directory, runs `quartodoc`
    to extract docstrings, and then converts the resulting Quarto Markdown
    (.qmd) files into MyST Markdown (.md) files. It also injects custom
    formatting for dataclass configurations.

    Raises
    ------
    SystemExit
        If the `quartodoc build` subprocess fails.
    """
    docs_dir = _get_docs_dir()
    api_dir = docs_dir / "api_reference"

    print("-" * 80)
    print("QUARTODOC".center(80, " "))
    print("-" * 80)
    print("🧹 Clearing old API reference files...")
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    print("🧬 Extracting Python docstrings via quartodoc...")
    try:
        # Run quartodoc from within the docs directory to find the config file
        subprocess.run(
            ["quartodoc", "build", "--config", "quartodoc.yml"], check=True, cwd=docs_dir
        )
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to build API documentation via quartodoc.")
        sys.exit(1)

    print("🔧 Translating Quarto Markdown to MyST Markdown...")

    for qmd_file in api_dir.rglob("*.qmd"):
        content = qmd_file.read_text(encoding="utf-8")

        # 1. Fix internal file extensions to point to standard Markdown
        content = content.replace(".qmd", ".md")

        # 2 & 3. Extract Quarto Anchors and clean up Quarto attribute blocks
        # Converts: ### Heading {.doc-method #anchor}  ->  (anchor)= \n ### Heading
        # And safely drops the Quarto-specific CSS classes in the process.
        content = re.sub(
            r"^(.*?)\s*\{[^\}]*?#([\w\.\-]+)[^\}]*\}\s*$",
            r"(\2)=\n\1",
            content,
            flags=re.MULTILINE,
        )

        # Strip any leftover pure CSS blocks that didn't have an ID (e.g., {.doc-signature})
        content = re.sub(r"\s*\{\.[^\}]+\}", "", content)

        # =====================================================================
        # NEW: DEAD LINK CLEANER
        # Quartodoc often lists inherited methods or properties in the summary
        # table but doesn't generate their body, resulting in dead markdown links.
        # =====================================================================

        # 1. Find all the valid MyST targets we just created in this file
        valid_targets = set(re.findall(r"^\(([\w\.\-]+)\)=", content, flags=re.MULTILINE))

        # 2. Define a replacer that checks if the link target actually exists
        def link_replacer(match):
            text = match.group(1)
            anchor = match.group(2)
            if anchor in valid_targets:
                return match.group(0)  # Target exists! Keep the full markdown link.
            else:
                # IT IS DANGEROUS TO SILENTLY HIDE THIS. Print a warning to the developer!
                print(
                    f"   ⚠️  Target missing for `#{anchor}`. Downgrading link to plain text."
                )
                return f"`{text}`"  # Downgrade to code text.

        # 3. Find all local markdown links [text](#anchor) and run them through the replacer
        content = re.sub(r"\[([^\]]+)\]\(#([\w\.\-]+)\)", link_replacer, content)
        # =====================================================================
        # 4. Auto-link Xarray type hints using MyST cross-reference syntax
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

        # 5. Clean up automatically generated dataclass references
        for cls in CONFIG_MAP.keys():
            cls_name = cls.__name__

            # Fix markdown table links to point to the custom generated files
            content = re.sub(
                rf"\[{cls_name}\]\(#.*?{cls_name}\)",
                rf"[{cls_name}]({cls_name}.md)",
                content,
            )

            # Delete the default Quartodoc section block for the dataclass
            content = re.sub(
                rf"\([^\)]*{cls_name}\)=\n### {cls_name}\n.*?(?=\n## |\Z)",
                "",
                content,
                flags=re.DOTALL,
            )

        # Save as a standard .md file and remove the old .qmd file
        md_file = qmd_file.with_suffix(".md")
        md_file.write_text(content, encoding="utf-8")
        qmd_file.unlink()

    # Inject our custom dataclass Markdown generator at the end
    docs_config_classes(api_dir)
    print("✅ API Markdown generation complete!")


def docs_notebooks() -> None:
    """
    Launch the MyST preview server to process and serve notebooks and Markdown.

    Raises
    ------
    SystemExit
        If the `myst start` subprocess fails abruptly.
    """
    docs_dir = _get_docs_dir()

    print("-" * 80)
    print("MYST".center(80, " "))
    print("-" * 80)
    print("🚀 Launching MyST preview server...")
    try:
        # Run myst from within the docs directory
        subprocess.run(["myst", "start"], check=True, cwd=docs_dir)
    except KeyboardInterrupt:
        print("\n👋 Preview server stopped.")
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to start the MyST server.")
        sys.exit(1)


def docs_all() -> None:
    """
    Execute the full documentation pipeline.

    Generates the API documentation first, then launches the local
    MyST preview server.
    """
    print("🔄 Running full documentation pipeline...")
    docs_api()
    docs_notebooks()
