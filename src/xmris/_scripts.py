import dataclasses
import inspect
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# IMPORT YOUR CONFIG CLASSES HERE
from xmris.visualization import (
    CarpetConfig,
    PlotQCGridConfig,
    PlotTrajectoryConfig,
    WaterfallConfig,
)

# The docs chapters whose pages are executed as tests. One directory per sidebar
# chapter, mirrored into tests/autogen_notebooks/<chapter>/. Keep in sync with
# GENRES in .claude/skills/docs-page/check_docs.py -- that file decides which
# house rules a chapter's pages are held to, this one decides which get run.
TUTORIAL_CHAPTERS = ("basics", "pipeline", "fitting", "visualization", "vendor")
CONCEPTS_CHAPTER = "concepts"

# Map configuration classes to their exact Quartodoc function anchors
CONFIG_MAP: dict[Any, dict[str, str]] = {
    WaterfallConfig: {
        "func_name": "plot_waterfall",
        "anchor": "xmris.visualization.plot.plot_waterfall",
    },
    PlotQCGridConfig: {
        "func_name": "plot_qc_grid",
        "anchor": "xmris.visualization.plot.plot_qc_grid",
    },
    PlotTrajectoryConfig: {
        "func_name": "plot_trajectory",
        "anchor": "xmris.visualization.plot.plot_trajectory",
    },
    CarpetConfig: {
        "func_name": "plot_carpet",
        "anchor": "xmris.visualization.plot.plot_carpet",
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
        docstring = inspect.cleandoc(cls.__doc__) if cls.__doc__ else "Configuration object."
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
                lines.append(f"| **`{f.name}`** | *`{safe_type}`* | {default_str} | {desc} |")

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
    api_dir = docs_dir / "api"

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

        # Strip any leftover pure CSS blocks that didn't have an ID
        # (e.g., {.doc-signature})
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
                    f"   ⚠️  Target missing for `#{anchor}`. Downgrading link to plain text."  # noqa: E501
                )
                return f"`{text}`"  # Downgrade to code text.

        # 3. Find all local markdown links [text](#anchor) and replace them
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

    # Give the generated landing page a stable MyST target. It is the chapter
    # entry the sidebar, the header nav and the home-page map all link to, and
    # quartodoc emits a bare `# Function reference` with no anchor -- so without
    # this every link into the API chapter would resolve against a URL instead.
    index_file = api_dir / "index.md"
    if index_file.exists():
        index_text = index_file.read_text(encoding="utf-8")
        if not index_text.startswith("(api-home)="):
            index_file.write_text(f"(api-home)=\n{index_text.lstrip()}", encoding="utf-8")

    print("✅ API Markdown generation complete!")


def docs_notebooks() -> None:
    """
    Launch the MyST preview server, supervising it across transient crashes.

    ``myst start`` occasionally dies *after* serving happily for a while — an
    upstream node/myst fault, unrelated to our config. Rather than report that
    as a fatal "failed to start", relaunch it a few times. Only a *fast* exit,
    one that never really came up, is treated as a genuine start failure and
    fails fast so a real config error is not hidden behind an endless restart
    loop.

    Raises
    ------
    SystemExit
        If the server never comes up, or keeps crashing past the retry cap.
    """
    docs_dir = _get_docs_dir()

    print("-" * 80)
    print("MYST".center(80, " "))
    print("-" * 80)
    print("🚀 Launching MyST preview server...")

    max_restarts = 3
    # A run shorter than this never really served, so treat it as a start
    # failure (fail fast) rather than a crash worth relaunching.
    min_healthy_seconds = 15.0
    # Return codes that mean "the user stopped it", not "it crashed".
    # POSIX: -2/-15 are SIGINT/SIGTERM as negative codes; 130/143 the 128+signal shell
    # convention. Windows: a Ctrl-C surfaces STATUS_CONTROL_C_EXIT (0xC000013A) as
    # -1073741510 (signed) or 3221225786 (unsigned), depending on how it is reported.
    stop_codes = (-2, -15, 130, 143, -1073741510, 3221225786)

    restarts = 0
    while True:
        started = time.monotonic()
        try:
            # Run myst from within the docs directory
            subprocess.run(["myst", "start", "--execute"], check=True, cwd=docs_dir)
            return  # clean exit — myst was told to stop
        except KeyboardInterrupt:
            print("\n👋 Preview server stopped.")
            return
        except subprocess.CalledProcessError as exc:
            # Ctrl-C can surface here rather than as KeyboardInterrupt, depending
            # on how the signal reaches the child; don't relaunch a user stop.
            if exc.returncode in stop_codes:
                print("\n👋 Preview server stopped.")
                return

            uptime = time.monotonic() - started
            if uptime < min_healthy_seconds:
                print("❌ Error: Failed to start the MyST server.")
                sys.exit(1)

            restarts += 1
            if restarts > max_restarts:
                print(f"❌ MyST crashed {restarts} times — giving up.")
                sys.exit(1)

            print(
                f"⚠️  MyST crashed after {uptime:.0f}s (upstream node/myst fault) — "
                f"restarting ({restarts}/{max_restarts})..."
            )
            time.sleep(2.0)


def docs_all() -> None:
    """
    Execute the full documentation pipeline.

    Generates the API documentation first, then launches the local
    MyST preview server.
    """
    print("🔄 Running full documentation pipeline...")
    docs_api()
    docs_notebooks()


def _is_executable_page(md: Path) -> bool:
    """
    Report whether a MyST page carries the jupytext kernelspec nbmake needs.

    Parameters
    ----------
    md : Path
        The Markdown page to inspect.

    Returns
    -------
    bool
        True when the page opens with frontmatter declaring a ``kernelspec``.
    """
    text = md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return False
    frontmatter = text.partition("---")[2].partition("\n---")[0]
    return "kernelspec:" in frontmatter


def generate_test_notebooks() -> None:
    """
    Convert MyST Markdown notebooks to Jupyter Notebooks for testing.

    This function safely resolves directories relative to this script to avoid
    working-directory bugs. It also clears any previously generated test notebooks
    to prevent testing stale files.

    Returns nothing: it is wired directly to the ``test-gen`` console script, and a
    non-``None`` return would be handed to ``sys.exit`` and reported as a failure.

    Raises
    ------
    SystemExit
        If the source directory cannot be found or Jupytext fails during conversion.
    """
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"
    test_dir = project_root / "tests" / "autogen_notebooks"

    missing = [c for c in (*TUTORIAL_CHAPTERS, CONCEPTS_CHAPTER) if not (docs_dir / c).exists()]
    if missing:
        print(f"❌ Error: docs chapter directories not found: {', '.join(missing)}")
        sys.exit(1)

    print(f"🧹 Clearing old test notebooks in '{test_dir.name}'...")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    # The tutorial chapters are the test suite: take every page, so one that lost
    # its kernelspec fails loud in nbmake rather than silently dropping out. Each
    # chapter keeps its own subtree, so test paths read
    # autogen_notebooks/<chapter>/<name>.ipynb.
    md_files = [
        (md, Path(chapter) / md.relative_to(docs_dir / chapter))
        for chapter in TUTORIAL_CHAPTERS
        for md in (docs_dir / chapter).rglob("*.md")
    ]

    # Explainers under docs/concepts/ join the suite too, but only once they
    # carry a jupytext kernelspec -- a frontmatter-less explainer would convert
    # to a notebook with no kernel to start.
    concepts_dir = docs_dir / CONCEPTS_CHAPTER
    md_files += [
        (md, Path(CONCEPTS_CHAPTER) / md.relative_to(concepts_dir))
        for md in concepts_dir.rglob("*.md")
        if _is_executable_page(md)
    ]

    if not md_files:
        print(f"⚠️  Warning: No .md files found under {docs_dir!s}")
        return

    # Wrap the md_files list in tqdm to generate a clean progress bar
    for md_file, rel_path in tqdm(md_files, desc="Converting MyST to .ipynb", unit="file"):
        out_file = test_dir / rel_path.with_suffix(".ipynb")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["jupytext", "--to", "ipynb", str(md_file), "-o", str(out_file)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # The \n ensures the error prints clearly below the interrupted progress bar
            print(f"\n❌ Error converting '{md_file.name}':\n{e.stderr}")
            sys.exit(1)

    print("✅ Notebook generation complete!")


def run_tests() -> None:
    """
    Execute the full test suite.

    This automatically generates isolated Jupyter notebooks from the MyST Markdown
    source files, then triggers pytest on the entire 'tests' directory.
    """
    print("🔄 Starting test pipeline...")

    # 1. Generate the isolated notebooks safely
    generate_test_notebooks()

    # 2. Run the test suite
    print("🚀 Running pytest...")
    try:
        # pytest will pick up the configuration form the pyproject.toml
        subprocess.run(["pytest"], check=True)
    except subprocess.CalledProcessError as e:
        # Pass the exact exit code from pytest back to the terminal/CI
        sys.exit(e.returncode)


def run_lint() -> None:
    """
    Check formatting and lint the tree, exactly as the CI ``Lint`` job does.

    Formatting is checked before linting because the mechanical failure is the
    cheaper one to fix. Neither step writes: a formatting failure is repaired
    with ``uv run ruff format .``, a lint failure often with
    ``uv run ruff check . --fix``.
    """
    print("🔎 Checking formatting...", flush=True)
    try:
        subprocess.run(["ruff", "format", "--check", "."], check=True)
    except subprocess.CalledProcessError as e:
        print("\n❌ Formatting drift. Fix it with: uv run ruff format .")
        # Pass the exact exit code from ruff back to the terminal/CI
        sys.exit(e.returncode)

    print("\n🔎 Linting...", flush=True)
    try:
        subprocess.run(["ruff", "check", "."], check=True)
    except subprocess.CalledProcessError as e:
        print("\n❌ Lint errors. Auto-fix the safe ones with: uv run ruff check . --fix")
        sys.exit(e.returncode)

    # Ruff excludes *.md, so the docs pages get their own mechanical check here --
    # same job, same reasoning: drift a machine can repair should never reach review.
    print()
    check_notebook_format(fix=False)

    print("\n✅ Format, lint and docs formatting clean!")


# ==============================================================================
# NOTEBOOK FORMAT CHECK
# ==============================================================================
# Every page under docs/ that carries a jupytext `text_representation` header is
# half of a paired notebook: contributors may edit the .md or the .ipynb, but only
# the .md is committed. Whichever side they touched, jupytext has exactly one
# canonical .md rendering (set by [tool.jupytext] in pyproject.toml) -- and a page
# hand-edited in an editor drifts from it silently, so the next contributor who
# opens it in Jupyter sees a diff they never made. This check round-trips each page
# md -> ipynb -> md and fails on any difference.


def _paired_docs_pages() -> list[Path]:
    """
    Collect the docs pages that are jupytext-paired notebooks.

    A page qualifies when its YAML front matter carries jupytext's
    ``text_representation`` key -- the marker jupytext itself writes. Generated
    API stubs and the ``_build/`` output are plain Markdown and are skipped.

    Returns
    -------
    list of Path
        Project-relative paths, sorted, of every paired page under ``docs/``.
    """
    docs_dir = _get_docs_dir()
    project_root = docs_dir.parent

    pages = []
    for path in sorted(docs_dir.rglob("*.md")):
        if "_build" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        if "text_representation" in path.read_text(encoding="utf-8")[:2000]:
            pages.append(path.relative_to(project_root))

    return pages


def check_notebook_format(fix: bool | None = None) -> None:
    """
    Verify every paired docs page is in jupytext's canonical Markdown form.

    Runs ``jupytext --test-strict``, which converts each page to a notebook and
    back and compares the result byte for byte.

    Parameters
    ----------
    fix : bool or None, optional
        Rewrite the drifted pages in place instead of failing. ``None`` (the
        default) reads the flag from the command line, so ``uv run docs-format
        --fix`` repairs while ``uv run lint`` -- which passes ``False`` -- never
        writes.

    Raises
    ------
    SystemExit
        With jupytext's exit code when a page is not in canonical form.
    """
    if fix is None:
        fix = "--fix" in sys.argv[1:]
    pages = [str(p) for p in _paired_docs_pages()]

    if not pages:
        print("⚠️  No jupytext-paired pages found under docs/ — nothing to check.")
        return

    if fix:
        print(f"🛠️  Rewriting {len(pages)} docs page(s) in canonical jupytext form...", flush=True)
        for page in pages:
            subprocess.run(
                ["jupytext", "--quiet", "--to", "md:myst", "--output", page, page], check=True
            )
        print("\n✅ Docs pages canonicalized. Review the diff before committing.")
        return

    print(f"🔎 Checking jupytext format of {len(pages)} docs page(s)...", flush=True)
    try:
        subprocess.run(
            ["jupytext", "--quiet", "--test-strict", "--to", "ipynb", *pages], check=True
        )
    except subprocess.CalledProcessError as e:
        print(
            "\n❌ Docs pages differ from jupytext's canonical form (diff above:"
            " 'expected' is the committed file, 'actual' is what jupytext writes)."
            "\n   Fix them with: uv run docs-format --fix"
        )
        sys.exit(e.returncode)

    print("\n✅ Docs notebook formatting is canonical!")
