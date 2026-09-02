(setup)=
# Recommended Setup


We use modern, Rust-based tooling to keep the development environment blisteringly fast, completely reproducible, and free of dependency conflicts.

(setup-uv)=
### 1. `uv` (Environment & Package Management)

`uv` replaces `pip`, `virtualenv`, and `poetry`. It manages our dependencies, locks versions, and ensures perfectly isolated virtual environments.

* **Install `uv`:** Run `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux) or check the [official docs](https://docs.astral.sh/uv/) for Windows/Homebrew methods.
* **Bootstrap the project:** Run `uv sync` in the root directory. This reads `pyproject.toml`, resolves dependencies, and automatically creates the `.venv` folder.
* **Run commands:** Always prefix development commands with `uv run` to ensure they execute inside the isolated environment (e.g., `uv run pytest` or `uv run jupyter lab`).
* **Add dependencies:** If your contribution requires a new package, do not use `pip install`. Instead, run `uv add <package_name>`. This automatically updates the `pyproject.toml` and lockfile.

(setup-ruff)=
### 2. `ruff` (Linting & Formatting)

`ruff` is our single source of truth for code style, replacing `black`, `flake8`, and `isort` with a single tool that runs in milliseconds.

* **Check both halves:** Run `uv run lint` — this is exactly what the `Lint` check on your pull request runs, so a green run here means a green check there. It ends with the docs-notebook format check; [Write a docs page](#contribute-docs-format) explains what that one guards and how to install it as a pre-commit hook.
* **Format code:** Run `uv run ruff format .`
* **Lint code:** Run `uv run ruff check .`
* **Auto-fix issues:** Run `uv run ruff check . --fix` (This automatically fixes safe issues like unused imports or variables).

`ruff format .` is safe to run over the whole repository: `pyproject.toml` excludes `*.md`, because ruff formats Python inside Markdown code blocks too and would flatten the hand-set snippets in the documentation. Prose layout is an authoring decision — see [the checks that gate a merge](#contribute-pr-checks).

(setup-vscode)=
### 3. VS Code Configuration

To ensure a seamless, "it-just-works" experience, we use the official **Ruff extension**.

**Required Extensions:**

* [Python (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [Ruff (Astral Software)](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

**Settings (`.vscode/settings.json`):**
The repository already ships one, and it is deliberately short — only the two things that are *properties of this project* rather than of you:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",

  "[python]": {
    "editor.rulers": [100]
  }
}
```

The interpreter path points at the `.venv` that `uv sync` created; the ruler matches `line-length = 100` from `pyproject.toml`, so your editor draws the margin ruff actually enforces.

Format-on-save is **not** set here, and that is on purpose: whether your editor reformats as you type is a personal preference, so it belongs in your own user settings rather than in everyone's checkout. If you want it, add this to your **user** `settings.json` once and every Python project benefits:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": true,
      "source.organizeImports.ruff": true
    }
  },

  "ruff.nativeServer": "on"
}
```

Nothing depends on you doing so — `uv run lint` and the `Lint` check catch the drift either way.

> **Pro Tip:** By setting `"ruff.nativeServer": "on"`, you bypass the Python wrapper and leverage the ultra-fast Rust-based language server directly. This provides near-instant, real-time feedback and auto-formatting as you type.