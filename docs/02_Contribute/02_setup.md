## 02 Recommended Setup

We use modern, Rust-based tooling to keep the development environment blisteringly fast, completely reproducible, and free of dependency conflicts.

### 1. `uv` (Environment & Package Management)

`uv` replaces `pip`, `virtualenv`, and `poetry`. It manages our dependencies, locks versions, and ensures perfectly isolated virtual environments.

* **Install `uv`:** Run `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux) or check the [official docs](https://docs.astral.sh/uv/) for Windows/Homebrew methods.
* **Bootstrap the project:** Run `uv sync` in the root directory. This reads `pyproject.toml`, resolves dependencies, and automatically creates the `.venv` folder.
* **Run commands:** Always prefix development commands with `uv run` to ensure they execute inside the isolated environment (e.g., `uv run pytest` or `uv run jupyter lab`).
* **Add dependencies:** If your contribution requires a new package, do not use `pip install`. Instead, run `uv add <package_name>`. This automatically updates the `pyproject.toml` and lockfile.

### 2. `ruff` (Linting & Formatting)

`ruff` is our single source of truth for code style, replacing `black`, `flake8`, and `isort` with a single tool that runs in milliseconds.

* **Format code:** Run `uv run ruff format .`
* **Lint code:** Run `uv run ruff check .`
* **Auto-fix issues:** Run `uv run ruff check . --fix` (This automatically fixes safe issues like unused imports or variables).

### 3. VS Code Configuration

To ensure a seamless, "it-just-works" experience, we use the official **Ruff extension**. This configuration enables "Format on Save" and organizes imports automatically using the project's specific `ruff` rules.

**Required Extensions:**

* [Python (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [Ruff (Astral Software)](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

**Settings (`.vscode/settings.json`):**
Create or update your workspace `.vscode/settings.json` with the exact configuration below. This tells the editor to use the `uv`-created environment and delegates all formatting and linting directly to `ruff`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.interpreter.infoVisibility": "always",
  "python.analysis.typeCheckingMode": "basic",

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

> **Pro Tip:** By setting `"ruff.nativeServer": "on"`, you bypass the Python wrapper and leverage the ultra-fast Rust-based language server directly. This provides near-instant, real-time feedback and auto-formatting as you type.