## 02 Recommended Setup

We use modern, Rust-based tooling to keep the development environment blisteringly fast and completely reproducible.

### 1. `uv` (Environment & Package Management)

`uv` replaces `pip`, `virtualenv`, and `poetry`. It manages our dependencies and ensures isolated environments.

* **Install `uv`:** `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux) or via standard installation methods.
* **Bootstrap the project:** Run `uv sync` in the root directory. This reads `pyproject.toml` and creates the `.venv`.
* **Run commands:** Always prefix development commands with `uv run` (e.g., `uv run pytest`).

### 2. `ruff` (Linting & Formatting)

`ruff` is our single source of truth for code style.

* **Format code:** Run `uv run ruff format .`
* **Lint code:** Run `uv run ruff check .`
* **Auto-fix:** Run `uv run ruff check . --fix`

### 3. VS Code Configuration

To ensure a seamless experience, we use the official **Ruff extension**. This configuration enables "Format on Save" and organizes imports automatically using the project's specific `ruff` rules.

**Recommended Extensions:**

* [Python (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [Ruff (Astral Software)](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

**Settings (`.vscode/settings.json`):**
Create or update your `.vscode/settings.json` with the following to ensure the editor uses the `uv` created environment and handles formatting via `ruff`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.interpreter.infoVisibility": "always",
  "python.analysis.typeCheckingMode": "basic",
  
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "ruff.nativeServer": "on"
}

```

> **Pro Tip:** By setting `ruff.nativeServer` to `on`, you leverage the ultra-fast Rust-based language server, providing near-instant feedback as you type.