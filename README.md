<div align="center">
  <img src="./docs/assets/logo.svg" alt="xmris logo" width="300" />
  
  <h1></h1>
  <p><b>A modern, N-dimensional, <code>xarray</code>-based toolbox for Magnetic Resonance Imaging and Spectroscopy.</b></p>

  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml">
    <img src="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml/badge.svg" alt="MyST GitHub Pages Deploy">
  </a>
  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/tests.yml">
    <img src="https://github.com/andrewendlinger/xmris/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://codecov.io/gh/andrewendlinger/xmris">
    <img src="https://codecov.io/gh/andrewendlinger/xmris/graph/badge.svg" alt="codecov">
  </a>
  <br>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python Versions">
  </a>
  <a href="https://www.gnu.org/licenses/agpl-3.0">
    <img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3">
  </a>
  <h1></h1>
</div>


## 📖 Documentation

Comprehensive documentation, tutorials, and API reference can be found **[here](https://andrewendlinger.github.io/xmris/)**.



## ⚡ Overview

`xmris` bridges the gap between modern data structures and magnetic resonance research. By building on top of `xarray`, it provides a robust environment for handling multi-dimensional MRI and MRS data with labeled coordinates, powerful broadcasting, and seamless integration with the scientific Python ecosystem.

**Key Features:**
* **N-Dimensional Data:** Native handling of complex MRI/MRS datasets using `xarray`.
* **MRS Integration:** Direct compatibility with tools like `pyAMARES` and `nmrglue`.
* **Modern Tooling:** Built for speed and reliability, developed using `uv` and strictly typed for modern Python environments.



## 🚀 Installation

*Note: `xmris` requires Python 3.11 or 3.12.*

You can install the package directly from PyPI using standard package managers:

```bash
# Using pip
pip install xmris

# Using uv (recommended)
uv add xmris

```

## 💻 Quick Start

```python
import numpy as np
import xarray as xr
import xmris  # Registers the .xmr accessor!

# 1. Create a dummy N-dimensional FID (e.g., 5 Voxels x 1024 Time points)
time = np.linspace(0, 1, 1024)
data = np.random.randn(5, 1024) + 1j * np.random.randn(5, 1024)

mrsi_data = xr.DataArray(
    data,
    dims=["Voxel", "Time"],
    coords={"Voxel": np.arange(5), "Time": time},
    attrs={"MHz": 120.0, "sw": 10000.0}
)

# 2. Process all voxels simultaneously using the .xmr accessor!
results = (
    mrsi_data
    .xmr.zero_fill(target_points=2048)
    .xmr.apodize_exp(lb=5.0)
    .xmr.to_spectrum()
    .xmr.autophase()
)
```

---

## 🛠️ Development

We use `uv` for lightning-fast dependency management and `Ruff` for linting/formatting. To set up a local development environment:

1. Fork this repository and then clone your version of this repository.

2. Sync the environment and install dependencies:
```bash
uv sync

```


3. Run tests via `pytest` (which includes notebook testing via `nbmake`):
```bash
uv run pytest

```


4. Build the [MyST](https://mystmd.org) based documentation :
```bash
uv run docs

```
More information can be found in the documentation [here](https://andrewendlinger.github.io/xmris/contribute/)


---

## ⚖️ License & Commercial Use

`xmris` is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
