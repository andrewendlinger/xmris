<div align="center">
  <a href="https://andrewendlinger.github.io/xmris/">
    <img height="200" alt="fid_showpiece_logo_always_light" src="https://github.com/user-attachments/assets/d05d3a2b-5742-4b7e-8623-b5fdab885a6f" />
  </a>

  <!-- <img src="https://raw.githubusercontent.com/andrewendlinger/xmris/main/docs/assets/logo.svg" alt="xmris logo" width="300" /> -->
  
  
  <p><b>A modern, N-dimensional, <code>xarray</code>-based toolbox for Magnetic Resonance Imaging and Spectroscopy.</b></p>

  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml"><img src="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml/badge.svg" alt="MyST GitHub Pages Deploy"></a>
  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/tests.yml"><img src="https://github.com/andrewendlinger/xmris/actions/workflows/ci-fast.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/andrewendlinger/xmris"><img src="https://codecov.io/gh/andrewendlinger/xmris/graph/badge.svg" alt="codecov"></a>
  <br>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python Versions"></a>
  <a href="https://opensource.org/license/bsd-3-clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD 3-Clause"></a>
</div>


## 01. Documentation

**[Explore the official documentation](https://andrewendlinger.github.io/xmris/)** for tutorials, complete API references, and advanced usage guides.

Upgrading? The **[changelog](https://andrewendlinger.github.io/xmris/changelog)** records what changed in each release. There is no `CHANGELOG.md` here — it is a rendered page, so every entry can link the issue, the pull request, and the documentation behind it.



## 02. Overview

`xmris` bridges the gap between modern data structures and magnetic resonance research. By building on top of `xarray`, it provides a robust environment for handling multi-dimensional MRI and MRS data with labeled coordinates, powerful broadcasting, and seamless integration with the scientific Python ecosystem.

**Key Features:**
* **N-Dimensional Data:** Native handling of complex MRI/MRS datasets using `xarray`.
* **MRS Integration:** Direct compatibility with tools like `pyAMARES` and `nmrglue`.
* **Modern Tooling:** Built for speed and reliability, developed using `uv` and strictly typed for modern Python environments.



## 03. Installation

You can install the package directly from PyPI using standard package managers:

```bash
# Using pip
pip install xmris

# Using uv (recommended)
uv add xmris

```

## 04. Quick Start

```python
import numpy as np
import xarray as xr
import xmris  # Registers the .xmr accessor!

# 1. Create a dummy N-dimensional FID (e.g., 5 Voxels x 1024 Time points)
time = np.linspace(0, 1, 1024)
data = np.random.randn(5, 1024) + 1j * np.random.randn(5, 1024)

mrsi_data = xr.DataArray(
    data,
    dims=["voxel", "time"],
    coords={"voxel": np.arange(5), "time": time},
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

## 05. Development

We use `uv` for lightning-fast dependency management and `Ruff` for linting/formatting. To set up a local development environment:

1. Fork this repository and then clone your version of this repository.

2. Sync the environment and install dependencies:
```bash
uv sync

```


3. Run tests via `pytest` (which includes notebook testing via `nbmake`):
```bash
uv run test

```


4. Preview the [MyST](https://mystmd.org) documentation locally — a live server that never exits:
```bash
uv run docs
```
For a one-shot render check instead (the same command CI runs), use `myst build --html` from `docs/`.

More information can be found in the [contributing guide](https://andrewendlinger.github.io/xmris/guide).


---

### License

`xmris` is licensed under the **BSD 3-Clause License** — see [LICENSE](LICENSE). Use it, build on
it, ship it, commercially or not; keep the copyright notice.
