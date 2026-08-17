<div align="center">
  <a href="https://andrewendlinger.github.io/xmris/">
    <img height="200" alt="fid_showpiece_logo_always_light" src="https://github.com/user-attachments/assets/d05d3a2b-5742-4b7e-8623-b5fdab885a6f" />
  </a>

  <!-- <img src="https://raw.githubusercontent.com/andrewendlinger/xmris/main/docs/assets/logo.svg" alt="xmris logo" width="300" /> -->
  
  
  <p><b>A modern, N-dimensional, <code>xarray</code>-based toolbox for Magnetic Resonance Imaging and Spectroscopy.</b></p>

  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml"><img src="https://github.com/andrewendlinger/xmris/actions/workflows/deploy.yml/badge.svg" alt="MyST GitHub Pages Deploy"></a>
  <a href="https://github.com/andrewendlinger/xmris/actions/workflows/ci-fast.yml"><img src="https://github.com/andrewendlinger/xmris/actions/workflows/ci-fast.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/andrewendlinger/xmris"><img src="https://codecov.io/gh/andrewendlinger/xmris/graph/badge.svg" alt="codecov"></a>
  <br>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python Versions"></a>
  <a href="https://opensource.org/license/bsd-3-clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD 3-Clause"></a>
</div>

MR data usually arrives as a bare array plus a pile of numbers you are expected to remember: which
axis is time, how wide the sweep was, what everything is referenced to. `xmris` keeps all of that
attached to the data instead. Your FID stays an ordinary
[`xarray`](https://xarray.dev) `DataArray` — dimensions named, coordinates in seconds, metadata
riding along — and the physics is one accessor away.

```python
spectrum = fid.xmr.to_spectrum().xmr.autophase().xmr.to_ppm()
```

Nothing gets wrapped in a custom class, so every `xarray` habit you already have keeps working, and
a grid of voxels goes through exactly the same call as a single one. Your spectrum comes out the
other side still knowing its ppm axis.

## Start with the documentation

### → [andrewendlinger.github.io/xmris](https://andrewendlinger.github.io/xmris/)

That is where the package really lives. Every tutorial page is an executable notebook, so the plots
and numbers you read there were produced by the code above them, and they are re-run on every pull
request. If you are new, [Basics](https://andrewendlinger.github.io/xmris/basics) walks the
FID → spectrum → ppm round trip from scratch;
[Concepts](https://andrewendlinger.github.io/xmris/concepts) explains why `xmris` is fussy about
names and metadata.

## Quick start

The shortest path to something you can look at: `simulate_fid` hands you a signal that already
carries the metadata the physics needs.

```python
import xmris  # importing registers the .xmr accessor on every xarray object

fid = xmris.simulate_fid(
    amplitudes=[1.0, 0.4],
    chemical_shifts=[0.0, 5.2],  # ppm
    reference_frequency=120.66,  # MHz, the Larmor frequency of your nucleus
    n_points=1024,
)

spectrum = (
    fid
    .xmr.apodize_exp(lb=5.0)
    .xmr.zero_fill(target_points=2048)
    .xmr.to_spectrum()
    .xmr.autophase()
    .xmr.to_ppm()
)

print(spectrum.dims)                         # ('chemical_shift',)
print(round(spectrum.attrs["phase_p0"], 2))  # 2.17 — autophase wrote down what it applied
```

With your own data you build the `DataArray` yourself, and then two things matter. The **time
coordinate** is the axis `xmris` reads: space it by your dwell time and the frequency axis follows
from it, no separate sweep-width setting to keep in sync. The **attributes** are the part only you
can know — `reference_frequency` (MHz) and `carrier_ppm` (which chemical shift sits at 0 Hz) — and
they are what turns Hz into ppm.

```python
import numpy as np
import xarray as xr
import xmris

n_points, spectral_width = 1024, 4000.0      # points, Hz
time = np.arange(n_points) / spectral_width  # seconds — this axis sets the frequency axis
signal = np.exp(2j * np.pi * 250.0 * time - time / 0.05)  # one peak, 250 Hz off centre

fid = xr.DataArray(
    np.stack([signal, 0.5 * signal, 0.25 * signal]),  # 3 voxels x 1024 points
    dims=["voxel", "time"],
    coords={"voxel": [0, 1, 2], "time": time},
    attrs={
        "reference_frequency": 120.66,  # MHz, what a shift in Hz gets measured against
        "carrier_ppm": 0.0,             # the ppm value sitting at 0 Hz (1H water: 4.7)
    },
)

spectrum = fid.xmr.to_spectrum()  # all three voxels at once, no loop
print(spectrum.coords["frequency"].values[[0, -1]])  # [-2000.  1996.09375]

ppm = spectrum.xmr.to_ppm()
print(ppm.dims)  # ('voxel', 'chemical_shift')
```

The 4000 Hz you spaced the time axis with is the 4000 Hz that comes back. Leave out
`reference_frequency` or `carrier_ppm` and `to_ppm` says so, by name, before doing any maths —
[Hz and ppm](https://andrewendlinger.github.io/xmris/basics/hz-and-ppm) takes it from here.

## What is in the box

- **Processing** — zero filling, exponential and Lorentz-to-Gauss apodization, manual and automatic
  phasing, asymmetric-least-squares baseline correction, FID ↔ spectrum, Hz ↔ ppm.
- **Vendor data** — Bruker ParaVision arrays and their parameter dicts become a fully labelled FID,
  digital-filter group delay included: the one that otherwise puts a phase roll through everything.
- **Fitting** — AMARES quantification via [pyAMARES](https://github.com/HawkMRS/pyAMARES), returning
  a `Dataset` with your signal, the fit and the residual aligned.
- **Plots and widgets** — matplotlib helpers, plus sliders you can drag to phase, apodize, or scroll
  through a stack of spectra.

And what is not: `xmris` is a `0.x` package, the spectroscopy side is further along than the
imaging side, Bruker is the only vendor loader so far, and full MRSI grids — lazy, chunked, on an
anatomical image — are still ahead. Core `xmris` will not do image reconstruction. The
[roadmap](https://andrewendlinger.github.io/xmris/roadmap) says what is shipped, what is moving,
and what is still being argued about.

## Install

```bash
pip install xmris             # or: uv add xmris
pip install "xmris[fitting]"  # adds AMARES quantification
```

Fitting is an extra because its dependency chain reaches `hlsvdpro`, which ships no arm64 wheel; the
`pyamares-xmris` repackage on PyPI carries the platform marker that makes it install on Apple
Silicon too. Everything else is in the bare install. Python 3.10 – 3.13.

## Contributing

Issues and pull requests are welcome. `uv sync --all-extras --dev` then `uv run test` gets you a
working checkout; the setup steps, the architecture contract and one page per kind of change are in
the [contributor guide](https://andrewendlinger.github.io/xmris/contribute).

## Changelog

Upgrading? The **[changelog](https://andrewendlinger.github.io/xmris/changelog)** records what
changed in each release. There is no `CHANGELOG.md` here — it is a rendered page, so every entry can
link the issue, the pull request, and the documentation behind it.

## License

`xmris` is licensed under the **BSD 3-Clause License** — see
[LICENSE](https://github.com/andrewendlinger/xmris/blob/main/LICENSE). Use it, build on
it, ship it, commercially or not; keep the copyright notice.
