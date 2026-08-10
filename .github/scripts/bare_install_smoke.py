"""Prove that a bare ``pip install xmris`` is importable and usable.

Every other CI job installs with ``uv sync --all-extras --dev``, so the dependency
set a real user receives is exercised nowhere. That is the hole #122 fell through:
matplotlib is imported at module level by ``core/accessor.py`` -- the one module
every ``import xmris`` loads -- but arrived only transitively via pyAMARES, and
moving pyAMARES into the ``fitting`` extra took it away. ``import xmris`` raised
``ModuleNotFoundError`` on a fresh install for an entire release while CI stayed green.

Run this with an interpreter whose environment was built from ``[project]
dependencies`` **alone** -- no extras, no dev group. It asserts three things:

1. the declared ``requires-python`` actually admits the running interpreter;
2. the documented processing chain runs, plotting included;
3. the ``fitting`` extra is still genuinely optional, and absent means a *friendly*
   error rather than a traceback.

Exits non-zero with a diagnostic on the first failure.
"""

import importlib.metadata
import sys

import matplotlib

# A CI runner is headless. Select a non-interactive backend before pyplot is
# imported anywhere, so a missing display fails as a real bug and never as Tk.
matplotlib.use("Agg")

import xarray as xr  # noqa: E402  (must follow matplotlib.use)

import xmris  # noqa: E402


def check_requires_python() -> None:
    """Assert the running interpreter satisfies the distribution's own metadata.

    ``requires-python`` was once spelled ``<=3.13``, which PEP 440 reads as
    ``<= 3.13.0`` -- excluding every 3.13 patch release from 3.13.1 on. Installing
    from a local path does not enforce the marker, so CI never noticed; a user
    running ``pip install xmris`` on any real 3.13 was refused before dependency
    resolution even began. This catches that class of typo permanently.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    declared = importlib.metadata.metadata("xmris")["Requires-Python"]
    running = Version(".".join(str(n) for n in sys.version_info[:3]))
    if running not in SpecifierSet(declared):
        raise AssertionError(
            f"Python {running} does not satisfy the declared Requires-Python "
            f"({declared}), so `pip install xmris` would refuse this interpreter.\n"
            f"To fix this, correct `requires-python` in pyproject.toml -- a highest "
            f'supported minor of X.Y is spelled ">=...,<X.(Y+1)", never "<=X.Y".'
        )
    print(f"requires-python  OK  ({declared} admits {running})")


def check_processing_chain() -> None:
    """Run the chain the README and tutorials promise, plotting included."""
    fid = xmris.simulate_fid(
        amplitudes=[1.0, 0.6],
        chemical_shifts=[0.0, 5.2],
        reference_frequency=120.66,
        n_points=1024,
    )
    spectrum = fid.xmr.apodize_exp(lb=5.0).xmr.to_spectrum().xmr.autophase().xmr.to_ppm()
    assert spectrum.sizes, "the processed spectrum is empty"
    print("processing chain OK  (simulate -> apodize -> spectrum -> autophase -> ppm)")

    # matplotlib fails at *import*, but plotting is the reason it is a core dep --
    # so exercise a real figure rather than trusting that the import alone suffices.
    # simulate_fid is 1-D (#113), so stack by hand the way the tutorials do.
    stack = xr.concat(
        [
            xmris.simulate_fid(
                amplitudes=[amp],
                chemical_shifts=[0.0],
                reference_frequency=120.66,
                n_points=256,
            )
            for amp in (1.0, 0.8, 0.6)
        ],
        dim="repetition",
    )
    ax = stack.xmr.to_spectrum().xmr.autophase().real.xmr.plot.waterfall()
    assert ax is not None, "waterfall returned no Axes"
    print("plotting         OK  (waterfall rendered on the Agg backend)")


def check_fitting_is_optional() -> None:
    """Assert the fitting extra is absent *and* fails helpfully when reached."""
    assert "fit_amares" not in xmris.__all__, (
        "fit_amares is in __all__ on a fitting-free install, so `from xmris import *` "
        "would force the resolver and raise"
    )
    try:
        xmris.fit_amares
    except ImportError as exc:
        assert "xmris[fitting]" in str(exc), (
            f"fitting is missing, but the error does not point at the extra: {exc}"
        )
        print("fitting extra    OK  (absent, and the ImportError names xmris[fitting])")
    else:
        raise AssertionError(
            "xmris.fit_amares resolved on a bare install -- pyAMARES leaked into the "
            "core dependency set, which is what the fitting extra exists to prevent"
        )


def main() -> int:
    """Run every check, reporting the first failure."""
    for check in (check_requires_python, check_processing_chain, check_fitting_is_optional):
        try:
            check()
        except Exception as exc:  # noqa: BLE001  (a smoke test reports, it does not handle)
            sys.stdout.flush()  # keep the passing lines above the failure in CI logs
            print(f"\nFAIL in {check.__name__}: {exc}", file=sys.stderr)
            return 1
    print("\nA bare `pip install xmris` is importable and usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
