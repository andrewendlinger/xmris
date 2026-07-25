"""Build validated pyAMARES prior-knowledge tables from a friendly spec.

pyAMARES reads prior knowledge from a *positional* CSV/XLSX whose row order and
bound syntax are easy to get subtly — and silently — wrong:

- a blank phase bound becomes ``min = -inf`` in pyAMARES and trips a ``NaN`` trap
  in the phase wrapping, so a fit that looks configured returns garbage;
- a trailing digit in a peak name is read as a J-coupling *multiplet* component
  and summed into the base peak (``"ATP2"`` folds into ``"ATP"``);
- a tie target must occupy a column to the *left* of the peaks referencing it, or
  lmfit raises ``UnboundLocalError`` partway through the fit.

`build_prior_knowledge` lets you name peaks and give plain numbers; it emits a
correct file and refuses each trap at the door. Its output is the raw pyAMARES
CSV *text* — save it with ``Path("pk.csv").write_text(...)``, or skip the file
entirely and hand the spec straight to ``fit_amares(prior_knowledge=...)``.

The builder is dependency-light (no pyAMARES), so it is available even without
the optional ``fitting`` extra installed.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Any

from xmris.core.config import VARS

# --- pyAMARES prior-knowledge vocabulary (kept local — pyAMARES's, not xmris's) --

# The lineshape parameter ``g`` (0 = Lorentzian, 1 = Gaussian) is a pyAMARES knob,
# not an xmris fit-output variable, so it has no ``VARS`` term and stays a plain
# string here (mirroring how the result-column labels live locally in ``amares``).
_G = "g"

# xmris parameter name -> pyAMARES CSV row label, in the positional order pyAMARES
# reads (rows 1-5 are initial values, rows 7-11 are bounds).
_PK_ROWS: dict[str, str] = {
    VARS.amplitude: "amplitude",
    VARS.chem_shift: "chemicalshift",
    VARS.linewidth: "linewidth",
    VARS.phase: "phase",
    _G: "g",
}

# Per-parameter default bounds. Phase is ALWAYS bounded (-180, 180): a blank phase
# bound is -inf in pyAMARES and NaNs the fit. Amplitude is non-negative with an
# open upper bound; g is confined to [0, 1]. Chemical shift has no fixed default —
# it is a window around each peak's initial value (see ``shift_window``).
_DEFAULT_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    VARS.amplitude: (0.0, None),
    VARS.linewidth: (0.0, None),
    VARS.phase: (-180.0, 180.0),
    _G: (0.0, 1.0),
}

_REQUIRED = (VARS.amplitude, VARS.chem_shift, VARS.linewidth)
_INIT_KEYS = frozenset(_PK_ROWS)  # amplitude, chem_shift, linewidth, phase, g
_BOUND_KEYS = frozenset(f"{k}_bounds" for k in _PK_ROWS)


def _validate_peak_name(name: str) -> None:
    """Reject peak names pyAMARES would silently misread as multiplets."""
    # `str.isalpha()` is True for Unicode letters ('γATP'.isalpha() -> True), so gate
    # on ASCII too: a non-ASCII name reaches the positional CSV where pyAMARES may
    # misencode it or fail to match it against result rows — the corruption this guards.
    if not (name.isascii() and name.isalpha()):
        raise ValueError(
            f"Peak name {name!r} must be ASCII letters only. pyAMARES reads a trailing "
            f"digit as a J-coupling multiplet component and silently sums it into "
            f"the base peak (e.g. 'ATP2' folds into 'ATP'). For a genuine multiplet "
            f"model, write the CSV by hand and pass its path to fit_amares."
        )


def _fmt_bounds(lo: float | None, hi: float | None) -> str:
    """Format one ``(lower, upper)`` cell in pyAMARES bound syntax.

    ``None`` opens that side: ``(lo, `` (lower only) or ``, hi)`` (upper only) —
    the exact strings ``pyAMARES.parse_bounds`` recognizes.
    """
    if lo is None and hi is None:
        return ""
    if hi is None:
        return f"({lo}, "
    if lo is None:
        return f", {hi})"
    return f"({lo}, {hi})"


def _validate_spec(name: str, spec: Mapping[str, Any]) -> None:
    """Check one peak's keys — required present, nothing unknown (no aliases)."""
    missing = [k for k in _REQUIRED if k not in spec]
    if missing:
        raise ValueError(
            f"Peak {name!r} is missing required parameter(s): {missing}. "
            f"Every peak needs {list(_REQUIRED)}."
        )
    unknown = set(spec) - _INIT_KEYS - _BOUND_KEYS
    if unknown:
        raise ValueError(
            f"Peak {name!r} has unknown key(s): {sorted(unknown)}. "
            f"Valid keys are {sorted(_INIT_KEYS)} and their "
            f"'<name>_bounds' companions."
        )


def _resolve_bounds(
    name: str, spec: Mapping[str, Any], chem_shift: float, shift_window: float
) -> dict[str, tuple[float | None, float | None]]:
    """Resolve explicit ``(lower, upper)`` bounds for all five parameters."""
    out: dict[str, tuple[float | None, float | None]] = {}
    cs_bounds = spec.get(f"{VARS.chem_shift}_bounds")
    out[VARS.chem_shift] = (
        (_as_bound(cs_bounds[0]), _as_bound(cs_bounds[1]))
        if cs_bounds is not None
        else (chem_shift - shift_window, chem_shift + shift_window)
    )
    for row in (VARS.amplitude, VARS.linewidth, VARS.phase, _G):
        given = spec.get(f"{row}_bounds")
        out[row] = (
            (_as_bound(given[0]), _as_bound(given[1]))
            if given is not None
            else _DEFAULT_BOUNDS[row]
        )
    for row, (lo, hi) in out.items():
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Peak {name!r} {row} bounds {(lo, hi)} have lower > upper.")
    return out


def _as_bound(value: Any) -> float | None:
    """Coerce a bound endpoint, letting ``None`` mean 'open'."""
    return None if value is None else float(value)


def _num(value: Any, default: float) -> float:
    """Coerce an optional numeric init value, letting ``None`` mean 'use the default'.

    An explicit ``None`` is treated exactly like an absent key — a plain
    ``float(value)`` would raise ``TypeError`` on ``None``, and ``value or default``
    would wrongly override a legitimate ``0.0``.
    """
    return float(default if value is None else value)


def build_prior_knowledge(
    peaks: Mapping[str, Mapping[str, Any]],
    *,
    tie_phase_to: str | None = None,
    shift_window: float = 0.5,
) -> str:
    """Build a validated pyAMARES prior-knowledge CSV from a friendly spec.

    Each peak is named and described with plain numbers; the trap-prone positional
    layout, bound syntax, and column ordering are handled for you.

    Parameters
    ----------
    peaks : Mapping[str, Mapping[str, Any]]
        Peaks keyed by name (letters only). Each value maps parameters to values:
        the required ``"amplitude"`` (a.u.), ``"chem_shift"`` (ppm — absolute
        literature ppm when ``fit_amares`` is given a ``carrier``, else relative to
        it) and ``"linewidth"`` (Hz); the optional ``"phase"`` (degrees, default 0) and
        ``"g"`` (lineshape 0=Lorentzian..1=Gaussian, default 0); and optional
        ``"<name>_bounds"`` companions holding a ``(lower, upper)`` tuple (``None``
        opens a side). Phase is always bounded ``(-180, 180)`` unless overridden.
    tie_phase_to : str, optional
        Name of an anchor peak. When given, every other peak's phase is tied to the
        anchor's (an lmfit expression), and the anchor is written first so it is
        defined before the peaks referencing it. Defaults to None (free phases).
    shift_window : float, optional
        Half-width (ppm) of the default chemical-shift bound window placed
        symmetrically around each peak's initial shift when no explicit
        ``"chem_shift_bounds"`` is given. Defaults to 0.5.

    Returns
    -------
    str
        The pyAMARES prior-knowledge CSV as text. Pass the spec straight to
        ``fit_amares(prior_knowledge=...)`` to skip the file, or write this text to
        disk to keep a reusable, inspectable prior-knowledge file.

    Raises
    ------
    ValueError
        If ``peaks`` is empty, a peak name is not letters-only, a required
        parameter is missing, an unknown key is present, ``tie_phase_to`` names a
        peak that is not present, or any bound has lower > upper.

    Examples
    --------
    >>> csv_text = build_prior_knowledge(
    ...     {
    ...         "PCr": {"amplitude": 10, "chem_shift": 0.0, "linewidth": 15},
    ...         "ATP": {"amplitude": 5, "chem_shift": -7.5, "linewidth": 20,
    ...                 "chem_shift_bounds": (-8.0, -7.0)},
    ...     }
    ... )
    """
    if not peaks:
        raise ValueError("`peaks` is empty — provide at least one peak.")

    names = list(peaks)
    for name in names:
        _validate_peak_name(name)

    if tie_phase_to is not None:
        if tie_phase_to not in peaks:
            raise ValueError(f"tie_phase_to={tie_phase_to!r} is not one of the peaks {names}.")
        # Anchor first: a tie target must sit left of the peaks that reference it.
        names = [tie_phase_to] + [n for n in names if n != tie_phase_to]

    init: dict[str, list[Any]] = {row: [] for row in _PK_ROWS}
    bounds: dict[str, list[str]] = {row: [] for row in _PK_ROWS}

    for name in names:
        spec = peaks[name]
        _validate_spec(name, spec)
        chem_shift = float(spec[VARS.chem_shift])

        init[VARS.amplitude].append(float(spec[VARS.amplitude]))
        init[VARS.chem_shift].append(chem_shift)
        init[VARS.linewidth].append(float(spec[VARS.linewidth]))
        init[_G].append(_num(spec.get(_G), 0.0))
        # A bare anchor name in the phase cell becomes an lmfit tie; else a number.
        if tie_phase_to is not None and name != tie_phase_to:
            init[VARS.phase].append(tie_phase_to)
        else:
            init[VARS.phase].append(_num(spec.get(VARS.phase), 0.0))

        peak_bounds = _resolve_bounds(name, spec, chem_shift, shift_window)
        for row in _PK_ROWS:
            bounds[row].append(_fmt_bounds(*peak_bounds[row]))

    return _to_csv_text(names, init, bounds)


def _to_csv_text(
    names: list[str],
    init: Mapping[str, list[Any]],
    bounds: Mapping[str, list[str]],
) -> str:
    """Assemble the positional pyAMARES prior-knowledge CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Index", *names])
    writer.writerow(["Initial Values", *[""] * len(names)])
    for row_key, label in _PK_ROWS.items():
        writer.writerow([label, *init[row_key]])
    writer.writerow(["Bounds", *[""] * len(names)])
    for row_key, label in _PK_ROWS.items():
        writer.writerow([label, *bounds[row_key]])
    return buf.getvalue()
