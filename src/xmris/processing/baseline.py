import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse.linalg import spsolve

from xmris.core.config import ATTRS, DIMS
from xmris.core.utils import _check_dims


def _als_core(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """1D Asymmetric Least Squares core algorithm.

    Optimized using SciPy sparse matrices.
    """
    L = len(y)

    # Explicitly set dtype=float to avoid the FutureWarning
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float)

    # Pre-convert the penalty matrix to CSC format outside the loop
    # for maximum efficiency.
    D_T_D = (lam * D.T.dot(D)).tocsc()

    # Initialize weights to 1
    w = np.ones(L)

    for _ in range(n_iter):
        # Create the weight matrix directly in CSC format
        W = sparse.diags(w, 0, format="csc", dtype=float)

        # Z is now the addition of two CSC matrices
        Z = W + D_T_D

        # spsolve now receives a pure CSC matrix, silencing the EfficiencyWarning
        z = spsolve(Z, w * y)

        # Asymmetric weighting: p for positive signal, (1-p) for baseline/noise
        w = p * (y > z) + (1 - p) * (y < z)

    return z


def baseline_als(
    da: xr.DataArray,
    dim: str = DIMS.frequency,
    lam: float = 1e5,
    p: float = 0.001,
    n_iter: int = 10,
) -> xr.DataArray:
    r"""Apply Asymmetric Least Squares (AsLS) baseline correction to a spectrum.

    This method automatically estimates and subtracts a smooth baseline without
    requiring user-defined signal-free regions. It operates strictly on the
    real (absorption) component of the data.

    .. warning::
        **Real-Valued Output Only:** This function discards the imaginary (dispersion) component
        of the data. AsLS relies on the asymmetry of absorption-mode peaks and cannot
        be applied to complex data without breaking Kramers-Kronig relations.
        The resulting real-valued spectrum is perfect for frequency-domain
        integration or peak fitting, but **cannot** be inverse-Fourier
        transformed back to the time domain for algorithms like AMARES.

    Parameters
    ----------
    da : xr.DataArray
        The input spectrum. Can be complex-valued and N-dimensional. If complex,
        the real component is extracted automatically.
    dim : str, optional
        The coordinate dimension along which to apply correction.
        Defaults to `DIMS.frequency`.
    lam : float, optional
        The smoothness penalty ($\lambda$). Higher values result in a stiffer,
        flatter baseline. Typical NMR ranges are 10,000 to 10,000,000.
        Defaults to 100,000.
    p : float, optional
        The asymmetry parameter. Controls how aggressively positive peaks are
        ignored during the fit. Typical ranges are 0.001 to 0.05.
        Defaults to 0.001.
    n_iter : int, optional
        Maximum number of iterations for the sparse solver. Defaults to 10.

    Returns
    -------
    xr.DataArray
        The strictly real-valued, baseline-corrected spectrum. Baseline
        parameters ($\lambda$, $p$, iterations) are appended to the dataset
        attributes to preserve data lineage.
    """
    _check_dims(da, dim, "baseline_als")

    # 1. Enforce real-valued math and intentionally discard the imaginary part
    is_complex = np.iscomplexobj(da.values)
    working_da = np.real(da) if is_complex else da

    # 2. Apply the 1D solver across N-dimensions automatically using xarray
    baseline = xr.apply_ufunc(
        _als_core,
        working_da,
        kwargs={"lam": lam, "p": p, "n_iter": n_iter},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="allowed",
    )

    # 3. Functional purity: Return a completely new array
    da_corrected = working_da - baseline

    # 4. Preserve original attributes
    da_corrected.attrs = da.attrs.copy()

    # 5. Append new metadata for strict lineage tracking
    da_corrected.attrs[ATTRS.baseline_method] = "als"
    da_corrected.attrs[ATTRS.baseline_lam] = lam
    da_corrected.attrs[ATTRS.baseline_p] = p
    da_corrected.attrs[ATTRS.baseline_iter] = n_iter

    return da_corrected
