"""Global runtime options for xmris, mirroring :func:`xarray.set_options`."""

from typing import Any

OPTIONS: dict[str, bool] = {
    "auto_convert": True,
}


class set_options:
    """Set global xmris options, either permanently or within a context.

    Parameters
    ----------
    auto_convert : bool, optional
        When ``True`` (the default), domain-decorated operations transform
        their input into the required physical domain automatically — the
        funnel (``@ensures_domain``) and domain-preserving (``@computes_in``)
        contracts. When ``False``, xmris runs *strict*: a domain mismatch
        raises an actionable error instead of converting, so every Fourier
        transform in a pipeline is written explicitly. Recommended for
        quantitative work.

    Examples
    --------
    Temporarily, as a context manager::

        with xmris.set_options(auto_convert=False):
            fid.xmr.autophase()   # raises: convert with .xmr.to_spectrum() first

    Or globally::

        xmris.set_options(auto_convert=False)
    """

    def __init__(self, **kwargs: bool):
        self.old: dict[str, bool] = {}
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"Unknown xmris option {key!r}. Available options: {sorted(OPTIONS)}"
                )
            self.old[key] = OPTIONS[key]
            OPTIONS[key] = value

    def __enter__(self) -> "set_options":
        """Enter the context; options were already applied in ``__init__``."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the option values that were active before this context."""
        OPTIONS.update(self.old)
