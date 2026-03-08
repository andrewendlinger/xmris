next, please give the corresponding snippet for the accessor. you must name it `apodize()`.

```python
class XmrisPlotAccessor:
    """Sub-accessor for xmris plotting functionalities (accessed via .xmr.plot)."""

    def __init__(self, obj: xr.DataArray):
        self._obj = obj

    def ridge(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: PlotRidgeConfig | None = None,
    ) -> plt.Axes:
        """Generate a ridge plot (2D waterfall) of stacked 1D spectra."""
        from xmris.visualization.plot import plot_ridge as _plot_ridge

        return _plot_ridge(
            da=self._obj, x_dim=x_dim, stack_dim=stack_dim, ax=ax, config=config
        )

    def heatmap(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: PlotHeatmapConfig | None = None,
    ) -> plt.Axes:
        """Generate a 2D heatmap plot of stacked 1D spectra."""
        from xmris.visualization.plot.plot_heatmap import plot_heatmap as _plot_heatmap

        return _plot_heatmap(
            da=self._obj, x_dim=x_dim, stack_dim=stack_dim, ax=ax, config=config
        )


class XmrisWidgetAccessor:
    """Sub-accessor for xmris interactive widget functionalities.

    This class provides a dedicated namespace for interactive UI components
    powered by AnyWidget. It is accessed via the `.xmr.widget` attribute
    on an xarray DataArray.
    """

    def __init__(self, obj: xr.DataArray):
        """
        Initialize the widget sub-accessor.

        Parameters
        ----------
        obj : xr.DataArray
            The underlying xarray DataArray object being operated on.
        """
        self._obj = obj

    def phase_spectrum(
        self,
        width: int = 740,
        height: int = 400,
        show_grid: bool = True,
        show_pivot: bool = True,
        **kwargs,
    ):
        """Open an interactive zero- and first-order phase correction widget.

        This method launches an AnyWidget-based user interface directly in the
        Jupyter Notebook. It allows for manual, real-time adjustment of the
        zero-order (p0) and first-order (p1) phase angles of a 1-D
        complex-valued NMR/MRS spectrum.

        Parameters
        ----------
        width : int, optional
            Width of the widget in pixels. Default is 740.
        height : int, optional
            Height of the widget in pixels. Default is 400.
        show_grid : bool, optional
            Toggle the background grid visibility. Default is True.
        show_pivot : bool, optional
            Toggle the visibility of the p1 pivot indicator. Default is True.
        **kwargs
            Additional arguments passed to the underlying PhaseWidget.

        Returns
        -------
        PhaseWidget
            The interactive widget instance. Assigning this to a variable allows
            you to programmatically extract the optimized phase angles after
            interacting with the UI.

        Raises
        ------
        ValueError
            If the underlying DataArray is not 1-dimensional or does not contain
            complex-valued data.

        Notes
        -----
        - **Zero-order phase (p0)**: Adjusts the phase uniformly across the spectrum.
        - **First-order phase (p1)**: Adjusts phase linearly relative to a pivot point.
        - The pivot point (p_pivot) is automatically set to the coordinate
          corresponding to the maximum magnitude peak.
        """
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import phase_spectrum

        # The underlying function handles the 1-D and complex-type validation
        return phase_spectrum(
            self._obj,
            width=width,
            height=height,
            show_grid=show_grid,
            show_pivot=show_pivot,
            **kwargs,
        )

    def scroll_spectra(
        self,
        scroll_axis: str | None = None,
        part: str = "real",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        show_trace: bool = True,
        trace_count: int = 10,
        width: int = 740,
        height: int = 400,
        **kwargs,
    ):
        """Open an interactive widget to scroll through a 2-D series of spectra.

        This method launches a user interface for exploring multi-dimensional
        spectroscopy data (e.g., transient repetitions, averages). It includes
        a timeline scrubber, animation playback, and fading historical traces.
        Clicking "Extract Slice" provides a copyable `.isel(...)` code snippet
        to isolate the current view while preserving pipeline lineage.

        Parameters
        ----------
        scroll_axis : str, optional
            The specific dimension to scroll through. If None, it attempts to
            auto-detect 'repetitions' or 'averages', or falls back to the
            non-spectral dimension.
        part : {'real', 'imag', 'abs'}, optional
            Which mathematical component of complex data to display. Default is 'real'.
        xlim : tuple of float, optional
            Static (min, max) bounds for the spectral axis.
        ylim : tuple of float, optional
            Static (min, max) bounds for intensity. If None, auto-ranges to the
            global minimum and maximum of the dataset.
        show_trace : bool, optional
            Show fading historical traces behind the current scan. Default is True.
        trace_count : int, optional
            The number of historical traces to overlay. Default is 10.
        width : int, optional
            Width of the widget in pixels. Default is 740.
        height : int, optional
            Height of the widget in pixels. Default is 400.
        **kwargs
            Additional arguments passed to the underlying ScrollWidget.

        Returns
        -------
        ScrollWidget
            The interactive widget instance.

        Raises
        ------
        ValueError
            If the input DataArray is not exactly 2-dimensional.
        """
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import scroll_spectra

        return scroll_spectra(
            self._obj,
            scroll_axis=scroll_axis,
            part=part,
            xlim=xlim,
            ylim=ylim,
            show_trace=show_trace,
            trace_count=trace_count,
            width=width,
            height=height,
            **kwargs,
        )
```



