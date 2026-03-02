import base64
import json
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from IPython.display import HTML


def export_widget_static(
    widget_func: Callable[..., Any],
    *args: Any,
    max_points: int = 100_000,
    height_padding: int = 80,
    width_padding: int = 3,
    debug: bool = False,
    hide_close_button: bool = True,
    hide_selectors: list[str] | None = None,
    **kwargs: Any,
) -> HTML:
    """
    Universal wrapper to render an AnyWidget as a static HTML iframe.

    Instantiates the widget from a given factory function and its arguments, extracts
    its synchronized traitlets, compresses float arrays, and builds a standalone DOM.
    This allows interactive widgets to be embedded seamlessly into static
    documentation (e.g., MyST, JupyterBook, Sphinx) without requiring a live
    Python kernel.

    Parameters
    ----------
    widget_func : Callable[..., Any]
        The factory function that returns an AnyWidget instance.
    *args : Any
        Positional arguments to pass to `widget_func`.
    max_points : int, default: 100_000
        Safety threshold. If any synchronized list/array trait exceeds this length,
        it raises an error to prevent HTML document bloat.
    height_padding : int, default: 80
        Extra vertical pixels added to the iframe to prevent scrolling.
    width_padding : int, default: 3
        Extra horizontal pixels added to the iframe to prevent horizontal clipping.
    debug : bool, default: False
        If True, prints a detailed readout of the extracted traits, array sizes,
        and final HTML/JSON payload sizes to the standard output.
    hide_close_button : bool, default: True
        If True, automatically hides any HTML element with the class `remove-me-close-btn`.
        This is the standard convention for hiding teardown UI in static docs.
    hide_selectors : list of str, optional
        A list of additional CSS selectors representing elements to hide in the
        static render (e.g., `[".my-tooltip", "#save-btn"]`).
    **kwargs : Any
        Keyword arguments to pass to `widget_func`.

    Returns
    -------
    HTML
        An IPython display object containing the self-contained widget iframe.

    Raises
    ------
    ValueError
        If any synchronized trait array exceeds `max_points` in length, or if the
        final JSON payload exceeds ~2.5 MB, preventing silent browser iframe failures.
    """
    widget = widget_func(*args, **kwargs)

    if debug:
        print(f"--- Static Export Debug: {widget.__class__.__name__} ---")

    def _compress_and_check(val: Any, name: str, depth: int = 0) -> Any:
        if depth > 5:
            return val

        if isinstance(val, dict):
            return {
                k: _compress_and_check(v, f"{name}.{k}", depth + 1)
                for k, v in val.items()
            }

        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val)
            if arr.size > max_points:
                raise ValueError(
                    f"Widget trait '{name}' contains an array of size {arr.size} "
                    f"(shape {arr.shape}), exceeding the static limit of {max_points}. "
                    "Large payloads cause silent browser iframe failures. Please downsample."
                )
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.round(arr, 4)
            return arr.tolist()

        return val

    payload = {}
    for name, trait in widget.traits().items():
        if name in ["layout", "style", "comm"] or name.startswith("_"):
            continue

        if trait.metadata.get("sync"):
            raw_val = getattr(widget, name)

            if debug:
                if isinstance(raw_val, (list, tuple, np.ndarray)):
                    arr_size = np.asarray(raw_val).size
                    print(f"  [Sync] {name:<15} : Array/List (Size: {arr_size})")
                elif isinstance(raw_val, dict):
                    print(f"  [Sync] {name:<15} : Dictionary")
                else:
                    val_str = (
                        str(raw_val)[:30] + "..."
                        if len(str(raw_val)) > 30
                        else str(raw_val)
                    )
                    print(f"  [Sync] {name:<15} : {type(raw_val).__name__} = {val_str}")

            payload[name] = _compress_and_check(raw_val, name)

    json_str = json.dumps(payload)
    json_kb = len(json_str) / 1024
    json_mb = json_kb / 1024

    if debug:
        print(f"\n  JSON Payload Size : {json_kb:.2f} KB ({json_mb:.2f} MB)")

    if len(json_str) > 2_500_000:
        raise ValueError(
            f"The exported widget payload is too large ({json_mb:.2f} MB). "
            "Browsers will refuse to render data URIs this large, resulting in a blank iframe. "
            "Please slice or downsample your DataArray before exporting."
        )

    js_source = (
        widget._esm.read_text(encoding="utf-8")
        if hasattr(widget._esm, "read_text")
        else widget._esm
    )
    css_source = (
        widget._css.read_text(encoding="utf-8")
        if hasattr(widget._css, "read_text")
        else widget._css
    )

    w = getattr(widget, "width", 680)
    h = getattr(widget, "height", 400)

    # Build the dynamic CSS to hide UI elements
    hide_list = []
    if hide_close_button:
        hide_list.append(".remove-me-close-btn")
    if hide_selectors:
        hide_list.extend(hide_selectors)

    hide_css = ""
    if hide_list:
        selectors_str = ", ".join(hide_list)
        hide_css = f"{selectors_str} {{ display: none !important; }}"

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
{hide_css}
{css_source}
</style>
</head>
<body>
<div id="widget-root"></div>
<script type="module">
class StandaloneModel {{
    constructor(data) {{
        this._data = data;
        this._listeners = {{}};
    }}
    get(key) {{ return this._data[key]; }}
    set(key, val) {{
        this._data[key] = val;
        const evts = this._listeners[`change:${{key}}`] || [];
        evts.forEach(fn => fn());
    }}
    save_changes() {{ return Promise.resolve(); }}
    send(msg, callbacks, buffers) {{ console.warn("Widget attempted to send message to missing kernel:", msg); }}
    on(events, fn) {{
        for (const evt of events.split(" ")) {{
            (this._listeners[evt] ||= []).push(fn);
        }}
    }}
}}

{js_source}

const data = {json_str};
const model = new StandaloneModel(data);
const el = document.getElementById("widget-root");
render({{ model, el }});
</script>
</body>
</html>"""

    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    data_uri = f"data:text/html;base64,{encoded}"

    if debug:
        uri_kb = len(data_uri) / 1024
        uri_mb = uri_kb / 1024
        print(f"  Base64 URI Size   : {uri_kb:.2f} KB ({uri_mb:.2f} MB)")
        print("-" * 50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return HTML(
            f'<iframe src="{data_uri}" '
            f'allow="clipboard-write" '
            f'width="{w + width_padding}" height="{h + height_padding}" '
            f'style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;" '
            f'scrolling="no" '
            f'loading="lazy"></iframe>'
        )
