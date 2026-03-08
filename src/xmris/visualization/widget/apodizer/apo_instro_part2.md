**System Role / Objective:**
Implement a robust, well-engineered, and highly responsive Python `anywidget` widget for interactive NMR spectrum apodization. Adhere strictly to the architecture and file layout of the reference widget (ATTACHMENT B) and the mathematical logic of the Python backend (ATTACHMENT A).

The final output must consist of exactly three files: `apodizer.py`, `apodizer.js`, and `apodizer.css`.

**1. Python Backend (`apodizer.py`) & Data Preparation:**

* **Zero-Filling:** If the incoming time-domain array length is not a power of two, you must apply zero-filling to the next power of two *before* sending the data to the frontend.
* **Coordinates:** The widget must support both 'frequency' (Hz) and 'chemical_shift' (ppm) as xarrar input coordinates. The input xarray will either have a coordinate axis of 'chemical_shift' or 'frequency', if not the user must specify a dimension. 
* **No Direct Output:** The widget function itself must not return the modified array. It must only return the widget instance.
* **Naming**: The final function must be named `apodize_interactive`
* **Arguments**: The user should be able to change width and height of the widget.

**2. JavaScript Integration (`apodizer.js`):**

* **Incorporate Math Engine:** I have already written the pure JavaScript math engine for the FFT pipeline and apodization (ATTACHMENT C). Embed these functions directly into `apodizer.js` (or import them cleanly).
* **Zero-Latency Execution:** All UI updates (slider dragging, method switching) must trigger the JS math and canvas redraw instantly entirely in the browser, without syncing back to Python.

**3. UI Layout & Canvas Rendering (`apodizer.css` & `apodizer.js`):**

* **Top Canvas (FID):** Show the time-domain FID overlaid with the apodization weighting envelope. It should span the full width of the widget but have a decreased height compared to the spectrum canvas.
* **Bottom Canvas (Spectrum):** Show the resulting frequency-domain spectrum. Use whatever the user has provided (ppm or Hz) on the x-axis.
* **Component Linking:** Provide a dropdown/argument to select the spectrum display mode: Real (default), Imaginary, or Magnitude. The top FID canvas must automatically display the Real component by default, but switch to displaying the Imaginary component if the user selects the Imaginary spectrum.
* **Trace Colors:** Enforce the following line colors: Blue for Real, Red for Imaginary, and Black for Magnitude.
* **Reference Trace:** Include a "Faded Orig." checkbox that overlays the original, un-apodized data as a faded out gray line in the background of both canvases.
* **Grid and Axis:** Ensure both grid, ticks and axis are rendered correctly, similar to the reference widget. ensure the x-axis is inverted, following NMR standards (right to left).
* **Parameter Values:**  The linebroadening should be applied using large horizontal sliders. Only allow physically sensible values for LB and GB. add a reset button. the default should always be 0.

**4. Interactivity & Completion State:**

* **Apodization Toggle:** The user must be able to switch between the two apodization methods (Exponential and Lorentz-Gauss) via a UI dropdown. The values should always reset to zero.
* **Control Bar Layout:** Group the "Close" button. The close button should simply say "Close" (similar to the reference widget). Since the bar will likely get crowded and the default widget size is limited, go directly into a multi column, multi row layout to ensure every part has enough space and gets properly rendered.
* **Snippet Generation:** Similar to the reference widget, clicking "Close" must unmount the canvases and replace the UI with a success banner. This banner must display a copyable, valid Python code snippet demonstrating how to apply the chosen filter exactly as it exists in the Python backend (e.g., `da_apodized = da_fid.xmr.apodize_exp(lb=1.5)` or `da_apodized = da_fid.xmr.apodize_lg(lb=1.5, gb=0.5)`). Since the widget will be integrated into the overspanning package accessor, the user will be able to call this via `da.xmr.widget.apodize(...)`

---

### ATTACHMENT A: Python Backend Context (Apodization Math)

```python
def apodize_exp(da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
    """
    Apply an exponential weighting filter function for line broadening.

    During apodization, the time-domain FID signal $f(t)$ is multiplied with a filter
    function $f_{filter}(t) = e^{-t/T_L}$. This improves the Signal-to-Noise Ratio (SNR)
    because data points at the end of the FID, which primarily contain noise, are
    attenuated. The time constant $T_L$ is calculated from the desired line broadening
    in Hz.


    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default `DIMS.time`.
    lb : float, optional
        The desired line broadening factor in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    _check_dims(da, dim, "apodize_exp")

    t = da.coords[dim]

    # Calculate exponential filter: exp(-t / T_L) where T_L = 1 / (pi * lb)
    # This simplifies to: exp(-pi * lb * t)
    weight = np.exp(-np.pi * lb * t)

    # Functional application (transpose ensures broadcasting doesn't scramble axis order)
    da_apodized = (da * weight).transpose(*da.dims).assign_attrs(da.attrs)

    # Record lineage
    da_apodized.attrs[ATTRS.apodization_lb] = lb

    return da_apodized


def apodize_lg(
    da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0, gb: float = 1.0
) -> xr.DataArray:
    """
    Apply a Lorentzian-to-Gaussian transformation filter.

    This filter converts a Lorentzian line shape to a Gaussian line shape, which decays
    to the baseline in a narrower frequency range. The time-domain FID
    is multiplied by $e^{+t/T_L}e^{-t^2/T_G^2}$. The time constants $T_L$ and $T_G$
    are derived from the `lb` and `gb` frequency-domain parameters.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default `DIMS.time`.
    lb : float, optional
        The Lorentzian line broadening to cancel in Hz, by default 1.0.
    gb : float, optional
        The Gaussian line broadening to apply in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    _check_dims(da, dim, "apodize_lg")

    t = da.coords[dim]

    # Calculate Lorentzian cancellation: exp(+t / T_L)
    # T_L = 1 / (pi * lb)
    weight_lorentzian = np.exp(np.pi * lb * t)

    # Calculate Gaussian broadening: exp(-t^2 / T_G^2)
    # T_G = 2 * sqrt(ln(2)) / (pi * gb)
    if gb != 0:
        t_g = (2 * np.sqrt(np.log(2))) / (np.pi * gb)
        weight_gaussian = np.exp(-(t**2) / (t_g**2))
    else:
        weight_gaussian = 1.0

    weight = weight_lorentzian * weight_gaussian

    da_apodized = (da * weight).transpose(*da.dims).assign_attrs(da.attrs)

    # Record lineage
    da_apodized.attrs[ATTRS.apodization_lb] = lb
    da_apodized.attrs[ATTRS.apodization_gb] = gb

    return da_apodized
```

### ATTACHMENT B: Reference Architecture


> **CONTEXT BUNDLE INSTRUCTIONS**
> This document is a flattened representation of the widget sub-folder.
>
> **Structure:**
> * **Headers:** `#### filename` indicates the start of a new file.
> * **Content:** The file content follows immediately in a code block.

---

# Context Bundle Summary
**Source Directory:** `phase`
**Recursion Depth:** 0

**Total Files:** 3 | **Total Lines:** 759 | **Est. Tokens:** ~6,230

| File Name | Lines | Est. Tokens |
| --------- | ----- | ----------- |
| phase.css | 216   | ~1,249      |
| phase.js  | 395   | ~3,810      |
| phase.py  | 148   | ~1,171      |

#### phase.css
```css
/* =========================================
   Container & Layout
   ========================================= */
.nmr-viewer {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: system-ui, -apple-system, sans-serif;
    user-select: none;
    margin: 10px 0;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    overflow: hidden; /* Ensures sharp corners are maintained */
}

/* =========================================
   Canvas Area
   ========================================= */
.nmr-canvas-container {
    position: relative;
    background: #fafafa;
    border-bottom: 1px solid #eee;
}

.nmr-canvas {
    cursor: crosshair;
    outline: none;
    display: block;
}

.nmr-canvas:focus {
    box-shadow: inset 0 0 0 2px rgba(100,150,200,.15);
}

/* =========================================
   Floating Legend
   ========================================= */
.nmr-legend {
    position: absolute;
    top: 12px;
    right: 16px;
    background: rgba(255, 255, 255, 0.85);
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid #ddd;
    font-size: 11px;
    font-weight: 600;
    color: #444;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    pointer-events: none; /* Prevents mouse events from interfering with canvas drag */
}

/* Legend line indicators */
.leg-re { display: inline-block; width: 10px; height: 3px; background: #0055aa; border-radius: 1px; }
.leg-im { display: inline-block; width: 10px; height: 2px; background: #e63946; border-radius: 1px; }

/* =========================================
   Control Bar (Bottom)
   ========================================= */
.nmr-bar {
    display: flex;
    width: 100%;
    align-items: center;
    box-sizing: border-box;
    justify-content: space-between;
    padding: 10px 14px;
    background: #fff;
}

.nmr-grp { display: flex; align-items: center; gap: 10px; }
.nmr-lbl { font-size: 13px; color: #444; font-weight: 600; }

/* Input fields for phase parameters */
.nmr-input {
    width: 55px;
    padding: 4px;
    font-size: 13px;
    border: 1px solid #ccc;
    border-radius: 4px;
    text-align: center;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    color: #111;
    background: #f9f9f9;
    outline: none;
    transition: border-color 0.2s;
}

.nmr-input:focus {
    border-color: #0055aa;
    background: #fff;
}

/* Base styling for interactive buttons */
.nmr-btn {
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
    outline: none;
    display: flex;
    align-items: center;
    justify-content: center;
}

.nmr-btn-outline {
    border: 1px solid #ccc;
    background: #fff;
    color: #333;
}

.nmr-btn-outline:hover {
    background: #f0f0f0;
    border-color: #bbb;
}

/* Contextual hints text */
.nmr-hints {
    font-size: 11px;
    color: #888;
    font-style: italic;
    margin-right: 8px;
    letter-spacing: 0.2px;
}

/* =========================================
   Completion State (Banner & Copying)
   ========================================= */
.nmr-success-banner {
    padding: 16px;
    border: 1px solid #cce0f5;
    background: #f8fbff; /* Professional light blue, matching widget theme */
    border-radius: 8px;
    width: 100%;
    box-sizing: border-box;
    text-align: left;
    font-family: system-ui, -apple-system, sans-serif;
}

.nmr-success-title {
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 15px;
    color: #0055aa; /* Aligned with the real-line color */
}

.nmr-success-subtitle {
    font-size: 13px;
    color: #555;
    margin-bottom: 12px;
}

.nmr-copy-container {
    display: flex;
    gap: 8px;
    align-items: stretch;
}

.nmr-code-block {
    flex-grow: 1;
    background: #fff;
    border: 1px solid #d9d9d9;
    padding: 10px 12px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 13px;
    display: flex;
    align-items: center;
}

/* Usage hint text - unselectable and grayed out */
.nmr-code-hint {
    color: #999;
    user-select: none;
}

/* The actual text to be copied */
.nmr-code-target {
    color: #000;
    user-select: all; /* Allows 1-click highlighting of just the target code */
}

/* Actionable copy button */
.nmr-copy-btn {
    padding: 0 16px;
    border: 1px solid #004488;
    background: #0055aa;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.2s ease;
}

.nmr-copy-btn:hover {
    background: #004488;
}

/* Dynamic modifiers for JavaScript state toggling */
.nmr-copy-btn.copied {
    background: #28a745; /* Transition to green only upon successful action */
    border-color: #218838;
}

.nmr-copy-btn.failed {
    background: #dc3545;
    border-color: #c82333;
}
```

#### phase.js
```javascript
/**
 * Main rendering function for the AnyWidget phase viewer.
 * @param {Object} context - The AnyWidget context containing the model and DOM element.
 */
export function render({ model, el }) {
    const dpr = window.devicePixelRatio || 1;
    let W = model.get("width");
    let H = model.get("height");

    /* =========================================================================
       DOM Construction
       ========================================================================= */
    const root = document.createElement("div");
    root.className = "nmr-viewer";
    root.style.width = W + "px";

    const canvasContainer = document.createElement("div");
    canvasContainer.className = "nmr-canvas-container";
    canvasContainer.style.width = W + "px";
    canvasContainer.style.height = H + "px";

    const canvas = document.createElement("canvas");
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + "px";
    canvas.style.height = H + "px";
    canvas.className = "nmr-canvas";
    canvas.tabIndex = 0;

    // Overlay legend indicating Real and Imaginary colors
    const legend = document.createElement("div");
    legend.className = "nmr-legend";
    legend.innerHTML = "<span class='leg-re'></span> Real &nbsp;&nbsp; <span class='leg-im'></span> Imag";

    canvasContainer.append(canvas, legend);

    /* Control Bar Elements */
    const bar = document.createElement("div");
    bar.className = "nmr-bar";

    // Left Control Group: Phase inputs and Reset
    const grpL = document.createElement("div");
    grpL.className = "nmr-grp";

    const p0Lbl = document.createElement("label"); p0Lbl.className = "nmr-lbl"; p0Lbl.textContent = "p0 [°]";
    const p0In  = document.createElement("input"); p0In.type = "number"; p0In.className = "nmr-input";
    p0In.step = "1"; p0In.value = model.get("p0").toFixed(1);

    const p1Lbl = document.createElement("label"); p1Lbl.className = "nmr-lbl"; p1Lbl.textContent = "p1 [°]";
    const p1In  = document.createElement("input"); p1In.type = "number"; p1In.className = "nmr-input";
    p1In.step = "1"; p1In.value = model.get("p1").toFixed(1);

    const resetBtn = document.createElement("button");
    resetBtn.className = "nmr-btn nmr-btn-outline"; resetBtn.textContent = "Reset";

    grpL.append(p0Lbl, p0In, p1Lbl, p1In, resetBtn);

    // Right Control Group: Interaction hints and Close action
    const grpR = document.createElement("div");
    grpR.className = "nmr-grp";

    const hints = document.createElement("div"); hints.className = "nmr-hints";
    hints.textContent = "Drag: p0 | Shift+Drag: p1";

    // CONVENTION: Always add the 'remove-me-close-btn' class to buttons that finalize,
    // close, or require a live Jupyter kernel. This allows the static documentation
    // exporter to automatically hide them when rendered in a standalone HTML iframe.
    // Also, keep this comment if you take this code as reference for a new widget.
    const closeBtn = document.createElement("button");
    closeBtn.className = "nmr-btn nmr-btn-outline remove-me-close-btn";
    closeBtn.textContent = "Close";
    closeBtn.title = "Finalize Phase Parameters";

    // Handle widget teardown and generation of the final code snippet
    closeBtn.onclick = () => {
        const p0 = model.get("p0").toFixed(2);
        const p1 = model.get("p1").toFixed(2);
        const pivot = model.get("pivot_val").toFixed(3);

        const hintStr = `phased_da = da`;
        const targetStr = `.xmr.phase(p0=${p0}, p1=${p1}, pivot=${pivot})`;

        // Replace the widget UI with a professional completion banner
        root.innerHTML = `
            <div class="nmr-success-banner">
                <div class="nmr-success-title">Phase Correction Parameters Extracted</div>
                <div class="nmr-success-subtitle">Copy the generated code snippet below to apply these parameters to your dataset:</div>
                <div class="nmr-copy-container">
                    <div class="nmr-code-block">
                        <span class="nmr-code-hint">${hintStr}</span><span class="nmr-code-target">${targetStr}</span>
                    </div>
                    <button id="nmr-copy-btn" class="nmr-copy-btn">Copy Code</button>
                </div>
            </div>
        `;

        // Configure the clipboard copy button behavior
        const copyBtn = root.querySelector("#nmr-copy-btn");
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(targetStr).then(() => {
                copyBtn.textContent = "Copied ✓";
                copyBtn.classList.add("copied");

                setTimeout(() => {
                    copyBtn.textContent = "Copy Code";
                    copyBtn.classList.remove("copied");
                }, 2000);
            }).catch(err => {
                console.error("Failed to copy text: ", err);
                copyBtn.textContent = "Failed";
                copyBtn.classList.add("failed");
            });
        };
    };

    grpR.append(hints, closeBtn);
    bar.append(grpL, grpR);
    root.append(canvasContainer, bar);
    el.appendChild(root);

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    /* =========================================================================
       Canvas Drawing & Math
       ========================================================================= */
    let gYMin = -1, gYMax = 1;

/**
     * Calculates the Y-axis boundaries. Instead of forcing symmetry,
     * it finds the actual data bounds and adds a 15% margin for visibility.
     */
    function recomputeY() {
        const Re = model.get("reals");
        const Im = model.get("imags");
        if (!Re?.length) return;

        let min = Infinity;
        let max = -Infinity;

        // Check both Real and Imaginary bounds to ensure both stay in view
        for (let i = 0; i < Re.length; i++) {
            if (Re[i] < min) min = Re[i];
            if (Re[i] > max) max = Re[i];
            if (Im[i] < min) min = Im[i];
            if (Im[i] > max) max = Im[i];
        }

        const range = max - min;
        const pad = range * 0.15 || 1.0;

        gYMax = max + pad;
        gYMin = min - pad;
    }

    let raf = null;
    function scheduleDraw() {
        if (!raf) raf = requestAnimationFrame(() => { raf = null; draw(); });
    }

    /**
     * Core rendering routine. Handles axis drawing, scaling, and applying
     * the mathematical phase shift to the real and imaginary components.
     */
    function draw() {
        const P = model.get("x_coords");
        const Re = model.get("reals");
        const Im = model.get("imags");
        if (!P?.length || !Re?.length) return;

        const p0_deg = model.get("p0");
        const p1_deg = model.get("p1");
        const pivot = model.get("pivot_val");

        // Sync inputs with current model state
        p0In.value = p0_deg.toFixed(1);
        p1In.value = p1_deg.toFixed(1);

        ctx.clearRect(0, 0, W, H);

        const x0 = Math.min(P[0], P[P.length-1]);
        const x1 = Math.max(P[0], P[P.length-1]);
        const x_range = x1 - x0;

        // Static margins guarantee room for axes and labels
        const mg = { t: 22, r: 22, b: 44, l: 64 };
        const pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;
        const toX = v => mg.l + pw * (x1 - v) / x_range;
        const toY = v => mg.t + ph * (1 - (v - gYMin) / (gYMax - gYMin));

        const xt = ticks(x0, x1, 8), yt = ticks(gYMin, gYMax, 6);

        /* 1. Draw Background Grid (if enabled) */
        const showGrid = model.get("show_grid") === false ? false : true;
        if (showGrid) {
            ctx.strokeStyle = "#e0e0e0";
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            for (const v of xt) { const x = toX(v); ctx.moveTo(x, mg.t); ctx.lineTo(x, mg.t+ph); }
            for (const v of yt) { const y = toY(v); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l+pw, y); }
            ctx.stroke();
        }

        /* 2. Draw Solid Axes bounding box */
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mg.l, mg.t); ctx.lineTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph);
        ctx.stroke();

        /* 3. Draw Ticks & Labels */
        ctx.fillStyle = "#666"; ctx.font = "11px sans-serif";
        ctx.beginPath();
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (const v of xt) {
            const x = toX(v);
            ctx.moveTo(x, mg.t + ph); ctx.lineTo(x, mg.t + ph + 5);
            ctx.fillText(v.toFixed(1), x, mg.t + ph + 8);
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        for (const v of yt) {
            const y = toY(v);
            ctx.moveTo(mg.l, y); ctx.lineTo(mg.l - 5, y);
            ctx.fillText(nfmt(v), mg.l - 8, y);
        }
        ctx.stroke();

        /* 4. Draw X-Axis Title */
        ctx.fillStyle = "#444"; ctx.font = "12px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(model.get("x_label"), mg.l + pw/2, mg.t + ph + 28);

        /* 5. Phase Correction Math & Data Plotting */
        ctx.save();
        ctx.beginPath(); ctx.rect(mg.l, mg.t, pw, ph); ctx.clip();

        // Draw Zero Baseline only if it's currently in view
        if (0 >= gYMin && 0 <= gYMax) {
            ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1;
            const zeroY = toY(0);
            ctx.beginPath(); ctx.moveTo(mg.l, zeroY); ctx.lineTo(mg.l+pw, zeroY); ctx.stroke();
        }

        ctx.save();
        ctx.beginPath(); ctx.rect(mg.l, mg.t, pw, ph); ctx.clip();

        const imPoints = [];
        const rePoints = [];

        const p0_rad = p0_deg * Math.PI / 180.0;
        const p1_rad = p1_deg * Math.PI / 180.0;

        // Apply phase transformation: (Re + i*Im) * exp(-i*phase)
        for (let i = 0; i < P.length; i++) {
            if (P[i] < x0 || P[i] > x1) continue;

            const phase = p0_rad + p1_rad * ((P[i] - pivot) / x_range);
            const cosP = Math.cos(phase);
            const sinP = Math.sin(phase);

            const r = Re[i], m = Im[i];
            const phasedRe = r * cosP - m * sinP;
            const phasedIm = r * sinP + m * cosP;

            const x = toX(P[i]);
            rePoints.push({x, y: toY(phasedRe)});
            imPoints.push({x, y: toY(phasedIm)});
        }

        // Render Imaginary Component (Red)
        ctx.beginPath();
        ctx.strokeStyle = "#e63946"; ctx.lineWidth = 1.0; ctx.globalAlpha = 0.8;
        for (let i=0; i<imPoints.length; i++) {
            i === 0 ? ctx.moveTo(imPoints[i].x, imPoints[i].y) : ctx.lineTo(imPoints[i].x, imPoints[i].y);
        }
        ctx.stroke();

        // Render Real Component (Blue)
        ctx.beginPath();
        ctx.strokeStyle = "#0055aa"; ctx.lineWidth = 1.8; ctx.globalAlpha = 1.0;
        for (let i=0; i<rePoints.length; i++) {
            i === 0 ? ctx.moveTo(rePoints[i].x, rePoints[i].y) : ctx.lineTo(rePoints[i].x, rePoints[i].y);
        }
        ctx.stroke();

        ctx.restore();

        // Mark the Pivot point with a dashed line (if enabled)
        const showPivot = model.get("show_pivot") !== false;

        if (showPivot) {
            const pivX = toX(pivot);
            if (pivX >= mg.l - 2 && pivX <= mg.l + pw + 2) {
                ctx.save();
                ctx.beginPath();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = "rgba(100, 100, 100, 0.5)";
                ctx.lineWidth = 1.5;
                ctx.moveTo(pivX, mg.t);
                ctx.lineTo(pivX, mg.t + ph);
                ctx.stroke();
                ctx.restore();
            }
        }
    }

    /**
     * Computes visually pleasing "nice" tick intervals for the axes.
     */
    function ticks(lo, hi, n) {
        const r = hi - lo; if (r <= 0) return [lo];
        const raw = r / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const q = raw / mag;
        const step = q < 1.5 ? mag : q < 3.5 ? 2*mag : q < 7.5 ? 5*mag : 10*mag;
        const out = []; let v = Math.ceil(lo / step) * step;
        while (v <= hi + step*1e-9) { out.push(parseFloat(v.toPrecision(12))); v += step; }
        return out;
    }

    /**
     * Formats tick labels to prevent overly long decimals or awkward scientific notation.
     */
    function nfmt(n) {
        const a = Math.abs(n);
        if (n === 0) return "0";
        if (a >= 1e4 || (a > 0 && a < .01)) return n.toExponential(1);
        return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(1) : n.toFixed(2);
    }

    /* =========================================================================
       Mouse Interactions
       ========================================================================= */
    let isDragging = false;
    let startY = 0, startP0 = 0, startP1 = 0;
    let activeMode = 'p0';

    canvas.addEventListener("mousedown", e => {
        isDragging = true;
        startY = e.clientY;
        startP0 = model.get("p0");
        startP1 = model.get("p1");
        // Shift key determines if we are adjusting zero-order (p0) or first-order (p1) phase
        activeMode = e.shiftKey ? 'p1' : 'p0';
        canvas.style.cursor = "ns-resize";
        e.preventDefault();
    });

    window.addEventListener("mousemove", e => {
        if (!isDragging) return;
        const dy = startY - e.clientY;
        if (activeMode === 'p0') { model.set("p0", startP0 + dy * 0.5); }
        else { model.set("p1", startP1 + dy * 1.5); }
        model.save_changes();
    });

    window.addEventListener("mouseup", () => {
        if (isDragging) { isDragging = false; canvas.style.cursor = "crosshair"; }
    });

    /* =========================================================================
       Event Observers & Wiring
       ========================================================================= */
    const reDraw = scheduleDraw;
    const recompDraw = () => { recomputeY(); scheduleDraw(); };

    model.on("change:p0 change:p1", reDraw);
    model.on("change:reals change:imags change:x_coords change:show_grid", recompDraw);

    // Ensure responsive resizing
    model.on("change:width change:height", () => {
        W = model.get("width");
        H = model.get("height");
        root.style.width = W + "px";
        canvasContainer.style.width = W + "px";
        canvasContainer.style.height = H + "px";
        canvas.width = W * dpr; canvas.height = H * dpr;
        canvas.style.width = W + "px"; canvas.style.height = H + "px";
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset scale matrix
        ctx.scale(dpr, dpr);
        scheduleDraw();
    });

    // Wire input boxes to update the model dynamically
    p0In.addEventListener("change", () => { model.set("p0", parseFloat(p0In.value) || 0); model.save_changes(); });
    p1In.addEventListener("change", () => { model.set("p1", parseFloat(p1In.value) || 0); model.save_changes(); });

    // Reset button zeroes out the phase
    resetBtn.addEventListener("click", () => { model.set("p0", 0); model.set("p1", 0); model.save_changes(); });

    recomputeY();
    scheduleDraw();
}
```

#### phase.py
```python
import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

_HERE = pathlib.Path(__file__).parent


class PhaseWidget(anywidget.AnyWidget):
    """Interactive widget for manual NMR spectra phase correction.

    Provides a graphical interface for adjusting zero-order (p0) and
    first-order (p1) phase terms.

    Attributes
    ----------
    width : int
        Pixel width of the rendering canvas.
    height : int
        Pixel height of the rendering canvas.
    show_grid : bool
        If True, renders background grid lines.
    show_pivot : bool
        If True, displays a dashed vertical line at the `pivot_val` location.
    x_coords : list of float
        The spectral axis coordinates (e.g., ppm or Hz).
    x_label : str
        The label displayed on the X-axis.
    reals : list of float
        Real component of the complex spectrum.
    imags : list of float
        Imaginary component of the complex spectrum.
    mag : list of float
        Magnitude ($|S|$) used for initial auto-scaling of the view.
    p0 : float
        Current zero-order phase correction in degrees.
    p1 : float
        Current first-order phase correction in degrees.
    pivot_val : float
        The frequency/coordinate where the $p_1$ phase shift is zero.
    """

    _esm = _HERE / "phase.js"
    _css = _HERE / "phase.css"

    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(400).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_pivot = traitlets.Bool(True).tag(sync=True)
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)
    reals = traitlets.List().tag(sync=True)
    imags = traitlets.List().tag(sync=True)
    mag = traitlets.List().tag(sync=True)
    p0 = traitlets.Float(0.0).tag(sync=True)
    p1 = traitlets.Float(0.0).tag(sync=True)
    pivot_val = traitlets.Float(0.0).tag(sync=True)


def phase_spectrum(
    da: xr.DataArray,
    width: int = 740,
    height: int = 400,
    show_grid: bool = True,
    show_pivot: bool = True,
) -> PhaseWidget:
    """
    Instantiate an interactive phase correction viewer for a 1-D complex xarray.

    This function automatically detects the spectral dimension and sets a
    physically sensible pivot point at the maximum signal intensity.

    Parameters
    ----------
    da : xr.DataArray
        A 1-dimensional, complex-valued DataArray. Must contain coordinates
        representing the spectral axis (e.g., 'ppm' or 'Hz').
    width : int, optional
        Width of the widget in pixels. The default is 740.
    height : int, optional
        Height of the widget in pixels. The default is 400.
    show_grid : bool, optional
        Toggle the background grid visibility. The default is True.
    show_pivot : bool, optional
        Toggle the visibility of the $p_1$ pivot indicator. The default is True.

    Returns
    -------
    PhaseWidget
        An interactive widget instance synchronized with the provided data.

    Raises
    ------
    ValueError
        If the input `da` is not 1-dimensional or contains non-complex data.

    Notes
    -----
    The pivot point is crucial for first-order phasing ($p_1$). This function
    sets the pivot to the maximum of the magnitude spectrum to simplify
    local phase adjustments.
    """
    if da.ndim != 1:
        raise ValueError(f"Input must be 1-D, but has shape {da.shape}.")

    if not np.iscomplexobj(da.values):
        raise ValueError("Phasing requires complex-valued data (Real + Imaginary).")

    spec_dim = None
    x_label = "Frequency"

    # Identify spectral dimension by common naming conventions
    for d in da.dims:
        d_str = str(d).lower()
        if any(k in d_str for k in ("ppm", "chem", "shift")):
            spec_dim = d
            x_label = "Chemical Shift [ppm]"
            break
        elif any(k in d_str for k in ("hz", "freq")):
            spec_dim = d
            x_label = "Frequency [Hz]"
            break

    if spec_dim is None:
        spec_dim = da.dims[0]
        x_label = str(spec_dim)

    x_vals = da.coords[spec_dim].values.astype(float)
    vals = da.values
    mag_vals = np.abs(vals).astype(float)

    # Heuristic: Pivot at the highest peak
    pivot = float(x_vals[np.argmax(mag_vals)])

    return PhaseWidget(
        width=width,
        height=height,
        show_grid=show_grid,
        show_pivot=show_pivot,
        x_coords=x_vals.tolist(),
        x_label=x_label,
        reals=np.real(vals).astype(float).tolist(),
        imags=np.imag(vals).astype(float).tolist(),
        mag=mag_vals.tolist(),
        pivot_val=pivot,
    )
```



### ATTACHMENT C: JS Math Engine

this can either live in `math_engine.js` and get stiched together in python, or be part of `apodizer.js`

```javascript
/**
 * Magnetic Resonance Spectroscopy (MRS) DSP Math Module
 * * Replicates core Python/NumPy/xarray functionality for FID processing
 * in pure, dependency-free JavaScript using TypedArrays.
 */

export const MRSDSP = {
    /**
     * Shifts an array circularly by a given number of positions.
     * Mimics `numpy.roll(arr, shift)`.
     * * @param {Float64Array} arr - The input array.
     * @param {number} shift - The number of positions to shift.
     * @returns {Float64Array} A new shifted array.
     */
    _roll(arr, shift) {
        const n = arr.length;
        const res = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            res[(i + shift) % n] = arr[i];
        }
        return res;
    },

    /**
     * Shifts the zero-frequency component to the center of the spectrum.
     * * @param {Float64Array} real - Real part of the complex array.
     * @param {Float64Array} imag - Imaginary part of the complex array.
     * @returns {{real: Float64Array, imag: Float64Array}} Shifted complex arrays.
     */
    fftshift(real, imag) {
        const n = real.length;
        const shift = Math.floor(n / 2);
        return {
            real: this._roll(real, shift),
            imag: this._roll(imag, shift)
        };
    },

    /**
     * Inverse of `fftshift`, moves the zero-frequency component back to index 0.
     * * @param {Float64Array} real - Real part of the complex array.
     * @param {Float64Array} imag - Imaginary part of the complex array.
     * @returns {{real: Float64Array, imag: Float64Array}} Shifted complex arrays.
     */
    ifftshift(real, imag) {
        const n = real.length;
        const shift = Math.floor((n + 1) / 2);
        return {
            real: this._roll(real, shift),
            imag: this._roll(imag, shift)
        };
    },

    /**
     * Core Cooley-Tukey Radix-2 FFT algorithm.
     * Assumes `N` is strictly a power of 2.
     * Applies `norm="ortho"` scaling by 1/sqrt(N).
     * * @param {Float64Array} real_in - Real input array.
     * @param {Float64Array} imag_in - Imaginary input array.
     * @param {boolean} isInverse - True for IFFT, False for forward FFT.
     * @returns {{real: Float64Array, imag: Float64Array}} Transformed arrays.
     */
    _fft_radix2(real_in, imag_in, isInverse = false) {
        const n = real_in.length;
        const real = new Float64Array(real_in);
        const imag = new Float64Array(imag_in);

        // Bit-reversal permutation
        let j = 0;
        for (let i = 0; i < n - 1; i++) {
            if (i < j) {
                let tr = real[i];
                let ti = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tr;
                imag[j] = ti;
            }
            let m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Butterfly operations
        for (let step = 1; step < n; step <<= 1) {
            const jump = step << 1;
            const theta = (isInverse ? Math.PI : -Math.PI) / step;

            for (let i = 0; i < step; i++) {
                let u_r = 1.0;
                let u_i = 0.0;
                if (i > 0) {
                    const angle = i * theta;
                    u_r = Math.cos(angle);
                    u_i = Math.sin(angle);
                }

                for (let k = i; k < n; k += jump) {
                    const match = k + step;
                    const tr = u_r * real[match] - u_i * imag[match];
                    const ti = u_r * imag[match] + u_i * real[match];

                    real[match] = real[k] - tr;
                    imag[match] = imag[k] - ti;
                    real[k] += tr;
                    imag[k] += ti;
                }
            }
        }

        // Ortho-normalization: scale by 1 / sqrt(N)
        const norm = 1.0 / Math.sqrt(n);
        for (let i = 0; i < n; i++) {
            real[i] *= norm;
            imag[i] *= norm;
        }

        return { real, imag };
    },

    /**
     * Standard Forward Ortho-normalized FFT.
     */
    fft_ortho(real, imag) {
        return this._fft_radix2(real, imag, false);
    },

    /**
     * Standard Inverse Ortho-normalized IFFT.
     */
    ifft_ortho(real, imag) {
        return this._fft_radix2(real, imag, true);
    },

    /**
     * Convert time-domain FID to frequency-domain spectrum.
     * Equivalent to python's `fft` followed by `fftshift`.
     */
    to_spectrum(real, imag) {
        const freq = this.fft_ortho(real, imag);
        return this.fftshift(freq.real, freq.imag);
    },

    /**
     * Convert frequency-domain spectrum back to time-domain FID.
     * Equivalent to python's `ifftshift` followed by `ifft`.
     */
    to_fid(real, imag) {
        const unshifted = this.ifftshift(real, imag);
        return this.ifft_ortho(unshifted.real, unshifted.imag);
    },

    /**
     * Centered FFT wrapper (ifftshift -> fft -> fftshift)
     */
    fftc(real, imag) {
        const unshifted = this.ifftshift(real, imag);
        const freq = this.fft_ortho(unshifted.real, unshifted.imag);
        return this.fftshift(freq.real, freq.imag);
    },

    /**
     * Centered IFFT wrapper (ifftshift -> ifft -> fftshift)
     */
    ifftc(real, imag) {
        const unshifted = this.ifftshift(real, imag);
        const time = this.ifft_ortho(unshifted.real, unshifted.imag);
        return this.fftshift(time.real, time.imag);
    },

    /**
     * Exponential apodization for line broadening.
     * Multiplies signal by e^(-pi * lb * t).
     * * @param {Float64Array} real - Real part of time domain.
     * @param {Float64Array} imag - Imaginary part of time domain.
     * @param {Float64Array} t - Time coordinates array.
     * @param {number} lb - Line broadening in Hz.
     * @returns {{real: Float64Array, imag: Float64Array}} Apodized arrays.
     */
    apodize_exp(real, imag, t, lb = 1.0) {
        const n = real.length;
        const outReal = new Float64Array(n);
        const outImag = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            const weight = Math.exp(-Math.PI * lb * t[i]);
            outReal[i] = real[i] * weight;
            outImag[i] = imag[i] * weight;
        }

        return { real: outReal, imag: outImag };
    },

    /**
     * Lorentzian-to-Gaussian apodization filter.
     * Multiplies signal by e^(+pi * lb * t) * e^(-t^2 / t_g^2).
     * * @param {Float64Array} real - Real part of time domain.
     * @param {Float64Array} imag - Imaginary part of time domain.
     * @param {Float64Array} t - Time coordinates array.
     * @param {number} lb - Lorentzian line broadening to cancel (Hz).
     * @param {number} gb - Gaussian line broadening to apply (Hz).
     * @returns {{real: Float64Array, imag: Float64Array}} Apodized arrays.
     */
    apodize_lg(real, imag, t, lb = 1.0, gb = 1.0) {
        const n = real.length;
        const outReal = new Float64Array(n);
        const outImag = new Float64Array(n);

        let tg_sq = 0;
        if (gb !== 0) {
            const tg = (2 * Math.sqrt(Math.log(2))) / (Math.PI * gb);
            tg_sq = tg * tg;
        }

        for (let i = 0; i < n; i++) {
            const w_lor = Math.exp(Math.PI * lb * t[i]);
            let w_gauss = 1.0;
            if (gb !== 0) {
                w_gauss = Math.exp(-(t[i] * t[i]) / tg_sq);
            }
            const weight = w_lor * w_gauss;
            
            outReal[i] = real[i] * weight;
            outImag[i] = imag[i] * weight;
        }

        return { real: outReal, imag: outImag };
    }
};
```