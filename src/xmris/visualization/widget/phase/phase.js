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

    const closeBtn = document.createElement("button");
    closeBtn.className = "nmr-btn nmr-btn-outline"; closeBtn.textContent = "Close";
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