/**
 * Rendering function for the AnyWidget spectra scroller.
 *
 * Shared helpers (`ticks`, `nfmt`, `setupCanvas`, `resizeCanvas`, `themeColors`,
 * `watchTheme`, `showSnippetBanner`) come from `_shared/canvas.js`, which the
 * Python asset loader concatenates ahead of this module.
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
    canvas.className = "nmr-canvas";
    canvas.tabIndex = 0;
    canvasContainer.append(canvas);

    /* Timeline Scrubber */
    const tlContainer = document.createElement("div");
    tlContainer.className = "nmr-timeline-container";

    const playBtn = document.createElement("button");
    playBtn.className = "nmr-btn nmr-btn-outline nmr-play-btn";
    playBtn.textContent = "▶";
    playBtn.title = "Play / Pause";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "nmr-slider";
    slider.min = "0";
    // Max will be set dynamically based on data length

    const idxDisplay = document.createElement("div");
    idxDisplay.className = "nmr-index-display";

    tlContainer.append(playBtn, slider, idxDisplay);

    /* Control Bar Elements */
    const bar = document.createElement("div");
    bar.className = "nmr-bar";

    const grpL = document.createElement("div");
    grpL.className = "nmr-grp";

    const traceLbl = document.createElement("label");
    traceLbl.className = "nmr-lbl";

    const traceChk = document.createElement("input");
    traceChk.type = "checkbox";
    traceChk.checked = model.get("show_trace");
    traceLbl.append(traceChk, " History Trails");

    const depthLbl = document.createElement("label");
    depthLbl.className = "nmr-lbl";
    depthLbl.textContent = "Depth:";

    const depthIn = document.createElement("input");
    depthIn.type = "number";
    depthIn.className = "nmr-input";
    depthIn.min = "0";
    depthIn.value = model.get("trace_count");

    grpL.append(traceLbl, depthLbl, depthIn);

    const grpR = document.createElement("div");
    grpR.className = "nmr-grp";

    const hints = document.createElement("div");
    hints.className = "nmr-hints";
    hints.textContent = "Scroll to browse | Space to play";

    // CONVENTION: Always add the 'remove-me-close-btn' class to buttons that finalize,
    // close, or require a live Jupyter kernel. This allows the static documentation
    // exporter to automatically hide them when rendered in a standalone HTML iframe.
    // Also, keep this comment if you take this code as reference for a new widget.
    const closeBtn = document.createElement("button");
    closeBtn.className = "nmr-btn nmr-btn-outline remove-me-close-btn";
    closeBtn.textContent = "Extract Slice";
    closeBtn.title = "Emit the .isel() snippet for the current index";

    closeBtn.onclick = () => {
        const dim = model.get("scroll_dim");
        const idx = model.get("current_index");
        showSnippetBanner(root, {
            title: "Slice Isolated",
            subtitle: `Copy the generated code snippet below to extract index ${idx} along '${dim}':`,
            hint: "slice_da = da",
            target: `.isel({${dim}: ${idx}})`,
        });
    };

    grpR.append(hints, closeBtn);
    bar.append(grpL, grpR);
    root.append(canvasContainer, tlContainer, bar);
    el.appendChild(root);

    const ctx = setupCanvas(canvas, W, H, dpr);

    /* =========================================================================
       Canvas Drawing & Math
       ========================================================================= */
    let gYMin = 0, gYMax = 1;

    function recomputeY() {
        const S = model.get("spectra");
        const userY = model.get("ylim");

        if (userY && userY.length === 2) {
            gYMin = Math.min(userY[0], userY[1]);
            gYMax = Math.max(userY[0], userY[1]);
            return;
        }

        if (!S || !S.length) return;

        let min = Infinity, max = -Infinity;
        for (const row of S) {
            for (const val of row) {
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        const pad = (max - min) * 0.1 || 1.0;
        gYMax = max + pad;
        gYMin = min - pad;
    }

    let raf = null;
    function scheduleDraw() {
        if (!raf) raf = requestAnimationFrame(() => { raf = null; draw(); });
    }

    function draw() {
        const P = model.get("x_coords");
        const S = model.get("spectra");
        if (!P?.length || !S?.length) return;

        const C = themeColors(root);

        const N = S.length;
        const idx = model.get("current_index");
        const doTrace = model.get("show_trace");
        const nTrace = model.get("trace_count");
        const userX = model.get("xlim");

        // Sync UI
        slider.max = N - 1;
        slider.value = idx;
        idxDisplay.textContent = `${idx} / ${N - 1}`;
        depthIn.max = N;

        ctx.clearRect(0, 0, W, H);

        let x0 = P[0], x1 = P[P.length - 1];
        if (userX && userX.length === 2) {
            x0 = userX[0];
            x1 = userX[1];
        }

        // Standardize min/max for math
        const xMin = Math.min(x0, x1);
        const xMax = Math.max(x0, x1);

        const mg = { t: 22, r: 22, b: 44, l: 64 };
        const pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;

        // Spectral axis: NMR convention plots higher values on the left.
        const toX = v => mg.l + pw * (xMax - v) / (xMax - xMin);
        const toY = v => mg.t + ph * (1 - (v - gYMin) / (gYMax - gYMin));

        const xt = ticks(xMin, xMax, 8);
        const yt = ticks(gYMin, gYMax, 6);

        /* Axes and Grid */
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1.0;
        ctx.beginPath();
        for (const v of xt) { const x = toX(v); ctx.moveTo(x, mg.t); ctx.lineTo(x, mg.t+ph); }
        for (const v of yt) { const y = toY(v); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l+pw, y); }
        ctx.stroke();

        ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mg.l, mg.t); ctx.lineTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph);
        ctx.stroke();

        /* Labels */
        ctx.fillStyle = C.muted; ctx.font = "11px sans-serif";
        ctx.beginPath(); ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (const v of xt) {
            const x = toX(v);
            ctx.moveTo(x, mg.t + ph); ctx.lineTo(x, mg.t + ph + 5);
            ctx.fillText(v.toFixed(1), x, mg.t + ph + 8);
        }
        ctx.stroke();

        ctx.beginPath(); ctx.textAlign = "right"; ctx.textBaseline = "middle";
        for (const v of yt) {
            const y = toY(v);
            ctx.moveTo(mg.l, y); ctx.lineTo(mg.l - 5, y);
            ctx.fillText(nfmt(v), mg.l - 8, y);
        }
        ctx.stroke();

        ctx.fillStyle = C.label; ctx.font = "12px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(model.get("x_label"), mg.l + pw/2, mg.t + ph + 28);

        /* Traces */
        ctx.save();
        ctx.beginPath(); ctx.rect(mg.l, mg.t, pw, ph); ctx.clip();

        if (doTrace && nTrace > 0) {
            // Cap history to available data
            const maxK = Math.min(nTrace, idx);
            for (let k = maxK; k >= 1; k--) {
                const ti = idx - k;
                const alpha = 0.5 * (1 - (k - 1) / nTrace); // Fade out older traces
                drawLine(S[ti], P, toX, toY, xMin, xMax, C.accent, 1, alpha);
            }
        }

        // Active Trace
        drawLine(S[idx], P, toX, toY, xMin, xMax, C.accent, 1.8, 1.0);

        ctx.restore();
    }

    function drawLine(d, P, toX, toY, xMin, xMax, color, width, alpha) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.globalAlpha = alpha === undefined ? 1.0 : alpha;
        let isStarted = false;

        for (let i = 0; i < P.length; i++) {
            if (P[i] < xMin || P[i] > xMax) continue;
            const x = toX(P[i]), y = toY(d[i]);
            if (isStarted) {
                ctx.lineTo(x, y);
            } else {
                ctx.moveTo(x, y);
                isStarted = true;
            }
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }

    /* =========================================================================
       Interactions & Wiring
       ========================================================================= */
    const reDraw = scheduleDraw;
    const recompDraw = () => { recomputeY(); scheduleDraw(); };

    model.on("change:current_index change:show_trace change:trace_count change:xlim change:ylim", reDraw);
    model.on("change:spectra change:x_coords", recompDraw);

    // Responsive resizing
    model.on("change:width change:height", () => {
        W = model.get("width");
        H = model.get("height");
        root.style.width = W + "px";
        canvasContainer.style.width = W + "px";
        canvasContainer.style.height = H + "px";
        resizeCanvas(canvas, ctx, W, H, dpr);
        scheduleDraw();
    });

    // Inputs
    slider.addEventListener("input", () => { model.set("current_index", parseInt(slider.value)); model.save_changes(); });
    traceChk.addEventListener("change", () => { model.set("show_trace", traceChk.checked); model.save_changes(); });
    depthIn.addEventListener("change", () => {
        let val = parseInt(depthIn.value) || 0;
        model.set("trace_count", val);
        model.save_changes();
    });

    // Scroll Wheel
    canvas.addEventListener("wheel", e => {
        e.preventDefault();
        const N = model.get("spectra")?.length || 1;
        let i = model.get("current_index") + (e.deltaY > 0 ? 1 : -1);
        model.set("current_index", Math.max(0, Math.min(i, N - 1)));
        model.save_changes();
    }, { passive: false });

    // Keyboard Navigation
    canvas.addEventListener("keydown", e => {
        const N = model.get("spectra")?.length || 1;
        let i = model.get("current_index");
        switch (e.key) {
            case "ArrowRight": case "ArrowDown": i = Math.min(i + 1, N - 1); break;
            case "ArrowLeft":  case "ArrowUp":   i = Math.max(i - 1, 0); break;
            case "Home": i = 0; break;
            case "End":  i = N - 1; break;
            case " ": playBtn.click(); e.preventDefault(); return;
            default: return;
        }
        e.preventDefault();
        model.set("current_index", i);
        model.save_changes();
    });

    // Play/Pause Animation
    let playing = false, tmr = null;
    playBtn.addEventListener("click", () => {
        playing = !playing;
        playBtn.textContent = playing ? "||" : "▶";
        if (playing) {
            const ms = 150; // Milliseconds per frame
            tmr = setInterval(() => {
                const N = model.get("spectra")?.length; if (!N) return;
                let i = model.get("current_index") + 1;
                if (i >= N) i = 0; // Loop back
                model.set("current_index", i);
                model.save_changes();
            }, ms);
        } else {
            clearInterval(tmr);
            tmr = null;
        }
    });

    // Redraw when the OS light/dark preference flips
    watchTheme(scheduleDraw);

    recomputeY();
    scheduleDraw();
}
