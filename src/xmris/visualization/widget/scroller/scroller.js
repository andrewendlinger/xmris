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

    const extractBtn = document.createElement("button");
    extractBtn.className = "nmr-btn nmr-btn-outline";
    extractBtn.textContent = "Close";

    extractBtn.onclick = () => {
        const dim = model.get("scroll_dim");
        const idx = model.get("current_index");
        const hintStr = `slice_da = da`;
        const targetStr = `.isel({${dim}: ${idx}})`;

        root.innerHTML = `
            <div class="nmr-success-banner">
                <div class="nmr-success-title">Slice Isolated</div>
                <div class="nmr-success-subtitle">Copy the generated code snippet below to extract index ${idx} along '${dim}':</div>
                <div class="nmr-copy-container">
                    <div class="nmr-code-block">
                        <span class="nmr-code-hint">${hintStr}</span><span class="nmr-code-target">${targetStr}</span>
                    </div>
                    <button id="nmr-copy-btn" class="nmr-copy-btn">Copy Code</button>
                </div>
            </div>
        `;

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
                copyBtn.textContent = "Failed";
                copyBtn.classList.add("failed");
            });
        };
    };

    grpR.append(hints, extractBtn);
    bar.append(grpL, grpR);
    root.append(canvasContainer, tlContainer, bar);
    el.appendChild(root);

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

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

        // NMR standard: higher ppm values are on the left
        const isNMR = model.get("x_label").toLowerCase().includes("ppm");
        const toX = v => isNMR
            ? mg.l + pw * (xMax - v) / (xMax - xMin)
            : mg.l + pw * (v - xMin) / (xMax - xMin);

        const toY = v => mg.t + ph * (1 - (v - gYMin) / (gYMax - gYMin));

        const xt = ticks(xMin, xMax, 8);
        const yt = ticks(gYMin, gYMax, 6);

        /* Axes and Grid */
        ctx.strokeStyle = "#e0e0e0"; ctx.lineWidth = 1.0;
        ctx.beginPath();
        for (const v of xt) { const x = toX(v); ctx.moveTo(x, mg.t); ctx.lineTo(x, mg.t+ph); }
        for (const v of yt) { const y = toY(v); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l+pw, y); }
        ctx.stroke();

        ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mg.l, mg.t); ctx.lineTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph);
        ctx.stroke();

        /* Labels */
        ctx.fillStyle = "#666"; ctx.font = "11px sans-serif";
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

        ctx.fillStyle = "#444"; ctx.font = "12px sans-serif";
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
                drawLine(S[ti], P, toX, toY, xMin, xMax, `rgba(0, 85, 170, ${alpha})`, 1);
            }
        }

        // Active Trace
        drawLine(S[idx], P, toX, toY, xMin, xMax, "#0055aa", 1.8);

        ctx.restore();
    }

    function drawLine(d, P, toX, toY, xMin, xMax, color, width) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
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
    }

    function ticks(lo, hi, n) {
        const r = hi - lo; if (r <= 0) return [lo];
        const raw = r / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const q = raw / mag;
        const step = q < 1.5 ? mag : q < 3.5 ? 2*mag : q < 7.5 ? 5*mag : 10*mag;
        const out = []; let v = Math.ceil(lo / step) * step;
        while (v <= hi + step*1e-9) { out.push(parseFloat(v.toPrecision(12))); v += step; }
        return out;
    }

    function nfmt(n) {
        const a = Math.abs(n);
        if (n === 0) return "0";
        if (a >= 1e4 || (a > 0 && a < .01)) return n.toExponential(1);
        return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(1) : n.toFixed(2);
    }

    /* =========================================================================
       Interactions & Wiring
       ========================================================================= */
    const reDraw = scheduleDraw;
    const recompDraw = () => { recomputeY(); scheduleDraw(); };

    model.on("change:current_index change:show_trace change:trace_count change:xlim change:ylim", reDraw);
    model.on("change:spectra change:x_coords", recompDraw);

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

    recomputeY();
    scheduleDraw();
}