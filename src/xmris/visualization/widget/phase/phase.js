export function render({ model, el }) {
    const dpr = window.devicePixelRatio || 1;
    let W = model.get("width");
    let H = model.get("height");

    /* ── DOM ── */
    const root = document.createElement("div");
    root.className = "nmr-viewer";
    root.style.width = W + "px"; // Dynamic sizing

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

    // Floating Legend (Feature f)
    const legend = document.createElement("div");
    legend.className = "nmr-legend";
    legend.innerHTML = "<span class='leg-re'></span> Real &nbsp;&nbsp; <span class='leg-im'></span> Imag";

    canvasContainer.append(canvas, legend);

    /* controls bar */
    const bar   = document.createElement("div"); bar.className = "nmr-bar";

    // Left Group: Controls
    const grpL  = document.createElement("div"); grpL.className = "nmr-grp";

    const p0Lbl = document.createElement("label"); p0Lbl.className = "nmr-lbl"; p0Lbl.textContent = "p0 [°]";
    const p0In  = document.createElement("input"); p0In.type = "number"; p0In.className = "nmr-input";
    p0In.step = "1"; p0In.value = model.get("p0").toFixed(1);

    const p1Lbl = document.createElement("label"); p1Lbl.className = "nmr-lbl"; p1Lbl.textContent = "p1 [°]";
    const p1In  = document.createElement("input"); p1In.type = "number"; p1In.className = "nmr-input";
    p1In.step = "1"; p1In.value = model.get("p1").toFixed(1);

    const resetBtn = document.createElement("button");
    resetBtn.className = "nmr-btn nmr-btn-outline"; resetBtn.textContent = "Reset";

    grpL.append(p0Lbl, p0In, p1Lbl, p1In, resetBtn);

    // Right Group: Hints & Close (Feature c & b)
    const grpR = document.createElement("div"); grpR.className = "nmr-grp";

    const hints = document.createElement("div"); hints.className = "nmr-hints";
    hints.textContent = "Drag: p0 | Shift+Drag: p1";

    const closeBtn = document.createElement("button");
    closeBtn.className = "nmr-btn nmr-btn-outline"; closeBtn.textContent = "Close";
    closeBtn.title = "Close Widget";

    // Updated click handler!
    closeBtn.onclick = () => {
        const p0 = model.get("p0").toFixed(2);
        const p1 = model.get("p1").toFixed(2);
        const pivot = model.get("pivot_val").toFixed(3);

        const codeStr = `phased_da = da.xmr.phase(p0=${p0}, p1=${p1}, pivot=${pivot})`;

        // Clean, semantic HTML referencing our new CSS classes
        root.innerHTML = `
            <div class="nmr-success-banner">
                <div class="nmr-success-title">🎯 Phasing Complete!</div>
                <div class="nmr-success-subtitle">Click the button to copy the code and apply it to your dataset:</div>
                <div class="nmr-copy-container">
                    <div class="nmr-code-block">${codeStr}</div>
                    <button id="nmr-copy-btn" class="nmr-copy-btn">Copy Code</button>
                </div>
            </div>
        `;

        // Wire up the Copy Button
        const copyBtn = root.querySelector("#nmr-copy-btn");
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(codeStr).then(() => {
                // Success animation using CSS classes
                copyBtn.textContent = "Copied! ✓";
                copyBtn.classList.add("copied");

                // Reset after 2 seconds
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

    /* ── stable global Y range based on MAGNITUDE ── */
    let gYMin = -1, gYMax = 1;

    function recomputeY() {
        const mag = model.get("mag");
        if (!mag?.length) return;
        let hi = 0;
        for (let i = 0; i < mag.length; i++) {
            if (mag[i] > hi) hi = mag[i];
        }
        const pad = hi * 0.1 || 1;
        gYMax = hi + pad;
        gYMin = -hi - pad;
    }

    let raf = null;
    function scheduleDraw() { if (!raf) raf = requestAnimationFrame(() => { raf = null; draw(); }); }

    function draw() {
        const P = model.get("x_coords");
        const Re = model.get("reals");
        const Im = model.get("imags");
        const showGrid = model.get("show_grid"); // Fetch grid state
        if (!P?.length || !Re?.length) return;

        const p0_deg = model.get("p0");
        const p1_deg = model.get("p1");
        const pivot = model.get("pivot_val");

        p0In.value = p0_deg.toFixed(1);
        p1In.value = p1_deg.toFixed(1);

        ctx.clearRect(0, 0, W, H);

        const x0 = Math.min(P[0], P[P.length-1]);
        const x1 = Math.max(P[0], P[P.length-1]);
        const x_range = x1 - x0;

        const mg = { t: showGrid ? 22 : 10, r: showGrid ? 22 : 10, b: showGrid ? 44 : 20, l: showGrid ? 64 : 10 };
        const pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;
        const toX = v => mg.l + pw * (x1 - v) / x_range;
        const toY = v => mg.t + ph * (1 - (v - gYMin) / (gYMax - gYMin));

        /* grid & axes (Feature a) */
        if (showGrid) {
            const xt = ticks(x0, x1, 8), yt = ticks(gYMin, gYMax, 6);
            ctx.strokeStyle = "#eee"; ctx.lineWidth = 0.5;
            for (const v of xt) { const x = toX(v); ctx.beginPath(); ctx.moveTo(x, mg.t); ctx.lineTo(x, mg.t+ph); ctx.stroke(); }
            for (const v of yt) { const y = toY(v); ctx.beginPath(); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l+pw, y); ctx.stroke(); }

            ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(mg.l, mg.t); ctx.lineTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph); ctx.stroke();

            ctx.fillStyle = "#666"; ctx.font = "11px sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            for (const v of xt) ctx.fillText(v.toFixed(1), toX(v), mg.t + ph + 5);
            ctx.textAlign = "right"; ctx.textBaseline = "middle";
            for (const v of yt) ctx.fillText(nfmt(v), mg.l - 6, toY(v));

            ctx.fillStyle = "#444"; ctx.font = "12px sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            ctx.fillText(model.get("x_label"), mg.l + pw/2, mg.t + ph + 26);
        } else {
            // Draw minimalist baseline if grid is hidden
            ctx.strokeStyle = "#888"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph); ctx.stroke();
        }

        // Zero line prominently
        ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1;
        const zeroY = toY(0);
        ctx.beginPath(); ctx.moveTo(mg.l, zeroY); ctx.lineTo(mg.l+pw, zeroY); ctx.stroke();

        ctx.save();
        ctx.beginPath(); ctx.rect(mg.l, mg.t, pw, ph); ctx.clip();

        const imPoints = [];
        const rePoints = [];

        const p0_rad = p0_deg * Math.PI / 180.0;
        const p1_rad = p1_deg * Math.PI / 180.0;

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

        ctx.beginPath();
        ctx.strokeStyle = "#e63946"; ctx.lineWidth = 1.0; ctx.globalAlpha = 0.8;
        for (let i=0; i<imPoints.length; i++) { i === 0 ? ctx.moveTo(imPoints[i].x, imPoints[i].y) : ctx.lineTo(imPoints[i].x, imPoints[i].y); }
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = "#0055aa"; ctx.lineWidth = 1.8; ctx.globalAlpha = 1.0;
        for (let i=0; i<rePoints.length; i++) { i === 0 ? ctx.moveTo(rePoints[i].x, rePoints[i].y) : ctx.lineTo(rePoints[i].x, rePoints[i].y); }
        ctx.stroke();

        ctx.restore();

        const pivX = toX(pivot);
        if (pivX >= mg.l && pivX <= mg.l + pw) {
            ctx.fillStyle = "#888"; ctx.beginPath();
            ctx.moveTo(pivX, mg.t); ctx.lineTo(pivX - 5, mg.t - 6); ctx.lineTo(pivX + 5, mg.t - 6); ctx.fill();
        }
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

    /* ── Mouse Drag ── */
    let isDragging = false;
    let startY = 0, startP0 = 0, startP1 = 0;
    let activeMode = 'p0';

    canvas.addEventListener("mousedown", e => {
        isDragging = true;
        startY = e.clientY;
        startP0 = model.get("p0");
        startP1 = model.get("p1");
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

    /* ── Wiring ── */
    const reDraw = scheduleDraw;
    const recompDraw = () => { recomputeY(); scheduleDraw(); };

    model.on("change:p0 change:p1", reDraw);
    model.on("change:reals change:imags change:x_coords change:show_grid", recompDraw);

    // Wire up dimension changes
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

    p0In.addEventListener("change", () => { model.set("p0", parseFloat(p0In.value) || 0); model.save_changes(); });
    p1In.addEventListener("change", () => { model.set("p1", parseFloat(p1In.value) || 0); model.save_changes(); });

    resetBtn.addEventListener("click", () => { model.set("p0", 0); model.set("p1", 0); model.save_changes(); });

    recomputeY();
    scheduleDraw();
}