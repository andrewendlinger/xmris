export const MRSDSP = {
    _roll(arr, shift) {
        const n = arr.length;
        const res = new Float64Array(n);
        for (let i = 0; i < n; i++) res[(i + shift) % n] = arr[i];
        return res;
    },
    fftshift(real, imag) {
        const shift = Math.floor(real.length / 2);
        return { real: this._roll(real, shift), imag: this._roll(imag, shift) };
    },
    ifftshift(real, imag) {
        const shift = Math.floor((real.length + 1) / 2);
        return { real: this._roll(real, shift), imag: this._roll(imag, shift) };
    },
    _fft_radix2(real_in, imag_in, isInverse = false) {
        const n = real_in.length;
        const real = new Float64Array(real_in);
        const imag = new Float64Array(imag_in);
        let j = 0;
        for (let i = 0; i < n - 1; i++) {
            if (i < j) {
                let tr = real[i], ti = imag[i];
                real[i] = real[j]; imag[i] = imag[j];
                real[j] = tr; imag[j] = ti;
            }
            let m = n >> 1;
            while (m >= 1 && j >= m) { j -= m; m >>= 1; }
            j += m;
        }
        for (let step = 1; step < n; step <<= 1) {
            const jump = step << 1;
            const theta = (isInverse ? Math.PI : -Math.PI) / step;
            for (let i = 0; i < step; i++) {
                let u_r = 1.0, u_i = 0.0;
                if (i > 0) { const angle = i * theta; u_r = Math.cos(angle); u_i = Math.sin(angle); }
                for (let k = i; k < n; k += jump) {
                    const match = k + step;
                    const tr = u_r * real[match] - u_i * imag[match];
                    const ti = u_r * imag[match] + u_i * real[match];
                    real[match] = real[k] - tr; imag[match] = imag[k] - ti;
                    real[k] += tr; imag[k] += ti;
                }
            }
        }
        const norm = 1.0 / Math.sqrt(n);
        for (let i = 0; i < n; i++) { real[i] *= norm; imag[i] *= norm; }
        return { real, imag };
    },
    fft_ortho(real, imag) { return this._fft_radix2(real, imag, false); },
    to_spectrum(real, imag) {
        const freq = this.fft_ortho(real, imag);
        return this.fftshift(freq.real, freq.imag);
    },
    apodize_exp(real, imag, t, lb = 1.0) {
        const n = real.length;
        const outReal = new Float64Array(n), outImag = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const weight = Math.exp(-Math.PI * lb * t[i]);
            outReal[i] = real[i] * weight; outImag[i] = imag[i] * weight;
        }
        return { real: outReal, imag: outImag };
    },
    apodize_lg(real, imag, t, lb = 1.0, gb = 1.0) {
        const n = real.length;
        const outReal = new Float64Array(n), outImag = new Float64Array(n);
        let tg_sq = gb !== 0 ? Math.pow((2 * Math.sqrt(Math.log(2))) / (Math.PI * gb), 2) : 0;
        for (let i = 0; i < n; i++) {
            const w_lor = Math.exp(Math.PI * lb * t[i]);
            const w_gauss = gb !== 0 ? Math.exp(-(t[i] * t[i]) / tg_sq) : 1.0;
            const weight = w_lor * w_gauss;
            outReal[i] = real[i] * weight; outImag[i] = imag[i] * weight;
        }
        return { real: outReal, imag: outImag };
    }
};

export function render({ model, el }) {
    const dpr = window.devicePixelRatio || 1;
    let W = model.get("width");
    let H = model.get("height");

    const root = document.createElement("div");
    root.className = "nmr-viewer";
    root.style.width = W + "px";

    const H_top = Math.floor(H * 0.35);
    const H_bot = Math.floor(H * 0.65);

    const canvasContainer = document.createElement("div");
    canvasContainer.className = "nmr-canvas-container";
    canvasContainer.style.width = W + "px";

    // Top Canvas (FID)
    const topCanvas = document.createElement("canvas");
    topCanvas.className = "nmr-canvas top-canvas";
    topCanvas.width = W * dpr; topCanvas.height = H_top * dpr;
    topCanvas.style.width = W + "px"; topCanvas.style.height = H_top + "px";

    // Bottom Canvas (Spectrum)
    const botCanvas = document.createElement("canvas");
    botCanvas.className = "nmr-canvas bot-canvas";
    botCanvas.width = W * dpr; botCanvas.height = H_bot * dpr;
    botCanvas.style.width = W + "px"; botCanvas.style.height = H_bot + "px";

    canvasContainer.append(topCanvas, botCanvas);

    // Dynamic extraction of configured limits
    const lbMin = model.get("lb_min"); const lbMax = model.get("lb_max");
    const gbMin = model.get("gb_min"); const gbMax = model.get("gb_max");

    // Control Bar UI
    const bar = document.createElement("div");
    bar.className = "nmr-bar";

    bar.innerHTML = `
        <div class="nmr-bar-row">
            <div class="nmr-slider-grp">
                <span class="nmr-lbl">LB [Hz]:</span>
                <input type="range" id="sld-lb" class="nmr-slider" min="${lbMin}" max="${lbMax}" step="0.1" value="0">
                <span id="val-lb" class="nmr-val">0.0</span>
            </div>
            <div class="nmr-slider-grp" id="grp-gb" style="display:none;">
                <span class="nmr-lbl">GB [Hz]:</span>
                <input type="range" id="sld-gb" class="nmr-slider" min="${gbMin}" max="${gbMax}" step="0.1" value="0">
                <span id="val-gb" class="nmr-val">0.0</span>
            </div>
        </div>
        <div class="nmr-bar-row">
            <div class="nmr-grp">
                <label class="nmr-lbl">Method:</label>
                <select id="sel-method" class="nmr-select">
                    <option value="exp">Exponential</option>
                    <option value="lg">Lorentz-Gauss</option>
                </select>
                <label class="nmr-lbl">Display:</label>
                <select id="sel-mode" class="nmr-select">
                    <option value="real">Real</option>
                    <option value="imag">Imaginary</option>
                    <option value="mag">Magnitude</option>
                </select>
                <label class="nmr-lbl" style="margin-left:8px;">
                    <input type="checkbox" id="chk-orig" style="vertical-align:middle;"> Show Original
                </label>
            </div>
            <div class="nmr-grp">
                <button id="btn-close" class="nmr-btn nmr-btn-outline remove-me-close-btn">Close</button>
            </div>
        </div>
    `;

    root.append(canvasContainer, bar);
    el.appendChild(root);

    const selMethod = root.querySelector("#sel-method");
    const selMode = root.querySelector("#sel-mode");
    const chkOrig = root.querySelector("#chk-orig");
    const sldLb = root.querySelector("#sld-lb");
    const valLb = root.querySelector("#val-lb");
    const sldGb = root.querySelector("#sld-gb");
    const valGb = root.querySelector("#val-gb");
    const grpGb = root.querySelector("#grp-gb");
    const btnClose = root.querySelector("#btn-close");

    const ctxT = topCanvas.getContext("2d"); ctxT.scale(dpr, dpr);
    const ctxB = botCanvas.getContext("2d"); ctxB.scale(dpr, dpr);

    let cached_orig_spec = null;

    function getTraceColor(mode) {
        if (mode === 'real') return "#0055aa";
        if (mode === 'imag') return "#e63946";
        return "#111111"; // Magnitude
    }

    function calcMag(re, im) {
        const n = re.length; const mag = new Float64Array(n);
        for(let i=0; i<n; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
        return mag;
    }

    function recomputeYBounds(arr, padRatio = 0.15) {
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] < min) min = arr[i];
            if (arr[i] > max) max = arr[i];
        }
        const pad = (max - min) * padRatio || 1.0;
        return { min: min - pad, max: max + pad };
    }

    let raf = null;
    function scheduleDraw() { if (!raf) raf = requestAnimationFrame(() => { raf = null; draw(); }); }

    function draw() {
        const P_t = model.get("t_coords");
        const P_x = model.get("x_coords");
        const re_t = model.get("reals_t");
        const im_t = model.get("imags_t");
        if (!P_t?.length || !re_t?.length) return;

        const N = P_t.length;
        const lb = model.get("lb");
        const gb = model.get("gb");
        const method = model.get("method");
        const mode = model.get("display_mode");
        const showOrig = model.get("show_orig");

        // UI sync
        selMethod.value = method;
        selMode.value = mode;
        chkOrig.checked = showOrig;
        sldLb.value = lb; valLb.textContent = lb.toFixed(1);
        sldGb.value = gb; valGb.textContent = gb.toFixed(1);
        grpGb.style.display = method === "lg" ? "flex" : "none";

        // 1. Math Execution
        if (!cached_orig_spec) cached_orig_spec = MRSDSP.to_spectrum(re_t, im_t);

        let apo_fid;
        let envelope = new Float64Array(N);
        let maxEnv = 0.0;

        if (method === 'exp') {
            apo_fid = MRSDSP.apodize_exp(re_t, im_t, P_t, lb);
            for(let i=0; i<N; i++) {
                envelope[i] = Math.exp(-Math.PI * lb * P_t[i]);
                if (envelope[i] > maxEnv) maxEnv = envelope[i];
            }
        } else {
            apo_fid = MRSDSP.apodize_lg(re_t, im_t, P_t, lb, gb);
            let tg_sq = gb !== 0 ? Math.pow((2 * Math.sqrt(Math.log(2))) / (Math.PI * gb), 2) : 0;
            for(let i=0; i<N; i++) {
                let w_lor = Math.exp(Math.PI * lb * P_t[i]);
                let w_gauss = gb !== 0 ? Math.exp(-(P_t[i]*P_t[i]) / tg_sq) : 1.0;
                envelope[i] = w_lor * w_gauss;
                if (envelope[i] > maxEnv) maxEnv = envelope[i];
            }
        }
        let apo_spec = MRSDSP.to_spectrum(apo_fid.real, apo_fid.imag);

        // Data arrays to render
        let fid_disp, fid_orig, spec_disp, spec_orig;

        const fidMode = mode === 'imag' ? 'imag' : 'real'; // FID defaults to real if mag selected
        if (fidMode === 'real') {
            fid_disp = apo_fid.real; fid_orig = re_t;
        } else {
            fid_disp = apo_fid.imag; fid_orig = im_t;
        }

        if (mode === 'real') {
            spec_disp = apo_spec.real; spec_orig = cached_orig_spec.real;
        } else if (mode === 'imag') {
            spec_disp = apo_spec.imag; spec_orig = cached_orig_spec.imag;
        } else {
            spec_disp = calcMag(apo_spec.real, apo_spec.imag);
            spec_orig = calcMag(cached_orig_spec.real, cached_orig_spec.imag);
        }

        const yT = recomputeYBounds(fid_orig, 0.1);

        // Dynamically expand spectrum Y axis ONLY if bounds are exceeded
        const yOrig = recomputeYBounds(spec_orig, 0.15);
        const yDisp = recomputeYBounds(spec_disp, 0.15);
        const yB = {
            min: Math.min(yOrig.min, yDisp.min),
            max: Math.max(yOrig.max, yDisp.max)
        };

        // 2. Rendering Helper
        function renderAxes(ctx, w, h, mg, xRange, yRange, invX, xLabel, xticks, yticks) {
            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
            const pw = w - mg.l - mg.r, ph = h - mg.t - mg.b;

            if (model.get("show_grid")) {
                ctx.strokeStyle = "#eee"; ctx.beginPath();
                for (const v of xticks) { const x = invX ? toXr(v) : toX(v); ctx.moveTo(x, mg.t); ctx.lineTo(x, mg.t+ph); }
                for (const v of yticks) { const y = toY(v); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l+pw, y); }
                ctx.stroke();
            }

            ctx.strokeStyle = "#333";
            ctx.beginPath(); ctx.moveTo(mg.l, mg.t); ctx.lineTo(mg.l, mg.t+ph); ctx.lineTo(mg.l+pw, mg.t+ph); ctx.stroke();

            ctx.fillStyle = "#666"; ctx.font = "10px sans-serif";
            ctx.beginPath(); ctx.textAlign = "center"; ctx.textBaseline = "top";
            for (const v of xticks) {
                const x = invX ? toXr(v) : toX(v);
                ctx.moveTo(x, mg.t + ph); ctx.lineTo(x, mg.t + ph + 4);
                ctx.fillText(nfmt(v), x, mg.t + ph + 6);
            }
            ctx.stroke();

            ctx.beginPath(); ctx.textAlign = "right"; ctx.textBaseline = "middle";
            for (const v of yticks) {
                const y = toY(v); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l - 4, y); ctx.fillText(nfmt(v), mg.l - 6, y);
            }
            ctx.stroke();

            ctx.fillStyle = "#444"; ctx.font = "11px sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "top";
            ctx.fillText(xLabel, mg.l + pw/2, mg.t + ph + 22);

            function toX(v) { return mg.l + pw * (v - xRange[0]) / (xRange[1] - xRange[0]); }
            function toXr(v) { return mg.l + pw * (xRange[1] - v) / (xRange[1] - xRange[0]); }
            function toY(v) { return mg.t + ph * (1 - (v - yRange.min) / (yRange.max - yRange.min)); }
            return { toX, toXr, toY, pw, ph };
        }

        function drawTrace(ctx, P, Y, mappingX, mappingY, color, width, alpha) {
            ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
            for (let i=0; i<P.length; i+=Math.ceil(P.length/2000)) {
                i === 0 ? ctx.moveTo(mappingX(P[i]), mappingY(Y[i])) : ctx.lineTo(mappingX(P[i]), mappingY(Y[i]));
            }
            ctx.stroke(); ctx.globalAlpha = 1.0;
        }

        // --- Render Top Canvas (FID) ---
        const mgT = { t: 15, r: 20, b: 40, l: 60 };
        const xtT = ticks(P_t[0], P_t[P_t.length-1], 8);
        const ytT = ticks(yT.min, yT.max, 4);
        const axT = renderAxes(ctxT, W, H_top, mgT, [P_t[0], P_t[P_t.length-1]], yT, false, "Time [s]", xtT, ytT);

        ctxT.save(); ctxT.beginPath(); ctxT.rect(mgT.l, mgT.t, axT.pw, axT.ph); ctxT.clip();
        if (showOrig) drawTrace(ctxT, P_t, fid_orig, axT.toX, axT.toY, "#999", 1.5, 0.5);
        drawTrace(ctxT, P_t, fid_disp, axT.toX, axT.toY, getTraceColor(fidMode), 1.5, 1.0);

        // Envelope normalized scaling to FID maximum limits
        const maxFid = Math.max(Math.abs(yT.max), Math.abs(yT.min));
        if (maxEnv === 0) maxEnv = 1.0; // Fail-safe division
        const envScaled = new Float64Array(N);
        for(let i=0; i<N; i++) envScaled[i] = (envelope[i] / maxEnv) * maxFid;

        ctxT.setLineDash([4, 4]);
        drawTrace(ctxT, P_t, envScaled, axT.toX, axT.toY, "#d97706", 2.0, 0.85);
        ctxT.restore();

        // --- Render Bottom Canvas (Spectrum) ---
        const mgB = { t: 15, r: 20, b: 40, l: 60 };
        const xtB = ticks(P_x[0], P_x[P_x.length-1], 8);
        const ytB = ticks(yB.min, yB.max, 5);
        const axB = renderAxes(ctxB, W, H_bot, mgB, [P_x[0], P_x[P_x.length-1]], yB, true, model.get("x_label"), xtB, ytB);

        ctxB.save(); ctxB.beginPath(); ctxB.rect(mgB.l, mgB.t, axB.pw, axB.ph); ctxB.clip();
        if (showOrig) drawTrace(ctxB, P_x, spec_orig, axB.toXr, axB.toY, "#999", 1.5, 0.5);
        drawTrace(ctxB, P_x, spec_disp, axB.toXr, axB.toY, getTraceColor(mode), 1.5, 1.0);
        ctxB.restore();
    }

    function ticks(lo, hi, n) {
        const r = hi - lo; if (r <= 0) return [lo];
        const raw = r / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const q = raw / mag, step = q < 1.5 ? mag : q < 3.5 ? 2*mag : q < 7.5 ? 5*mag : 10*mag;
        const out = []; let v = Math.ceil(lo / step) * step;
        while (v <= hi + step*1e-9) { out.push(parseFloat(v.toPrecision(12))); v += step; }
        return out;
    }
    function nfmt(n) {
        const a = Math.abs(n); if (n === 0) return "0";
        return (a >= 1e4 || (a > 0 && a < .001)) ? n.toExponential(1) : a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(1) : n.toFixed(2);
    }

    // 3. Observers & Wiring
    selMethod.onchange = e => { model.set("method", e.target.value); model.set("lb", 0); model.set("gb", 0); model.save_changes(); };
    selMode.onchange = e => { model.set("display_mode", e.target.value); model.save_changes(); };
    chkOrig.onchange = e => { model.set("show_orig", e.target.checked); model.save_changes(); };
    sldLb.oninput = e => { model.set("lb", parseFloat(e.target.value)); model.save_changes(); };
    sldGb.oninput = e => { model.set("gb", parseFloat(e.target.value)); model.save_changes(); };

    btnClose.onclick = () => {
        const method = model.get("method");
        const lb = model.get("lb").toFixed(2);
        const gb = model.get("gb").toFixed(2);

        const hintStr = `da_apodized = da`;
        const targetStr = method === "exp" ? `.xmr.apodize_exp(lb=${lb})` : `.xmr.apodize_lg(lb=${lb}, gb=${gb})`;

        root.innerHTML = `
            <div class="nmr-success-banner">
                <div class="nmr-success-title">Apodization Parameters Extracted</div>
                <div class="nmr-success-subtitle">Copy the generated code snippet below to apply this filter to your dataset:</div>
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
                copyBtn.textContent = "Copied ✓"; copyBtn.classList.add("copied");
                setTimeout(() => { copyBtn.textContent = "Copy Code"; copyBtn.classList.remove("copied"); }, 2000);
            });
        };
    };

    model.on("change:lb change:gb change:method change:display_mode change:show_orig", scheduleDraw);
    model.on("change:reals_t", () => { cached_orig_spec = null; scheduleDraw(); });

    scheduleDraw();
}