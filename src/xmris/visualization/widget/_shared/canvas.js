/**
 * Shared frontend helpers for xmris AnyWidgets.
 *
 * This module intentionally uses NO `import`/`export` statements: the Python
 * asset loader (`_shared/__init__.py::load_esm`) concatenates it *ahead* of a
 * widget's own `<name>.js`, and the widget module owns the single
 * `export function render`. Keeping these helpers export-free lets them live in
 * the same module scope as `render` in both live AnyWidget and the static-docs
 * iframe. See `docs/contributing/static_widgets.md`.
 */

/* =========================================================================
   Axis helpers
   ========================================================================= */

/** Compute visually pleasing "nice" tick positions between `lo` and `hi`. */
function ticks(lo, hi, n) {
    const r = hi - lo;
    if (r <= 0) return [lo];
    const raw = r / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const q = raw / mag;
    const step = q < 1.5 ? mag : q < 3.5 ? 2 * mag : q < 7.5 ? 5 * mag : 10 * mag;
    const out = [];
    let v = Math.ceil(lo / step) * step;
    while (v <= hi + step * 1e-9) {
        out.push(parseFloat(v.toPrecision(12)));
        v += step;
    }
    return out;
}

/** Format a tick label, avoiding overly long decimals / awkward exponents. */
function nfmt(n) {
    const a = Math.abs(n);
    if (n === 0) return "0";
    if (a >= 1e4 || (a > 0 && a < 0.01)) return n.toExponential(1);
    return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(1) : n.toFixed(2);
}

/* =========================================================================
   Canvas / HiDPI helpers
   ========================================================================= */

/** Size a canvas for the current device pixel ratio and return its 2d context. */
function setupCanvas(canvas, w, h, dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return ctx;
}

/** Re-size an existing canvas + context after a width/height change. */
function resizeCanvas(canvas, ctx, w, h, dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
}

/* =========================================================================
   Theming — read the `--nmr-*` CSS variables so <canvas> follows light/dark
   ========================================================================= */

/**
 * Resolve the widget palette from the `--nmr-*` CSS custom properties defined
 * in `_shared/theme.css`. Canvas strokes are drawn with these so the drawing
 * tracks the viewer's light/dark theme (the CSS chrome does so automatically).
 */
function themeColors(el) {
    const cs = getComputedStyle(el);
    const v = (name, fallback) => {
        const raw = cs.getPropertyValue(name).trim();
        return raw || fallback;
    };
    return {
        fg: v("--nmr-fg", "#333"),
        muted: v("--nmr-muted", "#666"),
        label: v("--nmr-label", "#444"),
        grid: v("--nmr-grid", "#e0e0e0"),
        gridSoft: v("--nmr-grid-soft", "#eee"),
        axis: v("--nmr-axis", "#333"),
        zeroLine: v("--nmr-zero-line", "#ccc"),
        pivot: v("--nmr-pivot", "rgba(100,100,100,0.5)"),
        real: v("--nmr-real", "#0055aa"),
        imag: v("--nmr-imag", "#e63946"),
        mag: v("--nmr-mag", "#111"),
        accent: v("--nmr-accent", "#0055aa"),
        envelope: v("--nmr-envelope", "#d97706"),
        origTrace: v("--nmr-orig-trace", "#999"),
    };
}

/** Invoke `cb` whenever the OS light/dark preference flips (for a live redraw). */
function watchTheme(cb) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", cb);
    else if (mq.addListener) mq.addListener(cb); // older browsers
}

/* =========================================================================
   Close / finalize flow — the reproducible-snippet success banner
   ========================================================================= */

/**
 * Replace the widget UI with the completion banner and wire the clipboard copy
 * button. `target` is the reproducible `.xmr.*` / `.isel(...)` snippet; `hint`
 * is the greyed-out prefix (e.g. `phased_da = da`).
 *
 * CONVENTION: the button that triggers this must carry the `remove-me-close-btn`
 * class so `export_widget_static` hides it in the kernel-less static docs. Keep
 * this note if you copy a widget as a template.
 */
function showSnippetBanner(root, { title, subtitle, hint, target }) {
    root.innerHTML = `
        <div class="nmr-success-banner">
            <div class="nmr-success-title">${title}</div>
            <div class="nmr-success-subtitle">${subtitle}</div>
            <div class="nmr-copy-container">
                <div class="nmr-code-block">
                    <span class="nmr-code-hint">${hint}</span><span class="nmr-code-target">${target}</span>
                </div>
                <button class="nmr-copy-btn">Copy Code</button>
            </div>
        </div>
    `;

    const copyBtn = root.querySelector(".nmr-copy-btn");
    copyBtn.onclick = () => {
        navigator.clipboard
            .writeText(target)
            .then(() => {
                copyBtn.textContent = "Copied ✓";
                copyBtn.classList.add("copied");
                setTimeout(() => {
                    copyBtn.textContent = "Copy Code";
                    copyBtn.classList.remove("copied");
                }, 2000);
            })
            .catch((err) => {
                console.error("Failed to copy snippet:", err);
                copyBtn.textContent = "Failed";
                copyBtn.classList.add("failed");
            });
    };
}
