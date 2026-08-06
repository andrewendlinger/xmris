---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (xmris)
  language: python
  name: python3
---

(basics)=
# Basics

Before you can process anything, four things have to stop being surprising: that MR data is
complex-valued, that the FFT has conventions nobody writes down, that a FID and a spectrum are the
same signal, and that the x-axis you actually want is measured in ppm. Every later chapter assumes
all four.

This chapter is the shortest path through them. Each page is a notebook — the plots and numbers you
see were produced by the code above them, on synthetic data you can regenerate.

(basics-pages)=
## The four pages

**Read them in order.** They build: the FFT page assumes complex numbers, the transformation page
assumes the FFT, the ppm page assumes a spectrum exists.

| | Page | What you walk away with |
|---|---|---|
| 1 | [Complex numbers](#complex-numbers) | Splitting and rebuilding complex data with `to_real_imag` / `to_complex`, without losing coordinates or metadata |
| 2 | [FFT basics](#fft) | Why raw `numpy.fft` puts DC at the edge and scales amplitudes oddly, and which xmris pair to reach for instead |
| 3 | [FID & spectrum](#fid-transforms) | The `to_spectrum` / `to_fid` round trip and the `fftshift` ordering behind it |
| 4 | [Hz and ppm](#hz-ppm) | Turning a B₀-dependent Hz axis into the hardware-independent ppm axis everyone quotes |

(basics-next)=
## Where to go next

If you want the design argument rather than the mechanics — why the dimension name carries the
domain, why metadata is guarded — that is [Concepts](#concepts). If you want to start processing
real data, go to the [Processing pipeline](#pipeline).
