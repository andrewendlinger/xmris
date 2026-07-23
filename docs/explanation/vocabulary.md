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

(vocabulary)=
# The Controlled Vocabulary

The [architecture guide](../notebooks/basics/architecture.md) introduced xmris's *data dictionary*:
one canonical name for every dimension, coordinate, and attribute the package understands —
`reference_frequency`, `time`, `chemical_shift`. Every function reads and writes those exact names.

Which raises the obvious question the moment you bring your own data:

> **My scanner doesn't call it `reference_frequency`. It's `spec_freq`. Now what?**

That single question shapes the whole vocabulary design. Let's follow it.

## Your data speaks a different dialect

MR data arrives under a hundred naming conventions. A Bruker export, a Siemens twix file, a labmate's
script — each has its own word for the spectrometer frequency (`MHz`, `SF`, `TransmitterFrequency`)
and its own name for the dynamic axis (`dyn`, `NR`, `repetitions`). xmris knows exactly one of each.

So when you hand it foreign data, it doesn't guess:

```python
fid.attrs                     # {'spec_freq': 120.3, 'carrier_ppm': 4.7, ...}
fid.xmr.to_ppm()
# ValueError: 'to_ppm' requires the following missing attributes
# in `obj.attrs`: ['reference_frequency'].
#
# To fix this, assign them using standard xarray methods:
#     >>> obj = obj.assign_attrs({'reference_frequency': value})
```

The frequency is *right there* under `spec_freq` — xmris simply refuses to assume `spec_freq` means
what it hopes. Fair enough. But how should it let you fix that?

## The tempting answer (and why we didn't)

The friendly-looking move is to teach xmris your names: let each term carry a little table of aliases
(`spec_freq`, `MHz`, `SF` → `reference_frequency`), consulted whenever the package reads an attribute.
Your data would "just work," untouched.

We built exactly that — starting with our own history, an old key `MHz` we'd long since renamed to
`reference_frequency`. Then we watched it break.

The trouble isn't the idea; it's *where the tolerance lives*. An alias only helps a reader that
remembers to look it up. Miss one, and the same data behaves two ways:

```python
fid.attrs                     # {'MHz': 120.3, ...}   ← legacy alias for reference_frequency

fid.xmr.fit_amares(pk)        # ✅ worked — this reader checked the alias
fid.xmr.to_ppm()              # ❌ ValueError — this one didn't
```

Same array, same key, opposite outcomes — decided by an implementation detail no user can see. And
that's the *forgiving* case, where we only missed one spot. The deeper problem is that there is no
finite alias list to finish: `spec_freq`, `SF`, `sfrq`, `TransmitterFrequency`, the next vendor's
spelling… A controlled vocabulary that also tries to accept every *un*controlled one isn't controlled
at all.

❌ **The road not taken:** bend the vocabulary to fit the data.

## The xmris way: bend the data to fit the vocabulary

So we flipped it. The vocabulary stays fixed and small; *you* translate your data onto it, once, at
the top of your analysis:

```python
fid = fid.rename({"dyn": "repetition"})                 # dimensions & coordinates
fid = fid.assign_attrs(reference_frequency=fid.attrs.pop("spec_freq"))  # attributes
```

One tension, gone for good. From that line on, your data *is* xmris data: every function sees
canonical names, so every function behaves identically. There is no reader left to forget, because
there is nothing foreign left to tolerate.

✅ **The rule:** move the data to the vocabulary, not the vocabulary to the data.

:::{note}
That rename is a little boilerplate today. A planned `da.xmr.map_vocab(...)` helper will do it in one
validated call — `fid.xmr.map_vocab(repetition="dyn", reference_frequency="spec_freq")` — checking
your targets against the real vocabulary as it goes so a typo fails loudly instead of silently. *(Not
yet shipped.)*
:::

Why not the *other* flip — a `set_vocab()` that makes xmris answer to your names for the whole
session? Because that is the alias problem wearing a hat: the tolerance becomes global mutable state,
two datasets with different conventions can no longer coexist, and *"what does `reference_frequency`
resolve to right now?"* becomes a live question again. Renaming keeps the answer boring and permanent.

(the-lowercase-convention)=
### What you're conforming to: the lowercase convention

The canonical names are all **lowercase `snake_case`**, deliberately aligned with the wider xarray
ecosystem rather than invented from scratch:

| Standard / package | Convention |
| --- | --- |
| [CF Conventions](https://cfconventions.org/) | `time`, `latitude`, `longitude` |
| [cf-xarray](https://cf-xarray.readthedocs.io/) | `time`, `latitude`, `vertical` |
| xarray docs & tutorials | `time`, `x`, `y`, `space` |
| **xmris** | `time`, `frequency`, `chemical_shift` |

`snake_case` also stays unambiguous for multi-word names — `chemical_shift` reads one way, where
`Chemical_Shift` is a hybrid no Python convention endorses. You're free to name *your* axes anything
you like (every function takes a `dim=` argument); it's only when xmris creates a name itself — the
`chemical_shift` coordinate from `to_ppm()`, say — that it is guaranteed lowercase.

## Why the vocabulary can afford to be this strict

All of this only works because the vocabulary is a *fixed point* you can trust. So we make it one,
structurally:

```python
from xmris.core import ATTRS

ATTRS.reference_frequency.unit = "kHz"     # AttributeError — terms are frozen
```

Terms are immutable, and the package refuses to import if two of them ever claim the same string (a
duplicate is caught at startup, not three hours into a run). The single source of truth genuinely
stays single — which is what earns you the right to conform your data to it once and never look back.

:::{dropdown} Aside — why are these `str` objects and not an `Enum`?
`ATTRS.reference_frequency` is a `str` *subclass*: it **is** the string `"reference_frequency"`, so it
drops straight into `da.attrs[...]`, and it also carries `.unit` and a description that feed
coordinate metadata and the auto-generated docstrings.

The known trap with this trick is that string operations "evaporate" the subclass —
`ATTRS.reference_frequency + "_x"` is a plain `str` with no `.unit`. It doesn't bite xmris because we
never read metadata off a string *flowing through a pipeline*; we read it off the imported constant,
and the one place that consumes `.unit`/`.long_name` (`as_variable`, when building a coordinate)
always has the real term in hand. Everywhere else compares terms *by value*, exactly like plain
strings.

The alternatives each cost more than they save here: a `StrEnum` re-hydrates cleanly but can't be
extended and renders awkwardly in signatures; plain constants plus a separate metadata registry split
one concept across two places; `pint` is the right tool for *unit math* but the wrong one for
*naming*. A frozen `str` subclass, kept honest by the immutability above, is the least machinery that
does the job. (Full deliberation: [issue #65](https://github.com/andrewendlinger/xmris/issues/65).)
:::

## Adding a word

Extending xmris and need a name that isn't there yet? Add it to `xmris.core.config` — never reach for
a bare string in package code (see [The Architecture Contract](../contributing/contract.md)). One new
`XmrisTerm`, and the whole package can speak the new word — with the freeze and the uniqueness check
keeping it honest from the moment it exists.
