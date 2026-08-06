(diary-about)=
# A dev diary for xmris

Sooner or later you will stumble upon a line of xmris code and wonder *why on earth is this designed this way* — why the vocabulary is locked down, why a function funnels into a
single domain, why the accessor is split across four mixins. The code shows you
*what* it does; it almost never shows you the argument that produced it. That
argument is what this diary aims to keep.

:::{image} ../assets/dev-diary-meme.jpg
:alt: Meme — "me, 5 minutes into ADDING A FEATURE: reading my old diaries" — someone engrossed in an old book instead of the task at hand.
:width: 300px
:align: center
:::

So if you are chasing the reasoning behind a design decision, you are in the
right place! Skim the entries below, or search for the thing that puzzled you.
Each entry is one decision, told as a story — and when a decision evolves, its
entry is rewritten in place rather than joined by a sequel. The muted *Last
edited* line under each title tells you the story is current.

(diary-about-how)=
## How an entry gets written

The entries are a by-product of how xmris is actually built. A significant change usually starts as a planning session — nowadays often with e.g. Claude Code — that ends in a precise, ordered plan. That plan is exactly
right for *doing* the work and sometimes not so good for *judging* it: twenty mechanical
steps with the one real decision buried underneath them.

So before any code is written, the plan is distilled into a short **draft
entry** — one screen: the tension, the decision, a diagram, and often a code snippet or two — a small user story sketching
the call we wish existed. That draft is what gets reviewed — and the work stops
until it has been.
Writing it this early, forcing the argument onto a single page helps catching bad decision early.

> It's almost like a docs-first approach.

Then the work happens, and reality argues back. So once the change lands, the
draft is **reconciled into the story you are reading**: corrections wherever
the plan turned out wrong, the real file paths and code snippets dropped in so
you can see how it works under the hood, and a straight account of why — having
now built it — we think the call was right.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    P["Plan"] --> D["Draft the entry"]
    D --> R["Review the shape"]
    R --> B["Build it"]
    B --> S["Reconcile into a story"]
```

:::{seealso}
Writing one yourself? [Write a dev-diary entry](#contribute-dev-diary) has the
mechanics, straight from the skill that drives it.
:::
