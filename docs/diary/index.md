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

The entries are a by-product of how xmris is actually built. A significant change usually starts
as a planning session — nowadays often with e.g. Claude Code — that ends in a precise, ordered
plan, reviewed in its own right (how, varies by contributor and tooling). That plan is exactly
right for *doing* the work and not so good for *keeping*: twenty mechanical steps with the one
real decision buried underneath them — and the plan file does not survive the merge.

So once the change lands, that decision is told here as a one-screen **story**: the tension, the
decision, the real file paths and code snippets so you can see how it works under the hood —
checked against the code as built, not drafted from the plan — and a straight account of why,
having now built it, we think the call was right.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    P["Plan"] --> B["Build"]
    B --> S["Tell the story"]
```

:::{seealso}
Writing one yourself? [Write a dev-diary entry](#contribute-dev-diary) has the
mechanics, straight from the skill that drives it.

After *what changed in a release* rather than why a design is the way it is? That
is the [changelog](#changelog) — one line per change, linking back to entries here
where one exists.
:::
