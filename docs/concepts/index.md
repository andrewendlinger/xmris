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

(concepts)=
# Concepts

Sooner or later xmris will refuse to do something you asked for — a function will insist on a
dimension called `"time"`, or reject data that has no `reference_frequency`. That strictness is not
an oversight; it is the whole design, and these three articles are the argument behind it.

Nothing here teaches a task. If you want to *do* something, the hands-on chapters
([Basics](#basics), [Processing pipeline](#pipeline)) are where the code lives. Come here when you
want to know why the code looks like that — or before you write library code yourself.

(concepts-pages)=
## The three articles

**Order is a suggestion, not a prerequisite.** Each stands alone; the architecture tour is the
widest and makes a good first stop.

| | Article | The question it answers |
|---|---|---|
| 1 | [The xmris architecture](#architecture) | Why an xarray accessor at all — what "parameter soup" and hidden state cost you, and what the data dictionary buys back |
| 2 | [The controlled vocabulary](#vocabulary) | My scanner calls it `spec_freq` — why won't xmris just guess? |
| 3 | [The two domains](#domains) | How a function knows whether it is holding a FID or a spectrum, when there is no flag saying so |

(concepts-executed)=
## These pages run

Explainers here are notebooks like any tutorial: their claims execute on every pull request. A
statement about what a function returns is not prose someone remembered to update — it is a cell
that fails the build when it stops being true.

The executable *proof* of the domain rules lives one chapter over, in
[Domain contracts in action](#domain-contracts): a table of what every operation class returns for
every input domain, asserted line by line.

(concepts-contributors)=
## If you are writing library code

These three articles are the reasoning; the rules distilled from them are the
[Architecture Contract](#contract) — eleven numbered Commandments, each naming the test that
enforces it. Read the concepts for *why*, cite the contract for *what*.
