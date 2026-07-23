# Guide — `docs/contributing/`

Procedural pages for people working *on* xmris rather than *with* it: setup, publishing, the
contributing walkthrough, widget authoring.

**This is the one genre exempt from the motivated-narrative rule.** A guide's reader has already
decided to do the thing and wants the steps. `setup.md` should be a numbered list of commands;
novelizing it makes it worse. Everything else in `CLAUDE.md` § "Documentation style" still binds —
one home per concept, stands alone, the palette carries the argument.

## Skeleton

Frontmatter only if the page needs live cells — `static_widgets.md` does, `setup.md` does not.
When you do add it, the kernel label is still exactly `Python 3 (xmris)`.

````markdown
(my-guide)=
# Doing The Thing

<One paragraph: what this page gets you, and when you would be here. State prerequisites
outright rather than assuming them.>

(my-guide-step-1)=
## Step 1: <imperative verb phrase>

<What to run, and what you should see.>

```bash
uv run <command>
```

(my-guide-step-2)=
## Step 2: <imperative verb phrase>

...

(my-guide-troubleshooting)=
## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| ... | ... | ... |
````

## Rules specific to this genre

- **Commands must be copy-pasteable and current.** Package manager is `uv`, never pip. Verify
  every command actually runs before committing — a guide is judged entirely on whether its steps
  work.
- **Point at the canonical source instead of restating it.** `guide.md` already does this well:
  *"Where any contributor page differs from The Architecture Contract (`contract.md`), that page
  wins."* Duplicated rules drift; a pointer cannot.
- **Say what a step is for.** One clause of motivation per step is not narrative, it is the
  difference between a runnable list and a cargo cult.
- A troubleshooting table beats scattered warnings — it is the shape a reader scans when stuck.

## Register

TOC group is **Contribute**, kept last in `docs/myst.yml`. These pages are also the ones most
likely to be read raw on GitHub rather than on the site, so keep them legible unrendered: no
reliance on a directive to carry essential meaning.
