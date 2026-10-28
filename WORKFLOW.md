# Workflow Checklist

Every session on this project follows this file. It exists so that at the end of the semester we never have to remember who did what, why a rule was chosen, or which run produced which number. The guideline grades contribution, reproducibility and justification, and all three come from records we keep *while* working, not from memory afterwards.

> Based on the project guideline currently on the course site (dated 1.10.2024). Update this file if the lecturer publishes a new one.

---

## 1. Before you start

- [ ] `git pull`. Read the latest `history.md` entry (and older ones if you've been away).
- [ ] Run the tests: `python -m unittest discover -s tests -v`. If they fail on a fresh pull, **stop and tell the group**. Don't build on a broken state.
- [ ] **Claim your task** in the group chat: *"Working on X until <day>."* One person per task, so two people never code the same thing on the same evening.
- [ ] Check that your git identity is your own real name and that it's the one you always use:
      `git config user.name` / `git config user.email`

## 2. While working

- [ ] **Rule change?** (fuse, blast, movement, buffs, win conditions, rewards…) Update `RULES.md` **and** add or adjust a test for it in the same commit. Code, rules doc and tests must never disagree.
- [ ] **Made a design decision?** (reward values, observation format, algorithm choice, hyperparameter ranges…) Add a line to `DECISIONS.md` (see §5). The report's formulation and methodology sections are built from this file.
- [ ] **Used external code** (a library, a snippet from a blog, a repo, a paper's implementation)? Add it to `CREDITS.md` with a link. The guideline forbids plagiarism; citing what you reused is how you stay clear of it.
- [ ] **Ran an experiment?** Log it in `EXPERIMENTS.md` (see §6). Failed and boring runs too, since the comparison is graded, not just the winner.

## 3. Before you push

- [ ] All tests pass.
- [ ] No generated files committed (`__pycache__/`, `.pyc`, model checkpoints, logs). These belong in `.gitignore`.
- [ ] Commit messages say *what* changed: `Add chain explosions + tests`, not `fix` or `update`.
- [ ] You can explain every line you're pushing. AI-generated code is allowed. Code nobody on the team understands is a risk in the presentation Q&A.
- [ ] Write your `history.md` entry (template in §4), **newest on top**.

## 4. `history.md` entry template

Copy this block to the top of `history.md` and fill it in. Write it yourself, short and plain.

```markdown
## YYYY-MM-DD — <short title>

**Author:** <name>
**Commits:** <first hash>..<last hash>
**Time spent:** ~<hours>h
**Tests:** all passing / <which failing and why>

### What I did
- ...

### How it works (only what the next person needs)
- ...

### Decisions I made (also added to DECISIONS.md)
- ... / none

### AI-assisted parts
- e.g. "Cursor generated the blast logic; I reviewed it and wrote the tests" / none

### Open questions for the team
- ... / none

### Next task for whoever picks this up
- ...
```

## 5. How to record a decision (inside your `history.md` entry)

Decisions go in the **"Decisions I made"** section of your history entry, not in a separate file. They're the raw material for the report's *Problem formulation* and *Methodology* sections, and for answering "why did you choose X?" in the presentation, so write each one in this shape:

```markdown
- **<what was decided>**: <the value>
  - Why: <the reason>
  - Alternatives considered: <what else, and why not>
  - Status: team-agreed / provisional (needs team OK)
```

Example (placeholder, not a real decision):

```markdown
- **<rule or parameter>**: <value>
  - Why: <reason>
  - Alternatives considered: <option A (why not)>, <option B (why not)>
  - Status: provisional
```

If a decision changes an earlier one, say so: "Replaces the fuse decision from 2026-11-02."

## 6. `EXPERIMENTS.md` format

One row per training run. Never delete rows, even for failed runs.

```markdown
| Date | Who | Commit | Algorithm | Config file | Seed(s) | Map(s) | Result (eval metrics) | Notes |
|---|---|---|---|---|---|---|---|---|
```

- Every run must be reproducible from its row: **commit hash + config + seed**.
- Evaluation numbers come only from the shared evaluation script, so results are comparable across people.
