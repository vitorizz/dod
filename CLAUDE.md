# CLAUDE.md

Capture library for AI agents with long-lived memory: at every memory write, record which retrieved items were in context, giving a lineage graph you can `trace`, `blast-radius`, and `excise`. Background (why, core primitive, build plan, demo): `docs/PRODUCT.md`. Decisions: `docs/decisions.md`. Agent orchestration: `docs/ORCHESTRATION.md`.

## Product rules
- Capture at the orchestration/harness layer, never the storage layer.
- Cross-store and model-agnostic: no hard dependency on one database or model vendor in core code.
- Trace + excise only. No code, docs, or claims implying automated detection of bad memories.
- Capture must be the only write path; any bypass breaks the guarantee. Capture overhead p99 < 1 ms.
- Export edges as OpenTelemetry attributes in our own namespace, never `gen_ai.*`.
- CrewAI: wrap `Memory.recall`/`remember`, not its events; scope per agent/task with `contextvars`; pinned to crewai 1.15.22.

## Working conventions
- Keep responses concise.
- Prefer small, working increments over large scaffolds.
- Instrument everything: we'll run an experiment measuring how far false memories propagate.

## Stack
- Python >= 3.9 (CI runs 3.12). Package `dod` in `src/` layout, built with hatchling (`pyproject.toml`).
- Ledger: stdlib `sqlite3`, append-only, WAL mode. No ORM.
- Tests: pytest. Benchmarks are plain pytest tests using `tests/perf.py` (p50/p99 via `time.perf_counter_ns`).
- Framework integrations are optional extras, never core dependencies: `[crewai]` pins `crewai==1.15.22`.
- CI and agents: GitHub Actions (`ci.yml`, `claude.yml`, `claude-code-review.yml`, `project-board.yml`).

## Test commands
```bash
pip install -e ".[dev]"              # install package + test deps
pytest -q                            # full suite, including p99 < 1 ms capture benchmarks
pytest -q tests/test_*_perf.py -s    # benchmarks only, printing p50/p99
pip install -e ".[dev,crewai]"       # add the real CrewAI dependency
pytest -q -m crewai                  # CrewAI integration tests (skipped without crewai)
```
In agent runs, use these exact forms. Don't create a venv or call pip through `python -m`: they are not in the allowed tools and get denied. (`python -m pytest` is allowed.)
Benchmark files and the `crewai` marker arrive with #11 and #13.

## Workflow
1. Every change starts as an issue using the agent-task template (Goal / Acceptance criteria / Files involved), sized for one PR. New issues go on the board as Backlog with Priority and Size.
2. A human approves by adding the `agent-ready` label. The board moves the issue to Ready, then to In progress when Claude starts.
3. Claude works on a `claude/issue-<N>-*` branch and pushes. A workflow step opens the PR with `Closes #N` (using `PAT_TOKEN`, so CI and review trigger).
4. CI (pytest + benchmarks) runs on the PR. Claude Code Review runs only after CI passes, and not on docs-only PRs (all files Markdown or under `docs/`). Fix findings with `@claude` comments on the PR.
5. A human merges. Merging closes the issue.

If blocked or unsure (ambiguous acceptance criteria, a design decision the issue doesn't settle, a conflict with this file, a missing dependency or permission):
1. Comment the specific question on the issue: `gh issue comment <N> --body "..."`. Say what you need decided, the options you see, and what you'd need to proceed.
2. Add the label: `gh issue edit <N> --add-label needs-human`.
3. Stop. No more commits and no PR; the PR-opening step skips issues labeled `needs-human`.

The human answers with an `@claude` comment on the issue. That run removes `needs-human` before Claude starts, so continue from the answer; the PR-opening step then works again.

Handoffs (Instinct → planner → builders; see `docs/ORCHESTRATION.md`):
- Instinct hands off through a feature issue labeled `needs-breakdown` stating goal, why, priority, and definition of done.
- The planner reads it once and creates the technical sub-issues.
- Questions go on the issue, with at most one clarifying round. Direction or design questions go to the human via `needs-human`.
- No open-ended agent discussions in Slack.

**Never guess on design decisions.** Data model, APIs, capture semantics, dependencies and anything recorded in `docs/decisions.md` get asked about, not assumed.

Rules for every PR:
- New or changed behavior comes with tests. Anything on the capture path includes or updates a benchmark asserting p99 < 1 ms.
- One issue, one PR. Stay within "Files involved"; if more is needed, say so in the PR instead of expanding scope.
- Don't edit `.github/workflows/` from agent runs (the GitHub App can't push workflow changes).
- Keep strategy (pricing, positioning, competitors) out of code, comments, and commit messages.
- Record decisions in `docs/decisions.md`, one line of why each.

Hard rules (never break, in any PR):
- The ledger never stores memory content. IDs, content hashes, edges, and metadata only; content stays in the customer's stores.
