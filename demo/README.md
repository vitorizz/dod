# Demo: one false memory, four stores, surgical cleanup

A customer-support agent for an online store, built with LangGraph. Its memory lives in four separate, real systems, all instrumented with `dod` against one ledger:

| Store | What it holds | Technology |
|---|---|---|
| Conversation state | each customer's thread | LangGraph state + checkpointer |
| Long-term memory | company policies, notes about customers | Postgres + pgvector (LangGraph `PostgresStore`) |
| Policy notes | updates the agent records, one file per note | Anthropic memory tool (files on disk) |
| CRM | orders, tickets, replies sent to customers | its own MCP server process over SQLite |

On the left of the web app is the customer chat, one tab per customer. On the right, the dod console: **one row per turn** (a customer message and everything the agent wrote while handling it), **one column per system**. A line ending in a dot means a later turn *read* that box: it was in the agent's context when that turn wrote its row. Press **?** in the app for a walkthrough.

## The story

1. **Ops** posts a policy update on the internal channel: the refund window is now 90 days. It's a staff mistake (the policy is still 30), but staff are a source the agent should trust, so it records the update as a policy note. (The same claim from a customer is answered but not recorded.)
2. **Dev** asks about shipping: unrelated, answered correctly.
3. **Ana** (45 days since delivery) asks for a refund. The agent reads the note, approves it, opens a refund ticket in the CRM and notes the approval.
4. **Ana** asks when her refund arrives; the agent confirms it from its note. The mistake builds on itself.

## Then someone notices

Cleanup starts when a person finds a mistake, so there are no cleanup buttons on records. The support lead opens the **Support lead** tab in the chat and tells the agent what's right: “The refund window is 30 days, not 90.” The agent answers in the chat, and nothing changes until the lead confirms:

1. **What contradicts you**: the beliefs about refunds (policy notes, customer notes, the handbook), split into what it will fix (those stating something else) and what it leaves alone; **Keep as is** / **Fix this too** move a record between the two; dod never decides what's false.
2. **What I'd do**: dod traces them to their origin (the ops update) and groups what was built on it by what to do:
   - **fix**: wrong notes stop being read; the correction is written as a policy note in their place;
   - **cancel**: Ana's refund ticket, so fulfilment doesn't act on it;
   - **notify**: replies already sent stay on record, and Ana and ops each get a correction;
   - **keep on record**: the conversations, hidden from the agent;
   - **untouched**: the handbook, Dev's conversation. Wiping memory would have lost those too.
3. **Fix it**: the correction is recorded (store `console`, source `user`), its note written, the ticket cancelled through the CRM, then `dod.regenerate` from the ops update quarantines what it reached (reversible: nothing is deleted) and sends the corrections: ops gets the correction itself, Ana's two questions are re-answered on fresh threads. Everything the correction writes traces back to it, and afterwards the agent answers refunds from the correction's note.
   The graph then replays the fix step by step, with a caption for each (in the ledger it takes milliseconds): the confirmed records, the trace to the ops update, the blast radius, each group quarantined or cancelled in turn, then the correction and each corrected turn appearing. Pause, step or **Skip** it with the bar above the graph.
4. **Audit report** (button on the agent's answer): every record the correction touched, what happened to it, and the audit entry that sealed it, with the ledger's hash chain checked. Downloadable as Markdown (`dod.audit.report`).

The Support lead tab runs `dod.admin` (the library's command layer, as a Slack bot would), so it also takes **@trace R3** (where did a reply come from?), **@impact** a fact (what did it affect?), **@undo** (release what the last fix quarantined), **@ledger** (what the ledger holds) and **@help**. Each answer shows the equivalent library call, and what a command names is highlighted on the graph.

You can still click any box: **where did this come from?** traces it back turn by turn; **what came from this?** shows its blast radius.

## Run it

From the repository root:

```bash
pip install -e ".[langgraph,anthropic,mcp,postgres]"
python -m demo.support_agent.app            # scripted model, in-memory long-term store
open http://localhost:8000
```

**With Postgres + pgvector** (any Postgres 13+ with the `vector` extension):

```bash
docker run -d --name dod-pg -e POSTGRES_PASSWORD=dod -p 5432:5432 pgvector/pgvector:pg16
python -m demo.support_agent.app --pg "postgresql://postgres:dod@localhost:5432/postgres"
```

**With a real model** (the agent asks the model for one JSON decision per message and performs the writes itself, so small open-weight models work too):

```bash
# Ollama, or any OpenAI-compatible server (vLLM, LM Studio, OpenAI): OPENAI_BASE_URL / OPENAI_API_KEY
ollama pull qwen3:8b
python -m demo.support_agent.app --model openai/qwen3:8b --pg ...

# Claude: put the key in a .env file at the repository root (git-ignored), once
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
python -m demo.support_agent.app          # uses Claude Haiku (anthropic/claude-haiku-5-5) when the key is set
```

Each message waits for the model (a few seconds with Claude); the chat shows the agent typing meanwhile. For another model put e.g. `DEMO_MODEL=anthropic/claude-sonnet-5-5` in `.env`.

A real model may question the ops update, or phrase things differently; that's the point of running it live. The scripted model (`--model scripted`, the default without a key) always follows it, so rehearsals and `tests/test_demo_support_agent.py` are deterministic.

## Presenting

- **▶ Play the story** starts over and plays the four messages one turn at a time, with a caption for each and the arrows into the new turn animated. Pause, step through with **Next step**, or stop. Or send them yourself with the dashed "Next in the story" box; it switches to that customer's tab.
- In the graph, arrows point forward, from a box into each later turn that read it. When you trace, they flip and point back to where things came from. Solid lines were in the model's prompt; dashed ones were retrieved during the turn (dod over-includes rather than miss a parent). Grey dashed boxes are no longer read (their tag says why: corrected, cancelled, on record, quarantined); green boxes were written by the correction. The correction's row has a green edge. **Show everything** clears a trace or blast-radius focus.
- The console draws dod's records by turn and version: a box is an item as written in one turn, reads resolve to the version that existed when the turn began, and writes within a turn aren't linked to each other. Blast radius and excise use dod's item-level `blast_radius`/`excise` directly, so a conversation thread counts as one record however many turns it has.

## What's simplified

- The embedder is a hashed bag of words so the demo runs offline; swap `world.embed` for a real embedding model.
- Policies are seeded through the instrumented store, so they show source `agent`.
- Finding candidates is a keyword match on the topic, and "contradicts" compares numbers of days with the correction; a real console would search semantically. Either way a person confirms.
- The correction re-answers affected replies on fresh threads and drops the rest (notes, state), cancelling tickets; a production system would decide per item type, and might batch a customer's corrections into one message.
