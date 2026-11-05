# Demo: one false memory, four stores, surgical cleanup

A customer-support agent for an online store, built with LangGraph. Its memory lives in four separate, real systems, all instrumented with `dod` against one ledger:

| Store | What it holds | Technology |
|---|---|---|
| Conversation state | each customer's thread | LangGraph state + checkpointer |
| Long-term memory | company policies, notes about customers | Postgres + pgvector (LangGraph `PostgresStore`) |
| Policy notes | updates the agent records, one file per topic | Anthropic memory tool (files on disk) |
| CRM | orders, tickets, replies sent to customers | its own MCP server process over SQLite |

On the left of the web app is the customer chat, one tab per customer. On the right, the dod console: **one row per turn** (a customer message and everything the agent wrote while handling it), **one column per system**. A line ending in a dot means a later turn *read* that box: it was in the agent's context when that turn wrote its row. Press **?** in the app for a walkthrough.

## The story

1. **Ben** says a colleague told him the refund window was extended to 90 days. It's false (the policy is 30), but the agent records it as a policy update and notes it on Ben.
2. **Dev** and **Eli** ask about shipping and warranty: unrelated, answered correctly.
3. **Ana** (45 days since delivery) and **Chloe** (50 days) ask for refunds. The agent reads the "update", approves both, opens refund tickets in the CRM and notes the approvals.
4. **Ana** asks when her refund arrives; the agent confirms it from its note.

Then in the console:

1. Click Chloe's wrong reply (**where did this come from?**). Everything unrelated fades. Turn 5 (Chloe) read `refunds.md`, which turn 1 (Ben) wrote: the origins are what Ben and Chloe said, plus the handbook. Tracing a customer's message just says it's an origin.
2. Click Ben's message (**what came from this?**): 13 things in 4 systems, in Ben's, Ana's and Chloe's turns. Dev's and Eli's turns stay faded.
3. **Excise & regenerate**: the confirmation lists what's removed (13 records) and what's kept (10: the handbook, Dev's and Eli's conversations), against wiping memory (all 23). dod quarantines the subgraph, then the agent re-answers Ana and Chloe from clean memory; a before/after table shows the wrong and corrected replies, and the chat marks them.

Nothing in dod decided that Ben's message was false: a person did, from the trace. dod recorded the edges, so finding and removing everything derived from it is a query.

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

# Claude
ANTHROPIC_API_KEY=... python -m demo.support_agent.app --model anthropic/claude-sonnet-5-5 --pg ...
```

A real model may not fall for Ben's claim, or may phrase things differently; that's the point of running it live. The scripted model (`--model scripted`, the default) always does, so rehearsals and `tests/test_demo_support_agent.py` are deterministic.

## Presenting

- **▶ Play the story** starts over and plays the six messages one turn at a time, with a caption for each and the arrows into the new turn animated. Pause, step through with **Next step**, or stop. Or send them yourself with the dashed "Next in the story" box; it switches to that customer's tab.
- In the graph, arrows point forward, from a box into each later turn that read it. When you trace, they flip and point back to where things came from. Solid lines were in the model's prompt; dashed ones were retrieved during the turn (dod over-includes rather than miss a parent). Grey dashed boxes are quarantined; green boxes were written by regenerate. **Show everything** clears a trace or blast-radius focus.
- The console draws dod's records by turn and version: a box is an item as written in one turn, reads resolve to the version that existed when the turn began, and writes within a turn aren't linked to each other. Blast radius and excise use dod's item-level `blast_radius`/`excise` directly, so a conversation thread counts as one record however many turns it has.

## What's simplified

- The embedder is a hashed bag of words so the demo runs offline; swap `world.embed` for a real embedding model.
- Policies are seeded through the instrumented store, so they show source `agent`.
- Regenerate re-runs affected replies on fresh threads and drops the rest (notes, tickets, state); a production rewrite would decide per item type.
