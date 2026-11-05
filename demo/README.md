# Demo: one false memory, four stores, surgical cleanup

A customer-support agent for an online store, built with LangGraph. Its memory lives in four separate, real systems, all instrumented with `dod` against one ledger:

| Store | What it holds | Technology |
|---|---|---|
| Conversation state | each customer's thread | LangGraph state + checkpointer |
| Long-term memory | company policies, notes about customers | Postgres + pgvector (LangGraph `PostgresStore`) |
| Policy notes | updates the agent records, one file per topic | Anthropic memory tool (files on disk) |
| CRM | orders, tickets, replies sent to customers | its own MCP server process over SQLite |

On the left of the web app is the customer chat; on the right, the dod console showing what every write was derived from.

## The story

1. **Ben** says a colleague told him the refund window was extended to 90 days. It's false (the policy is 30), but the agent records it as a policy update and notes it on Ben.
2. **Dev** and **Eli** ask about shipping and warranty: unrelated, answered correctly.
3. **Ana** (45 days since delivery) and **Chloe** (50 days) ask for refunds. The agent reads the "update", approves both, opens refund tickets in the CRM and notes the approvals.
4. **Ana** asks when her refund arrives; the agent confirms it from its note.

Then in the console:

1. Click Chloe's wrong reply (`trace ↗`). The trace runs back across all four stores to two customer messages: Chloe's own question, and Ben's claim.
2. Click Ben's message: its blast radius is 12 items in 4 stores (the policy note, three customer notes, two tickets, four replies, two conversations).
3. **Excise & regenerate**: the confirm dialog compares excising (13 contaminated items out, 10 correct items kept) with wiping every store (all 23 lost). dod quarantines the subgraph, then the agent re-answers Ana and Chloe from clean memory ("outside our 30-day refund window"), shown in green. Dev's and Eli's conversations are untouched.

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

- **Run the story** replays the six messages at once. Or type them one by one: the dashed box above the input sends the next story message.
- Solid edges were in the model's prompt; dashed ones were retrieved in the same task (dod over-includes rather than miss a parent).
- Grey dashed boxes are quarantined: instrumented reads (Postgres search, file views) no longer return them. Green boxes were written by regenerate.

## What's simplified

- The embedder is a hashed bag of words so the demo runs offline; swap `world.embed` for a real embedding model.
- Policies are seeded through the instrumented store, so they show source `agent`.
- Regenerate re-runs affected replies on fresh threads and drops the rest (notes, tickets, state); a production rewrite would decide per item type.
