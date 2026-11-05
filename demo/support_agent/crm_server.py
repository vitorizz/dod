"""The store's CRM, as its own MCP server process (stdio), over SQLite.

Run by the agent with ``python -m demo.support_agent.crm_server <db path>``.
It knows nothing about dod: the agent's MCP client session is what's
instrumented.
"""

import json
import sqlite3
import sys
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True)
TODAY = date(2026, 11, 4)

ORDERS = [
    # order id, customer, item, amount, days since delivery
    ("2001", "ana", "noise-cancelling headphones", 249.00, 45),
    ("2002", "ben", "espresso grinder", 129.00, 12),
    ("2003", "chloe", "trail running shoes", 139.00, 50),
    ("2004", "dev", "mechanical keyboard", 179.00, 3),
    ("2005", "eli", "standing desk mat", 59.00, 20),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, customer TEXT, item TEXT,
    amount REAL, delivered TEXT);
CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, customer TEXT, order_id TEXT,
    action TEXT, summary TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS replies (id TEXT PRIMARY KEY, customer TEXT, question TEXT,
    text TEXT);
"""


def connect(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    db.executescript(SCHEMA)
    if not db.execute("SELECT 1 FROM orders").fetchone():
        db.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [(o, c, i, a, (TODAY - timedelta(days=d)).isoformat()) for o, c, i, a, d in ORDERS],
        )
    return db


def _next_id(db, table: str, prefix: str) -> str:
    (n,) = db.execute(f"SELECT count(*) FROM {table}").fetchone()
    return f"{prefix}{n + 1}"


def build(path: str) -> FastMCP:
    db = connect(path)
    server = FastMCP("crm", log_level="WARNING")

    @server.tool(annotations=READ_ONLY)
    def lookup_order(order_id: str) -> str:
        """Look up an order: customer, item, amount, delivery date."""
        row = db.execute("SELECT id, customer, item, amount, delivered FROM orders"
                         " WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            return json.dumps({"error": f"no order {order_id}"})
        days = (TODAY - date.fromisoformat(row[4])).days
        return json.dumps({"order_id": row[0], "customer": row[1], "item": row[2],
                           "amount": row[3], "delivered": row[4],
                           "days_since_delivery": days})

    @server.tool()
    def create_ticket(customer: str, order_id: str, action: str, summary: str) -> str:
        """Open a ticket for the fulfilment team (e.g. action "approve_refund")."""
        tid = _next_id(db, "tickets", "T")
        db.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, 'open')",
                   (tid, customer, order_id, action, summary))
        return json.dumps({"id": tid})

    @server.tool()
    def send_reply(customer: str, question: str, text: str) -> str:
        """Send a reply to a customer; returns the reply's ID."""
        rid = _next_id(db, "replies", "R")
        db.execute("INSERT INTO replies VALUES (?, ?, ?, ?)", (rid, customer, question, text))
        return json.dumps({"id": rid})

    @server.tool(annotations=READ_ONLY)
    def get_record(record_id: str) -> str:
        """A ticket (T...) or reply (R...) by ID."""
        if record_id.startswith("T"):
            row = db.execute("SELECT id, customer, order_id, action, summary, status"
                             " FROM tickets WHERE id = ?", (record_id,)).fetchone()
            keys = ("id", "customer", "order_id", "action", "summary", "status")
        else:
            row = db.execute("SELECT id, customer, question, text FROM replies WHERE id = ?",
                             (record_id,)).fetchone()
            keys = ("id", "customer", "question", "text")
        return json.dumps(dict(zip(keys, row)) if row else {"error": "not found"})

    return server


if __name__ == "__main__":
    build(sys.argv[1]).run("stdio")
