"""SQLite persistence.

``sqlite3`` is in the standard library, so "self-hosted" here means *literally
one file* — no server to run, no migration tool, no connection pool. For a
single team that is comfortably enough: a million usage rows is a few hundred
megabytes, and the aggregations in :mod:`kharcha.core.usage` are the
bottleneck long before SQLite is.

Schema notes:

* Costs are stored as **TEXT**, not REAL. SQLite's REAL is an IEEE float and
  silently reintroduces exactly the error this project exists to prevent.
* Timestamps are stored as ISO-8601 UTC strings, so the file is readable in
  any editor and sorts correctly as text.
* ``prompt_key`` is a hash. The prompt itself is never stored — see
  :mod:`kharcha.core.redact`.

Pure standard library.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from .money import usd
from .usage import UsageRecord

__all__ = ["UsageStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   TEXT    NOT NULL,
    model                TEXT    NOT NULL,
    input_tokens         INTEGER NOT NULL DEFAULT 0,
    output_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens  INTEGER NOT NULL DEFAULT 0,
    team                 TEXT    NOT NULL DEFAULT '',
    project              TEXT    NOT NULL DEFAULT '',
    user                 TEXT    NOT NULL DEFAULT '',
    cost                 TEXT,
    prompt_key           TEXT    NOT NULL DEFAULT '',
    latency_ms           INTEGER,
    request_id           TEXT    NOT NULL DEFAULT '',
    ingested_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_ts       ON usage(ts);
CREATE INDEX IF NOT EXISTS idx_usage_model    ON usage(model);
CREATE INDEX IF NOT EXISTS idx_usage_team     ON usage(team);
CREATE INDEX IF NOT EXISTS idx_usage_project  ON usage(project);
CREATE INDEX IF NOT EXISTS idx_usage_prompt   ON usage(prompt_key, ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budgets (
    key        TEXT PRIMARY KEY,
    scope      TEXT NOT NULL DEFAULT 'team',
    limit_usd  TEXT NOT NULL,
    currency   TEXT NOT NULL DEFAULT 'USD'
);
"""


class UsageStore:
    """One SQLite file holding every usage record."""

    def __init__(self, path: Union[str, Path] = "kharcha.db") -> None:
        self.path = str(path)
        if path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- schema ------------------------------------------------------------

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
        self._conn.commit()

    @property
    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "UsageStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def add(self, record: UsageRecord) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO usage (ts, model, input_tokens, output_tokens,
                   cached_input_tokens, team, project, user, cost,
                   prompt_key, latency_ms, request_id, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.ts.astimezone(timezone.utc).isoformat(),
                record.model, record.input_tokens, record.output_tokens,
                record.cached_input_tokens, record.team, record.project,
                record.user,
                None if record.cost is None else str(record.cost),
                record.prompt_key, record.latency_ms, record.request_id, now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_many(self, records: Iterable[UsageRecord]) -> int:
        """Bulk insert in one transaction — ~50x faster than per-row commits."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                r.ts.astimezone(timezone.utc).isoformat(), r.model,
                r.input_tokens, r.output_tokens, r.cached_input_tokens,
                r.team, r.project, r.user,
                None if r.cost is None else str(r.cost),
                r.prompt_key, r.latency_ms, r.request_id, now,
            )
            for r in records
        ]
        with self._conn:
            self._conn.executemany(
                """INSERT INTO usage (ts, model, input_tokens, output_tokens,
                       cached_input_tokens, team, project, user, cost,
                       prompt_key, latency_ms, request_id, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> UsageRecord:
        return UsageRecord(
            ts=datetime.fromisoformat(row["ts"]),
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_input_tokens=row["cached_input_tokens"],
            team=row["team"],
            project=row["project"],
            user=row["user"],
            cost=None if row["cost"] is None else Decimal(row["cost"]),
            prompt_key=row["prompt_key"],
            latency_ms=row["latency_ms"],
            request_id=row["request_id"],
        )

    def all(self, *, limit: Optional[int] = None) -> List[UsageRecord]:
        sql = "SELECT * FROM usage ORDER BY ts"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self._row_to_record(r)
                for r in self._conn.execute(sql).fetchall()]

    def since(self, start: datetime) -> List[UsageRecord]:
        rows = self._conn.execute(
            "SELECT * FROM usage WHERE ts >= ? ORDER BY ts",
            (start.astimezone(timezone.utc).isoformat(),)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM usage").fetchone()["n"]

    def raw_cost_total(self) -> Decimal:
        """Sum of provider-reported costs only (ignores computed costs)."""
        row = self._conn.execute(
            "SELECT SUM(CAST(cost AS REAL)) AS total FROM usage").fetchone()
        return usd(row["total"] or 0)

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM usage")
            self._conn.execute(
                "DELETE FROM sqlite_sequence WHERE name='usage'")

    # -- budgets -----------------------------------------------------------

    def set_budget(self, key: str, limit_usd: Union[str, float, Decimal],
                   *, scope: str = "team", currency: str = "USD") -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO budgets (key, scope, limit_usd, currency)
                   VALUES (?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     scope=excluded.scope, limit_usd=excluded.limit_usd,
                     currency=excluded.currency""",
                (key, scope, str(usd(limit_usd)), currency))

    def budgets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM budgets ORDER BY key").fetchall()
        return [{"key": r["key"], "scope": r["scope"],
                 "limit_usd": Decimal(r["limit_usd"]), "currency": r["currency"]}
                for r in rows]

    def delete_budget(self, key: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM budgets WHERE key=?", (key,))
        return cur.rowcount > 0

    # -- export ------------------------------------------------------------

    def to_jsonl(self) -> str:
        """One JSON object per record — the interchange format for everything."""
        return "\n".join(
            json.dumps(r.to_dict(), ensure_ascii=False) for r in self.all()
        )
