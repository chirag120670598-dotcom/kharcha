"""Usage records and aggregation.

A :class:`UsageRecord` is one API call. Aggregation is done in Python over the
record set rather than in SQL, because the interesting rollups (per-team
burn rate, per-model cache-hit ratio, per-project share of a shared model)
are awkward to express in portable SQL and trivial to express as dict
group-bys. For very large deployments :class:`kharcha.core.store.UsageStore`
can pre-aggregate by day.

Pure standard library.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .money import round_cents, round_micros, usd
from .pricing import PricingTable

__all__ = ["UsageRecord", "UsageSet", "Rollup", "unknown_model_report"]


@dataclass(frozen=True)
class UsageRecord:
    """One billed API call.

    Attributes:
        ts: when it happened, timezone-aware.
        model: the model id the provider reported.
        input_tokens / output_tokens: full input count *including* anything
            served from cache. :meth:`fresh_input_tokens` gives the billable
            remainder.
        cached_input_tokens: how much of the input came from a prompt cache.
        team / project / user: attribution. Empty string means unattributed,
            which the waste detector treats as a finding in its own right.
        cost: provider-reported cost, if you have it. When set it wins over
            the computed cost, because the provider's number is the one you
            are actually billed.
        prompt_key: a hash of the prompt, used to detect duplicates. Store a
            hash, never the prompt itself.
        latency_ms: round-trip time, used to spot retries.
    """

    ts: datetime
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    team: str = ""
    project: str = ""
    user: str = ""
    cost: Optional[Decimal] = None
    prompt_key: str = ""
    latency_ms: Optional[int] = None
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        for label, value in (("input_tokens", self.input_tokens),
                             ("output_tokens", self.output_tokens),
                             ("cached_input_tokens", self.cached_input_tokens)):
            if value < 0:
                raise ValueError(f"{label} cannot be negative, got {value}")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                f"cached_input_tokens ({self.cached_input_tokens}) cannot exceed "
                f"input_tokens ({self.input_tokens})"
            )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.ts.tzinfo is None:
            raise ValueError(
                "ts must be timezone-aware — naive datetimes make 'today' "
                "ambiguous across a distributed team"
            )

    # -- derived -----------------------------------------------------------

    @property
    def fresh_input_tokens(self) -> int:
        """Input tokens billed at the full rate."""
        return self.input_tokens - self.cached_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> Optional[float]:
        """Share of input served from cache, or ``None`` if there was no input."""
        if self.input_tokens == 0:
            return None
        return self.cached_input_tokens / self.input_tokens

    @property
    def day(self) -> date:
        return self.ts.astimezone(timezone.utc).date()

    @property
    def hour(self) -> int:
        return self.ts.astimezone(timezone.utc).hour

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "team": self.team,
            "project": self.project,
            "user": self.user,
            "cost": None if self.cost is None else str(self.cost),
            "prompt_key": self.prompt_key,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UsageRecord":
        ts = data["ts"]
        if isinstance(ts, str):
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            parsed = ts
        return cls(
            ts=parsed,
            model=data["model"],
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cached_input_tokens=int(data.get("cached_input_tokens", 0) or 0),
            team=data.get("team", "") or "",
            project=data.get("project", "") or "",
            user=data.get("user", "") or "",
            cost=None if data.get("cost") is None else usd(data["cost"]),
            prompt_key=data.get("prompt_key", "") or "",
            latency_ms=data.get("latency_ms"),
            request_id=data.get("request_id", "") or "",
        )


@dataclass
class Rollup:
    """Aggregated numbers for one bucket (a team, a model, a day …)."""

    key: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost: Decimal = Decimal("0")
    models: set = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def avg_cost_per_call(self) -> Decimal:
        return round_micros(self.cost / self.calls) if self.calls else Decimal("0")

    @property
    def cache_hit_ratio(self) -> Optional[float]:
        return (self.cached_input_tokens / self.input_tokens) if self.input_tokens else None

    @property
    def avg_output_tokens(self) -> float:
        return (self.output_tokens / self.calls) if self.calls else 0.0

    def merge(self, record: "UsageRecord", cost: Decimal) -> None:
        self.calls += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cached_input_tokens += record.cached_input_tokens
        self.cost += cost
        self.models.add(record.model)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost": str(round_cents(self.cost)),
            "avg_cost_per_call": str(self.avg_cost_per_call),
            "cache_hit_ratio": self.cache_hit_ratio,
            "models": sorted(self.models),
        }


def unknown_model_report(records: Iterable[UsageRecord],
                         table: PricingTable) -> Dict[str, int]:
    """Count of records whose model has no price.

    These are calls whose cost the dashboard cannot compute. Surfacing them is
    the difference between a dashboard you trust and one that quietly
    under-reports your bill.
    """
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        if table.get(record.model) is None:
            counts[record.model] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


class UsageSet:
    """An immutable-ish collection of records with aggregation helpers."""

    def __init__(self, records: Sequence[UsageRecord] = (),
                 table: Optional[PricingTable] = None) -> None:
        self.records: List[UsageRecord] = list(records)
        self.table = table or PricingTable.default()

    def add(self, record: UsageRecord) -> "UsageSet":
        self.records.append(record)
        return self

    def extend(self, records: Iterable[UsageRecord]) -> "UsageSet":
        self.records.extend(records)
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    # -- costing -----------------------------------------------------------

    def cost_of(self, record: UsageRecord) -> Decimal:
        """The provider's own number if present, else computed from the table."""
        if record.cost is not None:
            return usd(record.cost)
        computed = self.table.cost(
            record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
        )
        return round_micros(computed) if computed is not None else Decimal("0")

    def total_cost(self, records: Optional[Iterable[UsageRecord]] = None) -> Decimal:
        return round_cents(sum(
            (self.cost_of(r) for r in (records if records is not None else self.records)),
            Decimal("0"),
        ))

    # -- rollups -----------------------------------------------------------

    def _group(self, records: Iterable[UsageRecord], keyfn) -> Dict[str, Rollup]:
        buckets: Dict[str, Rollup] = {}
        for record in records:
            key = keyfn(record)
            bucket = buckets.setdefault(key, Rollup(key=key))
            bucket.merge(record, self.cost_of(record))
        return buckets

    def by_team(self) -> Dict[str, Rollup]:
        return self._group(self.records, lambda r: r.team or "(unattributed)")

    def by_project(self) -> Dict[str, Rollup]:
        return self._group(self.records, lambda r: r.project or "(unattributed)")

    def by_model(self) -> Dict[str, Rollup]:
        return self._group(self.records, lambda r: r.model)

    def by_provider(self) -> Dict[str, Rollup]:
        def provider_of(record: UsageRecord) -> str:
            rate = self.table.get(record.model)
            return rate.provider if rate else "(unknown)"
        return self._group(self.records, provider_of)

    def by_day(self) -> Dict[date, Rollup]:
        buckets: Dict[date, Rollup] = {}
        for record in self.records:
            key = record.day
            buckets.setdefault(key, Rollup(key=key.isoformat())).merge(
                record, self.cost_of(record))
        return dict(sorted(buckets.items()))

    def by_hour(self) -> Dict[int, Rollup]:
        buckets: Dict[int, Rollup] = {}
        for record in self.records:
            buckets.setdefault(record.hour, Rollup(key=str(record.hour))).merge(
                record, self.cost_of(record))
        return dict(sorted(buckets.items()))

    def top(self, keyfn, n: int = 10) -> List[Rollup]:
        """The ``n`` most expensive buckets under ``keyfn``."""
        if keyfn is UsageSet.by_team.__func__:  # pragma: no cover - defensive
            buckets = self.by_team()
        return sorted(self._group(self.records, keyfn).values(),
                      key=lambda b: -b.cost)[:n]

    # -- windowing ---------------------------------------------------------

    def between(self, start: datetime, end: datetime) -> "UsageSet":
        """Records in ``[start, end)``."""
        if end <= start:
            raise ValueError("end must be after start")
        return UsageSet(
            [r for r in self.records if start <= r.ts < end], self.table)

    def last_days(self, days: int, *, now: Optional[datetime] = None) -> "UsageSet":
        if days <= 0:
            raise ValueError("days must be positive")
        now = now or datetime.now(timezone.utc)
        return self.between(now - timedelta(days=days), now)

    def daily_costs(self, days: int = 30, *,
                    now: Optional[datetime] = None) -> List[Tuple[date, Decimal]]:
        """Cost per day for the last ``days`` days, zero-filled.

        Zero-filling matters: a missing day means nothing was spent, not that
        the data is missing, and a burn-rate projection over a gappy series is
        nonsense.
        """
        now = now or datetime.now(timezone.utc)
        start = (now - timedelta(days=days)).date()
        by_day = self.by_day()
        out: List[Tuple[date, Decimal]] = []
        for offset in range(days):
            day = start + timedelta(days=offset + 1)
            bucket = by_day.get(day)
            out.append((day, round_cents(bucket.cost) if bucket else Decimal("0")))
        return out

    def summary(self) -> dict:
        total_input = sum(r.input_tokens for r in self.records)
        total_cached = sum(r.cached_input_tokens for r in self.records)
        return {
            "calls": len(self.records),
            "total_cost": str(self.total_cost()),
            "input_tokens": total_input,
            "output_tokens": sum(r.output_tokens for r in self.records),
            "cached_input_tokens": total_cached,
            "cache_hit_ratio": (total_cached / total_input) if total_input else None,
            "models_used": len({r.model for r in self.records}),
            "unattributed_calls": sum(
                1 for r in self.records if not r.team and not r.project),
            "unpriced_models": unknown_model_report(self.records, self.table),
        }
