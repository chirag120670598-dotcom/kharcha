"""Budgets, burn rate and alerts.

The feature that actually stops a bill going wrong is not a pie chart — it is
knowing on the 9th of the month that a team is on track to blow through a
budget on the 24th.

Projection is deliberately naive-but-honest: a linear projection plus a
weekday-weighted one, both reported, with the *range* between them. AI spend
is spiky (a batch job on Tuesday, nothing at the weekend), and a single number
presented as a forecast is a lie with extra steps.

Pure standard library.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence

from .money import round_cents, usd

__all__ = [
    "Budget",
    "BudgetStatus",
    "Projection",
    "Alert",
    "check_budget",
    "project_month",
    "burn_rate",
    "DEFAULT_THRESHOLDS",
]

DEFAULT_THRESHOLDS: tuple = (0.5, 0.8, 1.0)


def _days_in_month(day: date) -> int:
    return calendar.monthrange(day.year, day.month)[1]


def burn_rate(cost: Decimal, days_elapsed: int) -> Decimal:
    """Average spend per day so far.

    Returns zero rather than raising on a zero divisor — a budget that has not
    started yet has no burn rate, not an error.
    """
    if days_elapsed <= 0:
        return Decimal("0")
    return usd(cost) / Decimal(days_elapsed)


@dataclass(frozen=True)
class Projection:
    """Where the month lands, as a range rather than a false-precision point."""

    linear: Decimal
    weekday_weighted: Decimal

    @property
    def low(self) -> Decimal:
        return round_cents(min(self.linear, self.weekday_weighted))

    @property
    def high(self) -> Decimal:
        return round_cents(max(self.linear, self.weekday_weighted))

    def as_text(self, *, symbol: str = "$") -> str:
        if self.low == self.high:
            return f"{symbol}{self.low}"
        return f"{symbol}{self.low}\u2013{symbol}{self.high}"


def project_month(
    spent: Decimal,
    *,
    today: date,
    daily_costs: Optional[Sequence[Decimal]] = None,
) -> Projection:
    """Project total month-end spend.

    Args:
        spent: total spent so far this month.
        today: today's date.
        daily_costs: optional per-day costs, oldest first. When given, the
            weekday-weighted projection uses the average for *the remaining
            weekdays/weekends separately* instead of assuming every day costs
            the same.
    """
    days_in_month = _days_in_month(today)
    elapsed = today.day
    remaining = days_in_month - elapsed

    linear = usd(spent) + burn_rate(spent, elapsed) * Decimal(remaining)

    if not daily_costs or remaining == 0:
        return Projection(linear, linear)

    first = today.replace(day=1)
    weekday_cost: List[Decimal] = []
    weekend_cost: List[Decimal] = []
    for offset, amount in enumerate(daily_costs):
        day = first + timedelta(days=offset)
        (weekend_cost if day.weekday() >= 5 else weekday_cost).append(usd(amount))

    def _avg(values: List[Decimal]) -> Decimal:
        return (sum(values, Decimal("0")) / Decimal(len(values))) if values else Decimal("0")

    remaining_weekdays = remaining_weekend = 0
    for offset in range(1, remaining + 1):
        day = today + timedelta(days=offset)
        if day.weekday() >= 5:
            remaining_weekend += 1
        else:
            remaining_weekdays += 1

    weighted = (usd(spent)
                + _avg(weekday_cost) * Decimal(remaining_weekdays)
                + _avg(weekend_cost) * Decimal(remaining_weekend))
    return Projection(linear, weighted)


@dataclass
class Budget:
    """A spend limit for one team or project over a calendar month."""

    key: str
    limit: Decimal
    scope: str = "team"          # "team" | "project" | "company"
    currency: str = "USD"
    thresholds: tuple = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        self.limit = usd(self.limit)
        if self.limit <= 0:
            raise ValueError(f"budget limit for {self.key} must be positive")
        if not self.key:
            raise ValueError("budget needs a key")
        if not self.thresholds:
            raise ValueError("budget needs at least one threshold")
        for t in self.thresholds:
            if not 0 < float(t) <= 2:
                raise ValueError(f"threshold {t} for {self.key} is out of range")


@dataclass(frozen=True)
class Alert:
    """A crossed threshold, or a projection that the budget will be crossed.

    ``kind`` separates "where you are" from "where you are going" so a caller
    can tie-break sensibly: at equal severity the threshold alert is the more
    concrete of the two.
    """

    key: str
    level: str            # "warning" | "critical" | "exceeded"
    fraction: Decimal
    spent: Decimal
    limit: Decimal
    message: str
    kind: str = "threshold"   # "threshold" | "projection"

    @property
    def pct(self) -> float:
        return float(self.fraction * 100)

    def to_dict(self) -> dict:
        return {
            "key": self.key, "level": self.level, "kind": self.kind,
            "fraction": str(self.fraction),
            "spent": str(self.spent), "limit": str(self.limit),
            "message": self.message,
        }


@dataclass(frozen=True)
class BudgetStatus:
    """Everything you need to know about one budget right now."""

    budget: Budget
    spent: Decimal
    today: date
    projection: Projection
    alerts: tuple = ()

    @property
    def fraction(self) -> Decimal:
        return self.spent / self.budget.limit if self.budget.limit else Decimal("0")

    @property
    def remaining(self) -> Decimal:
        return round_cents(self.budget.limit - self.spent)

    @property
    def days_left(self) -> int:
        return _days_in_month(self.today) - self.today.day

    @property
    def daily_allowance(self) -> Decimal:
        """What can be spent per day for the rest of the month."""
        if self.days_left <= 0:
            return Decimal("0")
        return round_cents(max(Decimal("0"), self.remaining) / Decimal(self.days_left))

    @property
    def projected_overage(self) -> Decimal:
        return round_cents(max(Decimal("0"), self.projection.high - self.budget.limit))

    @property
    def on_track(self) -> bool:
        return self.projection.high <= self.budget.limit

    @property
    def worst_alert(self) -> Optional[Alert]:
        """Most severe alert; ties go to the threshold alert over a projection."""
        order = {"exceeded": 3, "critical": 2, "warning": 1}
        return max(
            self.alerts,
            key=lambda a: (order.get(a.level, 0), a.kind == "threshold"),
            default=None,
        )

    def to_dict(self) -> dict:
        return {
            "key": self.budget.key,
            "scope": self.budget.scope,
            "limit": str(self.budget.limit),
            "spent": str(round_cents(self.spent)),
            "fraction": str(round(self.fraction, 4)),
            "remaining": str(self.remaining),
            "days_left": self.days_left,
            "daily_allowance": str(self.daily_allowance),
            "projection": {"linear": str(round_cents(self.projection.linear)),
                           "weekday_weighted": str(round_cents(self.projection.weekday_weighted)),
                           "low": str(self.projection.low),
                           "high": str(self.projection.high)},
            "projected_overage": str(self.projected_overage),
            "on_track": self.on_track,
            "alerts": [a.to_dict() for a in self.alerts],
        }


def check_budget(
    budget: Budget,
    spent: Decimal,
    *,
    today: Optional[date] = None,
    daily_costs: Optional[Sequence[Decimal]] = None,
) -> BudgetStatus:
    """Evaluate one budget and raise any crossed thresholds.

    >>> from decimal import Decimal
    >>> b = Budget("team-a", Decimal("1000"))
    >>> s = check_budget(b, Decimal("850"), today=date(2026, 3, 10))
    >>> s.worst_alert.level
    'critical'
    """
    today = today or datetime.now(timezone.utc).date()
    spent = round_cents(usd(spent))
    projection = project_month(spent, today=today, daily_costs=daily_costs)

    fraction = spent / budget.limit if budget.limit else Decimal("0")
    alerts: List[Alert] = []

    # 1. where you are right now
    for threshold in sorted(budget.thresholds, reverse=True):
        if fraction >= Decimal(str(threshold)):
            if fraction > 1:
                level, message = "exceeded", (
                    f"{budget.key} is over budget: {spent} of {budget.limit} "
                    f"({fraction:.0%})")
            elif fraction >= Decimal("0.8"):
                level, message = "critical", (
                    f"{budget.key} has used {fraction:.0%} of its "
                    f"{budget.limit} budget")
            else:
                level, message = "warning", (
                    f"{budget.key} has used {fraction:.0%} of its "
                    f"{budget.limit} budget")
            alerts.append(Alert(
                key=budget.key, level=level, fraction=fraction,
                spent=spent, limit=budget.limit, message=message,
            ))
            break  # only the most severe threshold

    # 2. where you are going — reported even when a threshold already fired,
    #    because "you are fine today but will not be on the 24th" is the alert
    #    that is actually actionable.
    if projection.high > budget.limit:
        alerts.append(Alert(
            key=budget.key,
            level="critical" if fraction >= Decimal("0.8") else "warning",
            fraction=fraction, spent=spent, limit=budget.limit,
            kind="projection",
            message=(f"{budget.key} is on track to overspend: projected "
                     f"{projection.high} against a {budget.limit} limit"),
        ))

    return BudgetStatus(
        budget=budget, spent=spent, today=today,
        projection=projection, alerts=tuple(alerts),
    )


def check_all(
    budgets: Iterable[Budget],
    spend_by_key: Dict[str, Decimal],
    *,
    today: Optional[date] = None,
    daily_by_key: Optional[Dict[str, Sequence[Decimal]]] = None,
) -> List[BudgetStatus]:
    """Evaluate every budget, worst first."""
    today = today or datetime.now(timezone.utc).date()
    daily_by_key = daily_by_key or {}
    statuses = [
        check_budget(b, spend_by_key.get(b.key, Decimal("0")), today=today,
                     daily_costs=daily_by_key.get(b.key))
        for b in budgets
    ]
    return sorted(statuses, key=lambda s: -s.fraction)
