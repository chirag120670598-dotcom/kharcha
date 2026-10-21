"""Reports.

Renders a :class:`~kharcha.core.usage.UsageSet` as text or Markdown. Plain
text is what you paste into Slack; Markdown is what you commit to a repo or
email to whoever signs the cloud bill.

Both include the *same* numbers — a dashboard and a report that disagree is
how you lose an argument about the bill.

Pure standard library.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence

from .budgets import BudgetStatus
from .money import format_inr, format_usd, human_amount, round_cents, usd
from .usage import Rollup, UsageSet
from .waste import Finding, find_waste, total_potential_saving

__all__ = ["text_report", "markdown_report", "table", "sparkline"]


def _bar(fraction: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(fraction * width))))
    return "\u2588" * filled + "\u2591" * (width - filled)


_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values: Sequence[Decimal]) -> str:
    """A one-line trend, so you can see a spike in a terminal.

    >>> sparkline([Decimal(1), Decimal(5), Decimal(3)])
    '\u2581\u2588\u2584'
    """
    if not values:
        return ""
    numbers = [float(v) for v in values]
    lo, hi = min(numbers), max(numbers)
    if hi == lo:
        return _SPARK[3] * len(numbers)
    return "".join(
        _SPARK[int((n - lo) / (hi - lo) * (len(_SPARK) - 1))] for n in numbers
    )


def table(rows: Sequence[Sequence[str]], headers: Sequence[str],
          *, align: Optional[Sequence[str]] = None) -> str:
    """A monospace table. ``align`` is a string of ``'l'``/``'r'`` per column."""
    align = align or "l" * len(headers)
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    out = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("\u2500" * w for w in widths))
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            text = str(cell)
            cells.append(text.rjust(widths[i]) if align[i] == "r"
                         else text.ljust(widths[i]))
        out.append("  ".join(cells).rstrip())
    return "\n".join(out)


def _rollup_rows(rollups: Iterable[Rollup], *, top: int = 10) -> List[List[str]]:
    ordered = sorted(rollups, key=lambda r: -r.cost)[:top]
    rows = []
    for r in ordered:
        cache = f"{r.cache_hit_ratio:.0%}" if r.cache_hit_ratio is not None else "\u2014"
        rows.append([
            r.key,
            f"{r.calls:,}",
            human_amount(r.cost),
            f"{r.avg_cost_per_call}",
            f"{r.total_tokens:,}",
            cache,
        ])
    return rows


_ROLLUP_HEADERS = ["key", "calls", "cost", "avg/call", "tokens", "cache"]
_ROLLUP_ALIGN = "lrrrrr"


def text_report(
    usage: UsageSet,
    *,
    inr_rate: Optional[Decimal] = None,
    budgets: Sequence[BudgetStatus] = (),
    waste: Optional[Sequence[Finding]] = None,
    top: int = 10,
) -> str:
    """The whole picture, in plain text, sized for a terminal."""
    summary = usage.summary()
    findings = list(waste) if waste is not None else find_waste(usage)
    saving = total_potential_saving(findings)

    out: List[str] = []
    out.append("=" * 62)
    out.append("KHARCHA \u2014 AI spend report")
    out.append("=" * 62)

    total = usd(summary["total_cost"])
    out.append(f"Total spend        {format_usd(total)}"
               + (f"   ({format_inr(total * inr_rate)})" if inr_rate else ""))
    out.append(f"Calls              {summary['calls']:,}")
    out.append(f"Tokens             {summary['input_tokens'] + summary['output_tokens']:,}"
               f"  ({summary['input_tokens']:,} in / {summary['output_tokens']:,} out)")
    ratio = summary["cache_hit_ratio"]
    out.append(f"Cache hit rate     {ratio:.1%}" if ratio is not None
               else "Cache hit rate     \u2014")
    out.append(f"Models used        {summary['models_used']}")
    if summary["unattributed_calls"]:
        out.append(f"Unattributed       {summary['unattributed_calls']:,} calls "
                   f"(no team, no project)")

    daily = usage.daily_costs(30)
    if any(cost > 0 for _, cost in daily):
        out.append("")
        out.append("Last 30 days       " + sparkline([c for _, c in daily]))

    for label, rollups in (("By team", usage.by_team().values()),
                           ("By project", usage.by_project().values()),
                           ("By model", usage.by_model().values()),
                           ("By provider", usage.by_provider().values())):
        rows = _rollup_rows(rollups, top=top)
        if not rows:
            continue
        out.append("")
        out.append(label)
        out.append(table(rows, _ROLLUP_HEADERS, align=_ROLLUP_ALIGN))

    if budgets:
        out.append("")
        out.append("Budgets")
        rows = []
        for status in budgets:
            rows.append([
                status.budget.key,
                f"{format_usd(status.spent)} / {format_usd(status.budget.limit)}",
                _bar(float(status.fraction), 12),
                f"{status.fraction:.0%}",
                status.projection.as_text(),
                "OK" if status.on_track else "OVER",
            ])
        out.append(table(rows, ["key", "spent / limit", "progress", "used",
                                "projected", "status"],
                         align="llrrrl"))

    if findings:
        out.append("")
        out.append(f"Waste \u2014 estimated {format_usd(saving)} avoidable")
        for finding in findings[:10]:
            amount = (f"{format_usd(finding.estimated_saving)}"
                      if finding.estimated_saving else "review")
            out.append(f"  [{finding.severity:^6}] {amount:>10}  {finding.title}")
            out.append(f"           {finding.detail}")
    else:
        out.append("")
        out.append("Waste \u2014 nothing flagged.")

    if summary["unpriced_models"]:
        out.append("")
        out.append("WARNING: these models have no price and are counted as $0:")
        for model, count in summary["unpriced_models"].items():
            out.append(f"  {model} ({count} calls)")

    return "\n".join(out)


def markdown_report(
    usage: UsageSet,
    *,
    title: str = "AI spend report",
    inr_rate: Optional[Decimal] = None,
    budgets: Sequence[BudgetStatus] = (),
    waste: Optional[Sequence[Finding]] = None,
    top: int = 10,
) -> str:
    """The same numbers as :func:`text_report`, as Markdown."""
    summary = usage.summary()
    findings = list(waste) if waste is not None else find_waste(usage)
    saving = total_potential_saving(findings)
    total = usd(summary["total_cost"])

    out = [f"# {title}", ""]
    out.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| | |")
    out.append("|---|---|")
    out.append(f"| **Total spend** | **{format_usd(total)}**"
               + (f" ({format_inr(total * inr_rate)})" if inr_rate else "") + " |")
    out.append(f"| Calls | {summary['calls']:,} |")
    out.append(f"| Tokens | {summary['input_tokens'] + summary['output_tokens']:,} |")
    ratio = summary["cache_hit_ratio"]
    out.append(f"| Cache hit rate | {ratio:.1%} |" if ratio is not None
               else "| Cache hit rate | \u2014 |")
    out.append(f"| Models used | {summary['models_used']} |")
    out.append(f"| Unattributed calls | {summary['unattributed_calls']:,} |")
    out.append("")

    def _md_table(label: str, rollups) -> None:
        rows = _rollup_rows(rollups, top=top)
        if not rows:
            return
        out.append(f"## {label}")
        out.append("")
        out.append("| " + " | ".join(_ROLLUP_HEADERS) + " |")
        out.append("|" + "|".join(
            ["---" if a == "l" else "---:" for a in _ROLLUP_ALIGN]) + "|")
        for row in rows:
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    _md_table("By team", usage.by_team().values())
    _md_table("By project", usage.by_project().values())
    _md_table("By model", usage.by_model().values())

    if budgets:
        out.append("## Budgets")
        out.append("")
        out.append("| key | spent / limit | used | projected | status |")
        out.append("|---|---|---:|---:|:--|")
        for status in budgets:
            # computed outside the f-string: backslash escapes are not allowed
            # inside f-string expressions before Python 3.12
            flag = "OK" if status.on_track else "OVER"
            out.append(
                f"| {status.budget.key} "
                f"| {format_usd(status.spent)} / {format_usd(status.budget.limit)} "
                f"| {status.fraction:.0%} "
                f"| {status.projection.as_text()} "
                f"| {flag} |")
        out.append("")

    if findings:
        out.append(f"## Waste — ~{format_usd(saving)} avoidable")
        out.append("")
        out.append("| severity | est. saving | calls | finding |")
        out.append("|---|---:|---:|---|")
        for f in findings:
            amount = (format_usd(f.estimated_saving)
                      if f.estimated_saving else "review")
            out.append(f"| {f.severity} | {amount} | {f.affected_calls} | "
                       f"**{f.title}** — {f.detail} |")
        out.append("")
        out.append("> Estimates are heuristic. Each rule explains how it "
                   "computed its number — check before acting.")
        out.append("")

    return "\n".join(out)
