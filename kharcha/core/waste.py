"""Waste detection — the part that actually saves money.

A spend dashboard that only *shows* you the bill is a receipt. One that tells
you which ₹40,000 was avoidable is a tool.

Every rule here is a plain heuristic over the usage log. No model, no
clustering, no cloud. Each finding carries:

* **what** it found, with the specific records,
* **how much** it estimates would be saved, and
* **how** the estimate was computed — so you can argue with it.

The estimates are deliberately conservative and always explained. A saving
you cannot audit is a saving you will not act on.

Pure standard library.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .money import format_usd, round_cents, usd
from .pricing import PricingTable
from .usage import UsageRecord, UsageSet

__all__ = [
    "Finding",
    "RULES",
    "detect_duplicate_prompts",
    "detect_low_cache_hit",
    "detect_oversized_context",
    "detect_cheap_work_on_flagship",
    "detect_retry_storms",
    "detect_unattributed_spend",
    "detect_unpriced_models",
    "detect_off_hours",
    "find_waste",
]


@dataclass
class Finding:
    """One piece of avoidable spend."""

    rule: str
    severity: str          # "low" | "medium" | "high"
    title: str
    detail: str
    estimated_saving: Decimal = Decimal("0")
    affected_calls: int = 0
    examples: tuple = ()
    how_computed: str = ""

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "estimated_saving": str(round_cents(self.estimated_saving)),
            "affected_calls": self.affected_calls,
            "examples": list(self.examples),
            "how_computed": self.how_computed,
        }


def _severity_for(saving: Decimal, total: Decimal) -> str:
    if total <= 0:
        return "low"
    share = saving / total
    if share >= Decimal("0.15") or saving >= 100:
        return "high"
    if share >= Decimal("0.05") or saving >= 25:
        return "medium"
    return "low"


# --- rule 1: duplicate prompts ----------------------------------------------

def detect_duplicate_prompts(
    usage: UsageSet, *, min_repeats: int = 3, window: timedelta = timedelta(hours=24)
) -> List[Finding]:
    """Same prompt, sent many times, within a window.

    This is the single biggest waste category in practice: a retry loop, a
    missing cache, or the same summarisation re-run on every page load.

    Savings assume prompt caching would serve all but the first call at the
    ``cached_input`` rate, which is what providers actually charge.
    """
    groups: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in usage.records:
        if record.prompt_key:
            groups[record.prompt_key].append(record)

    findings: List[Finding] = []
    total = usage.total_cost()

    for key, records in groups.items():
        if len(records) < min_repeats:
            continue
        records.sort(key=lambda r: r.ts)
        span = records[-1].ts - records[0].ts
        if span > window:
            continue

        # what caching would have cost instead of repeating at full price
        saving = Decimal("0")
        first, rest = records[0], records[1:]
        rate = usage.table.get(first.model)
        if rate is None or rate.cached_input is None:
            # no cache rate known: assume a conservative 90% saving on input
            saving = sum(
                (usage.cost_of(r) * Decimal("0.9") for r in rest), Decimal("0"))
            how = ("no cache rate known for this model — assumed 90% of each "
                   "repeat call's cost is avoidable")
        else:
            saving = sum(
                (Decimal(r.fresh_input_tokens) * (rate.input - rate.cached_input)
                 / Decimal(1_000_000) for r in rest),
                Decimal("0"))
            how = ("repeats charged at the cached_input rate instead of the "
                   "full input rate")

        saving = round_cents(saving)
        if saving <= 0:
            continue
        findings.append(Finding(
            rule="duplicate-prompts",
            severity=_severity_for(saving, total),
            title=f"prompt repeated {len(records)}x within {span}",
            detail=(f"the same prompt key was sent {len(records)} times on "
                    f"model {first.model}. Cache it, memoise it, or fix the "
                    f"retry loop."),
            estimated_saving=saving,
            affected_calls=len(rest),
            examples=(key[:16],),
            how_computed=how,
        ))
    return findings


# --- rule 2: prompt caching not used ----------------------------------------

def detect_low_cache_hit(
    usage: UsageSet, *, threshold: float = 0.2, min_input_tokens: int = 2000
) -> List[Finding]:
    """Models with a big, uncached input where a cache rate exists.

    A long system prompt re-sent on every call is the classic case: 20k input
    tokens at full price when 90% of them are identical every time.
    """
    by_model: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in usage.records:
        rate = usage.table.get(record.model)
        if rate is not None and rate.cached_input is not None:
            by_model[record.model].append(record)

    findings: List[Finding] = []
    total = usage.total_cost()

    for model, records in by_model.items():
        input_tokens = sum(r.input_tokens for r in records)
        if input_tokens < min_input_tokens:
            continue
        cached = sum(r.cached_input_tokens for r in records)
        ratio = cached / input_tokens if input_tokens else 0.0
        if ratio >= threshold:
            continue

        rate = usage.table.get(model)
        delta = rate.input - (rate.cached_input or rate.input)
        cacheable = int(input_tokens * Decimal("0.8"))   # conservative
        saving = round_cents(Decimal(cacheable) * delta / Decimal(1_000_000))
        if saving <= 0:
            continue

        findings.append(Finding(
            rule="low-cache-hit",
            severity=_severity_for(saving, total),
            title=f"{model}: cache hit rate {ratio:.0%}",
            detail=(f"{input_tokens:,} input tokens across {len(records)} calls "
                    f"with only {ratio:.0%} served from cache. A repeated "
                    f"system prompt is the usual cause."),
            estimated_saving=saving,
            affected_calls=len(records),
            how_computed=(f"assumed 80% of input is cacheable, priced at "
                          f"{rate.input} vs {rate.cached_input} per 1M tokens"),
        ))
    return findings


# --- rule 3: oversized context ----------------------------------------------

def detect_oversized_context(
    usage: UsageSet, *, ratio_threshold: int = 40, min_input_tokens: int = 4000
) -> List[Finding]:
    """Calls with a huge input and a tiny output.

    ``40,000 in → 30 out`` usually means a whole file was stuffed into the
    prompt when a 200-line excerpt would have done.
    """
    by_model: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in usage.records:
        if record.input_tokens < min_input_tokens:
            continue
        if record.output_tokens == 0:
            continue
        if record.input_tokens / record.output_tokens >= ratio_threshold:
            by_model[record.model].append(record)

    findings: List[Finding] = []
    total = usage.total_cost()
    for model, records in by_model.items():
        rate = usage.table.get(model)
        if rate is None:
            continue
        # assume trimming to a quarter of the input still works
        saving = round_cents(sum(
            (Decimal(int(r.input_tokens * 0.75)) * rate.input / Decimal(1_000_000)
             for r in records), Decimal("0")))
        if saving <= 0:
            continue
        avg_in = sum(r.input_tokens for r in records) // len(records)
        avg_out = sum(r.output_tokens for r in records) // len(records)
        findings.append(Finding(
            rule="oversized-context",
            severity=_severity_for(saving, total),
            title=f"{model}: avg {avg_in:,} in / {avg_out} out",
            detail=(f"{len(records)} calls send a very large prompt for a very "
                    f"small answer. Chunk, filter or summarise the context "
                    f"before sending it."),
            estimated_saving=saving,
            affected_calls=len(records),
            how_computed="assumed the input could be trimmed by 75%",
        ))
    return findings


# --- rule 4: flagship model doing trivial work ------------------------------

def detect_cheap_work_on_flagship(
    usage: UsageSet, *, max_output_tokens: int = 40
) -> List[Finding]:
    """A flagship or reasoning model producing one-line answers.

    Classification, yes/no, short extraction — a mini model does these at a
    twentieth of the price and usually just as well.
    """
    by_model: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in usage.records:
        rate = usage.table.get(record.model)
        if rate is None or rate.family not in ("flagship", "reasoning"):
            continue
        if record.output_tokens and record.output_tokens <= max_output_tokens:
            by_model[record.model].append(record)

    findings: List[Finding] = []
    total = usage.total_cost()
    for model, records in by_model.items():
        rate = usage.table.get(model)
        # the cheapest mini model from the same provider as a stand-in
        candidates = [
            r for r in usage.table.rates.values()
            if r.provider == rate.provider and r.family == "mini"
        ]
        cheap = min(candidates, key=lambda r: r.input + r.output) if candidates else None
        actual = sum((usage.cost_of(r) for r in records), Decimal("0"))
        if cheap is None:
            # no cheaper sibling known: assume a 10x price gap
            saving = round_cents(actual * Decimal("0.9"))
            how = f"assumed a model 10x cheaper would do this work"
        else:
            cheaper_cost = sum(
                (Decimal(r.input_tokens) * cheap.input
                 + Decimal(r.output_tokens) * cheap.output) / Decimal(1_000_000)
                for r in records)
            saving = round_cents(actual - Decimal(cheaper_cost))
            how = f"priced the same tokens on {cheap.model} instead"
        if saving <= 0:
            continue

        findings.append(Finding(
            rule="cheap-work-on-flagship",
            severity=_severity_for(saving, total),
            title=f"{model}: {len(records)} calls producing \u2264{max_output_tokens} tokens",
            detail=(f"a {rate.family} model is being used for short answers. "
                    f"Try a mini model for these and keep the flagship for "
                    f"the hard calls."),
            estimated_saving=saving,
            affected_calls=len(records),
            how_computed=how,
        ))
    return findings


# --- rule 5: retry storms ---------------------------------------------------

def detect_retry_storms(
    usage: UsageSet, *, window: timedelta = timedelta(minutes=5), min_calls: int = 4
) -> List[Finding]:
    """The same prompt key fired several times in a few minutes.

    Almost always a retry loop without backoff, or a user clicking the button
    repeatedly because the first call was slow.
    """
    groups: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in usage.records:
        if record.prompt_key:
            groups[record.prompt_key].append(record)

    findings: List[Finding] = []
    total = usage.total_cost()
    for key, records in groups.items():
        records.sort(key=lambda r: r.ts)
        # sliding window: find the densest run
        best: List[UsageRecord] = []
        for i, start in enumerate(records):
            run = [r for r in records[i:] if r.ts - start.ts <= window]
            if len(run) > len(best):
                best = run
        if len(best) < min_calls:
            continue
        waste = round_cents(sum(
            (usage.cost_of(r) for r in best[1:]), Decimal("0")))
        if waste <= 0:
            continue
        findings.append(Finding(
            rule="retry-storm",
            severity=_severity_for(waste, total),
            title=f"{len(best)} calls with the same prompt in {int(window.total_seconds() // 60)} min",
            detail=("looks like a retry loop or a double-clicked button. Add "
                    "backoff, an idempotency key, or a client-side lock."),
            estimated_saving=waste,
            affected_calls=len(best) - 1,
            examples=(key[:16],),
            how_computed="all but the first call in the densest window",
        ))
    return findings


# --- rule 6: unattributed spend ---------------------------------------------

def detect_unattributed_spend(usage: UsageSet) -> List[Finding]:
    """Calls with no team and no project.

    Not a saving exactly — but money nobody owns is money nobody will ever
    cut. This finding is what makes chargeback possible.
    """
    orphans = [r for r in usage.records if not r.team and not r.project]
    if not orphans:
        return []
    cost = round_cents(sum((usage.cost_of(r) for r in orphans), Decimal("0")))
    total = usage.total_cost()
    return [Finding(
        rule="unattributed-spend",
        severity=_severity_for(cost, total),
        title=f"{cost} of spend has no owner",
        detail=(f"{len(orphans)} calls ({cost}) carry neither a team nor a "
                f"project. Nobody will cut spend they do not own."),
        estimated_saving=Decimal("0"),
        affected_calls=len(orphans),
        how_computed="no saving claimed — this is an attribution gap, not waste",
    )]


# --- rule 7: unpriced models ------------------------------------------------

def detect_unpriced_models(usage: UsageSet) -> List[Finding]:
    """Models the rate card does not know, so their cost is uncounted."""
    counts = usage.summary()["unpriced_models"]
    if not counts:
        return []
    detail = ", ".join(f"{m} ({n})" for m, n in list(counts.items())[:8])
    return [Finding(
        rule="unpriced-model",
        severity="high",
        title=f"{len(counts)} model(s) have no price in the rate card",
        detail=(f"their cost is being reported as zero, so this dashboard is "
                f"under-reporting your bill: {detail}. Add them to pricing.json."),
        estimated_saving=Decimal("0"),
        affected_calls=sum(counts.values()),
        how_computed="no saving claimed — this is a correctness gap",
    )]


# --- rule 8: off-hours ------------------------------------------------------

def detect_off_hours(
    usage: UsageSet, *, hours: Tuple[int, int] = (0, 6), share_threshold: float = 0.15
) -> List[Finding]:
    """A large share of spend between midnight and 6am UTC.

    Legitimate for batch jobs — but if it is 40% of the bill and nobody
    planned a nightly pipeline, something is looping.
    """
    start, end = hours
    off = [r for r in usage.records if start <= r.hour < end]
    if not off:
        return []
    off_cost = sum((usage.cost_of(r) for r in off), Decimal("0"))
    total = usage.total_cost()
    if total <= 0:
        return []
    share = float(off_cost / total)
    if share < share_threshold:
        return []
    return [Finding(
        rule="off-hours",
        severity="medium" if share >= 0.3 else "low",
        title=f"{share:.0%} of spend happens between {start:02d}:00 and {end:02d}:00 UTC",
        detail=(f"{len(off)} calls totalling {round_cents(off_cost)} ran "
                f"overnight. If that is a scheduled batch job, fine — if it "
                f"is not, something is looping."),
        estimated_saving=Decimal("0"),
        affected_calls=len(off),
        how_computed="no saving claimed — flagged for review, not assumed wasteful",
    )]


#: Every rule, in the order they run.
RULES = (
    detect_duplicate_prompts,
    detect_low_cache_hit,
    detect_oversized_context,
    detect_cheap_work_on_flagship,
    detect_retry_storms,
    detect_unattributed_spend,
    detect_unpriced_models,
    detect_off_hours,
)


def find_waste(usage: UsageSet, *, rules: Optional[Sequence] = None) -> List[Finding]:
    """Run every rule and return findings, biggest saving first."""
    rules = rules or RULES
    findings: List[Finding] = []
    for rule in rules:
        findings.extend(rule(usage))
    return sorted(findings, key=lambda f: (-f.estimated_saving, f.rule))


def total_potential_saving(findings: Iterable[Finding]) -> Decimal:
    """Sum of estimated savings across findings."""
    return round_cents(sum(
        (f.estimated_saving for f in findings), Decimal("0")))
