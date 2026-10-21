"""``kharcha`` command line.

Everything runs against a local SQLite file. No API keys, no cloud account, no
network — which is the entire point of the project.

Usage::

    kharcha import logs.jsonl
    kharcha import openai-responses.jsonl --format openai
    kharcha report
    kharcha report --inr-rate 83.5 --markdown > spend.md
    kharcha waste
    kharcha budget set team-a 500
    kharcha budgets
    kharcha demo            # no data needed — generates a sample month

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters.importers import import_file
from .core.budgets import Budget, check_all
from .core.money import format_inr, format_usd, usd
from .core.pricing import PricingTable
from .core.report import markdown_report, text_report
from .core.store import UsageStore
from .core.usage import UsageRecord, UsageSet
from .core.waste import find_waste, total_potential_saving

__all__ = ["main", "build_parser"]

FORMATS = ("auto", "jsonl", "openai", "anthropic", "litellm")


def _load(args) -> UsageSet:
    """Build a UsageSet from the store, optionally windowed."""
    store = UsageStore(args.db)
    try:
        records = store.all()
    finally:
        store.close()
    table = (PricingTable.from_file(args.pricing) if args.pricing
             else PricingTable.default())
    usage = UsageSet(records, table)
    if args.days:
        usage = usage.last_days(args.days)
    return usage


def _load_budgets(args) -> List[Budget]:
    store = UsageStore(args.db)
    try:
        rows = store.budgets()
    finally:
        store.close()
    return [Budget(row["key"], row["limit_usd"], scope=row["scope"])
            for row in rows]


# --- subcommands ------------------------------------------------------------

def cmd_import(args, out) -> int:
    result = import_file(args.path, format=args.format)
    store = UsageStore(args.db)
    try:
        store.add_many(result.records)
        total = store.count()
    finally:
        store.close()
    print(f"imported {result.imported} records from {args.path} "
          f"({result.source} format)", file=out)
    if result.skipped:
        print(f"skipped {result.failures}:", file=out)
        for number, reason in result.skipped[:10]:
            print(f"  line {number}: {reason}", file=out)
        if len(result.skipped) > 10:
            print(f"  ... and {len(result.skipped) - 10} more", file=out)
    print(f"{total} records now in {args.db}", file=out)
    return 0 if result.records else 1


def cmd_report(args, out) -> int:
    usage = _load(args)
    if not usage.records:
        print(f"no usage data in {args.db}. Try: kharcha import <file> "
              f"   (or: kharcha demo)", file=out)
        return 1

    budgets = []
    if _load_budgets(args):
        spend = {k: v.cost for k, v in usage.by_team().items()}
        spend.update({k: v.cost for k, v in usage.by_project().items()})
        budgets = check_all(_load_budgets(args), spend)

    inr_rate = usd(args.inr_rate) if args.inr_rate else None
    if args.markdown:
        print(markdown_report(usage, inr_rate=inr_rate, budgets=budgets), file=out)
    else:
        print(text_report(usage, inr_rate=inr_rate, budgets=budgets), file=out)
    return 0


def cmd_waste(args, out) -> int:
    usage = _load(args)
    if not usage.records:
        print(f"no usage data in {args.db}.", file=out)
        return 1

    findings = find_waste(usage)
    saving = total_potential_saving(findings)
    if not findings:
        print("nothing flagged.", file=out)
        return 0

    if args.json:
        print(json.dumps(
            {"estimated_saving": str(saving),
             "findings": [f.to_dict() for f in findings]},
            ensure_ascii=False, indent=2), file=out)
        return 0

    print(f"Estimated avoidable spend: {format_usd(saving)}"
          + (f"  ({format_inr(saving * usd(args.inr_rate))})" if args.inr_rate else ""))
    print()
    for finding in findings:
        amount = (format_usd(finding.estimated_saving)
                  if finding.estimated_saving else "review")
        print(f"[{finding.severity:^6}] {amount:>10}  {finding.title}")
        print(f"         {finding.detail}")
        print(f"         how: {finding.how_computed}")
        print()
    return 0


def cmd_budget(args, out) -> int:
    store = UsageStore(args.db)
    try:
        if args.budget_action == "set":
            store.set_budget(args.key, args.limit, scope=args.scope)
            print(f"budget for {args.key} set to {format_usd(usd(args.limit))}",
                  file=out)
        elif args.budget_action == "rm":
            if store.delete_budget(args.key):
                print(f"removed budget for {args.key}", file=out)
            else:
                print(f"no budget for {args.key}", file=out)
                return 1
        else:
            rows = store.budgets()
            if not rows:
                print("no budgets set. Try: kharcha budget set team-a 500", file=out)
                return 0
            usage = UsageSet(store.all(), PricingTable.default())
            spend = {k: v.cost for k, v in usage.by_team().items()}
            spend.update({k: v.cost for k, v in usage.by_project().items()})
            budgets = [Budget(r["key"], r["limit_usd"], scope=r["scope"])
                       for r in rows]
            for status in check_all(budgets, spend):
                flag = "OK" if status.on_track else "OVER"
                print(f"{status.budget.key:<24} "
                      f"{format_usd(status.spent)} / {format_usd(status.budget.limit)}"
                      f"  ({status.fraction:.0%})  "
                      f"projected {status.projection.as_text()}  [{flag}]", file=out)
                for alert in status.alerts:
                    print(f"    ! {alert.message}", file=out)
    finally:
        store.close()
    return 0


def cmd_demo(args, out) -> int:
    """Generate a realistic sample month so you can see it working."""
    random.seed(args.seed)
    now = datetime.now(timezone.utc)
    models = [
        ("claude-sonnet-4", "team-a", "chatbot", 40),
        ("gpt-4o-mini", "team-a", "classifier", 300),
        ("gpt-4o", "team-b", "reports", 25),
        ("claude-opus-4", "team-b", "analysis", 8),
        ("gemini-1.5-flash", "team-c", "summarise", 200),
        ("mystery-model-x", "team-c", "experiment", 15),
    ]
    records: List[UsageRecord] = []
    for day in range(args.days):
        ts_day = now - timedelta(days=day)
        weekend = ts_day.weekday() >= 5
        for model, team, project, calls in models:
            volume = calls // 5 if weekend else calls
            for i in range(volume):
                ts = ts_day.replace(hour=random.randint(0, 23),
                                    minute=random.randint(0, 59),
                                    second=random.randint(0, 59),
                                    microsecond=0)
                big_context = random.random() < 0.12
                in_tok = (random.randint(30_000, 60_000) if big_context
                          else random.randint(200, 3_000))
                records.append(UsageRecord(
                    ts=ts, model=model,
                    input_tokens=in_tok,
                    output_tokens=random.randint(10, 30) if "mini" in model or "flash" in model
                    else random.randint(200, 1_500),
                    # must never exceed the input — UsageRecord rejects that
                    cached_input_tokens=random.randint(0, min(400, in_tok)),
                    team=team, project=project,
                    prompt_key=f"key-{random.randint(1, 12)}",
                ))
    # an unattributed chunk and a retry storm, so the waste rules have something to find
    for i in range(30):
        records.append(UsageRecord(
            ts=now - timedelta(hours=random.randint(1, 40)), model="gpt-4o",
            input_tokens=random.randint(1_000, 4_000),
            output_tokens=random.randint(50, 300)))
    storm = now - timedelta(hours=3)
    for i in range(6):
        records.append(UsageRecord(
            ts=storm + timedelta(seconds=i * 20), model="claude-sonnet-4",
            input_tokens=8_000, output_tokens=400, team="team-a",
            prompt_key="storm-key"))

    store = UsageStore(args.db)
    try:
        store.add_many(records)
        total = store.count()
    finally:
        store.close()
    print(f"generated {len(records)} sample records over {args.days} days "
          f"into {args.db} ({total} total)", file=out)
    print("now run: kharcha report", file=out)
    return 0


# --- wiring -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kharcha",
        description="Self-hosted AI spend tracking. SQLite, INR, budgets, "
                    "waste detection. Zero dependencies.",
    )
    parser.add_argument("--version", action="version", version="kharcha 0.1.0")
    parser.add_argument("--db", default="kharcha.db",
                        help="SQLite file (default: kharcha.db)")
    parser.add_argument("--pricing", default="",
                        help="path to a pricing.json override file")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="import usage logs")
    p.add_argument("path")
    p.add_argument("--format", choices=FORMATS, default="auto")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="spend report")
    p.add_argument("--inr-rate", type=str, default="", help="INR per USD")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--days", type=int, default=0, help="limit to the last N days")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("waste", help="what could you be spending less on")
    p.add_argument("--inr-rate", type=str, default="")
    p.add_argument("--json", action="store_true")
    p.add_argument("--days", type=int, default=0)
    p.set_defaults(func=cmd_waste)

    p = sub.add_parser("budget", help="manage budgets")
    bsub = p.add_subparsers(dest="budget_action", required=True)
    bset = bsub.add_parser("set")
    bset.add_argument("key")
    bset.add_argument("limit")
    bset.add_argument("--scope", default="team")
    brm = bsub.add_parser("rm")
    brm.add_argument("key")
    blist = bsub.add_parser("list")
    blist.add_argument("days", nargs="?", type=int, default=0)
    p.set_defaults(func=cmd_budget, days=0)

    p = sub.add_parser("demo", help="generate a sample month of data")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, sys.stdout)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
