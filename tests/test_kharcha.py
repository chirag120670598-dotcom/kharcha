"""Tests for kharcha.

No network, no API keys, no cloud account. The whole point of the project is
that it runs on one file on your own disk, so the tests do too.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from kharcha.adapters.importers import (
    import_anthropic,
    import_any,
    import_jsonl,
    import_litellm,
    import_openai,
)
from kharcha.cli import main
from kharcha.core.budgets import (
    Budget,
    burn_rate,
    check_all,
    check_budget,
    project_month,
)
from kharcha.core.money import (
    format_inr,
    format_usd,
    human_amount,
    round_cents,
    round_micros,
    usd,
)
from kharcha.core.pricing import DEFAULT_RATES, ModelRate, PricingTable
from kharcha.core.redact import find_sensitive, prompt_key, redact
from kharcha.core.report import (
    markdown_report,
    sparkline,
)
from kharcha.core.report import table as render_table
from kharcha.core.report import text_report
from kharcha.core.store import UsageStore
from kharcha.core.usage import UsageRecord, UsageSet, unknown_model_report
from kharcha.core.waste import (
    detect_cheap_work_on_flagship,
    detect_duplicate_prompts,
    detect_low_cache_hit,
    detect_oversized_context,
    detect_retry_storms,
    detect_unattributed_spend,
    detect_unpriced_models,
    find_waste,
    total_potential_saving,
)


def rec(model="gpt-4o", *, ts=None, inp=1000, out=200, cached=0, team="team-a",
        project="chatbot", key="", cost=None, **kw):
    return UsageRecord(
        ts=ts or datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
        model=model, input_tokens=inp, output_tokens=out,
        cached_input_tokens=cached, team=team, project=project,
        prompt_key=key, cost=None if cost is None else usd(cost), **kw)


@pytest.fixture
def table():
    return PricingTable.default()


# --- money ------------------------------------------------------------------

class TestMoney:
    def test_float_goes_through_str(self):
        assert usd(0.1) == Decimal("0.1")

    def test_none_is_zero(self):
        assert usd(None) == Decimal("0")

    def test_round_half_up_not_bankers(self):
        """Banker's rounding would give 1.00 here."""
        assert round_cents(Decimal("1.005")) == Decimal("1.01")
        assert round_cents(Decimal("1.015")) == Decimal("1.02")

    def test_micro_precision(self):
        assert round_micros(Decimal("0.0000005")) == Decimal("0.000001")

    def test_usd_format(self):
        assert format_usd(usd("1234.5678")) == "$1,234.57"
        assert format_usd(usd("0")) == "$0.00"
        assert format_usd(usd("-5")) == "-$5.00"

    def test_inr_uses_indian_grouping(self):
        """12,34,567 not 1,234,567 — this is the whole point of format_inr."""
        assert format_inr(usd("1234567.50")) == "\u20b912,34,567.50"
        assert format_inr(usd("100000")) == "\u20b91,00,000.00"
        assert format_inr(usd("999")) == "\u20b9999.00"

    def test_human_amount(self):
        assert human_amount(usd("1200000")) == "$1.2M"
        assert human_amount(usd("48300")) == "$48.3K"
        assert human_amount(usd("92")) == "$92"

    def test_no_float_drift_over_many_calls(self):
        total = sum((usd("0.1") for _ in range(10)), Decimal("0"))
        assert total == Decimal("1.0")


# --- pricing ----------------------------------------------------------------

class TestPricing:
    def test_known_model_prices(self, table):
        cost = table.cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert cost == Decimal("2.50")

    def test_output_priced_separately(self, table):
        cost = table.cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert cost == Decimal("10.00")

    def test_cached_tokens_are_cheaper(self, table):
        full = table.cost("gpt-4o", input_tokens=1_000_000)
        cached = table.cost("gpt-4o", input_tokens=1_000_000,
                            cached_input_tokens=1_000_000)
        assert cached < full
        assert cached == Decimal("1.25")

    def test_cached_tokens_not_billed_twice(self, table):
        """Cached must be *removed* from the full-price input, not added on."""
        mixed = table.cost("gpt-4o", input_tokens=1_000_000,
                           cached_input_tokens=500_000)
        expected = Decimal("1.25") + Decimal("0.625")   # 0.5M full + 0.5M cached
        assert mixed == expected

    def test_cached_cannot_exceed_input(self, table):
        with pytest.raises(ValueError):
            table.cost("gpt-4o", input_tokens=10, cached_input_tokens=20)

    def test_negative_tokens_rejected(self, table):
        with pytest.raises(ValueError):
            table.cost("gpt-4o", input_tokens=-1)

    def test_unknown_model_returns_none_not_zero(self, table):
        """Silently pricing an unknown model at zero is how dashboards lie."""
        assert table.cost("gpt-9000", input_tokens=1000) is None

    def test_gateway_prefix_stripped(self, table):
        assert table.get("openai/gpt-4o") is not None
        assert table.get("azure/gpt-4o") is not None

    def test_case_insensitive(self, table):
        assert table.get("GPT-4o") is not None

    def test_dated_snapshot_falls_back(self, table):
        assert table.get("gpt-4o-2024-08-06") is not None

    def test_rate_validation(self):
        with pytest.raises(ValueError):
            ModelRate("x", usd("-1"), usd("1"))
        with pytest.raises(ValueError):
            ModelRate("x", usd("1"), usd("1"), cached_input=usd("5"))
        with pytest.raises(ValueError):
            ModelRate("", usd("1"), usd("1"))

    def test_override_wins(self, table):
        table.set_price("gpt-4o", input="1.00", output="2.00")
        assert table.cost("gpt-4o", input_tokens=1_000_000) == Decimal("1.00")

    def test_roundtrip_through_file(self, tmp_path, table):
        path = tmp_path / "pricing.json"
        table.to_file(path)
        reloaded = PricingTable.from_file(path)
        assert reloaded.get("gpt-4o").input == table.get("gpt-4o").input

    def test_override_file_layers_on_defaults(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"models": [
            {"model": "my-model", "input": "3", "output": "9"}]}))
        t = PricingTable.from_file(path, base=PricingTable.default())
        assert t.get("my-model").input == Decimal("3")
        assert t.get("gpt-4o") is not None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PricingTable.from_file(tmp_path / "nope.json")

    def test_staleness_warning(self, table):
        assert table.staleness_warning(today=date(2027, 1, 1)) is not None
        assert table.staleness_warning(today=date(2026, 2, 1)) is None

    def test_every_default_rate_is_sane(self):
        for model, rate in DEFAULT_RATES.items():
            assert rate.input >= 0, model
            assert rate.output >= 0, model
            if rate.cached_input is not None:
                assert rate.cached_input <= rate.input, model


# --- usage records ----------------------------------------------------------

class TestUsageRecord:
    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError):
            UsageRecord(ts=datetime(2026, 1, 1), model="gpt-4o")

    def test_cached_cannot_exceed_input(self):
        with pytest.raises(ValueError):
            rec(inp=10, cached=20)

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValueError):
            rec(inp=-5)
        with pytest.raises(ValueError):
            rec(out=-5)

    def test_model_required(self):
        with pytest.raises(ValueError):
            UsageRecord(ts=datetime(2026, 1, 1, tzinfo=timezone.utc), model="")

    def test_fresh_input_tokens(self):
        assert rec(inp=100, cached=30).fresh_input_tokens == 70

    def test_cache_hit_ratio(self):
        assert rec(inp=100, cached=25).cache_hit_ratio == 0.25
        assert rec(inp=0).cache_hit_ratio is None

    def test_roundtrip(self):
        r = rec(key="abc", cost="1.23", latency_ms=450)
        back = UsageRecord.from_dict(r.to_dict())
        assert back.model == r.model
        assert back.cost == Decimal("1.23")
        assert back.latency_ms == 450


class TestUsageSet:
    def test_total_cost_computed(self, table):
        usage = UsageSet([rec(inp=1_000_000)], table)
        assert usage.total_cost() == Decimal("2.50")

    def test_provider_cost_wins_over_computed(self, table):
        """The provider's own number is what you are actually billed."""
        usage = UsageSet([rec(inp=1_000_000, cost="9.99")], table)
        assert usage.total_cost() == Decimal("9.99")

    def test_unpriced_model_counts_as_zero_but_is_reported(self, table):
        usage = UsageSet([rec("gpt-9000", inp=1_000_000)], table)
        assert usage.total_cost() == Decimal("0")
        assert unknown_model_report(usage.records, table) == {"gpt-9000": 1}

    def test_by_team(self, table):
        usage = UsageSet([rec(team="a"), rec(team="a"), rec(team="b")], table)
        assert usage.by_team()["a"].calls == 2
        assert usage.by_team()["b"].calls == 1

    def test_unattributed_grouped(self, table):
        usage = UsageSet([rec(team="", project="")], table)
        assert "(unattributed)" in usage.by_team()

    def test_by_day_is_sorted_and_zero_filled(self, table):
        usage = UsageSet([
            rec(ts=datetime(2026, 3, 10, 12, tzinfo=timezone.utc)),
            rec(ts=datetime(2026, 3, 8, 12, tzinfo=timezone.utc)),
        ], table)
        daily = usage.daily_costs(5, now=datetime(2026, 3, 11, tzinfo=timezone.utc))
        assert len(daily) == 5
        # the 9th has no data but must still appear
        assert daily[1][1] == Decimal("0")

    def test_daily_costs_zero_fills(self, table):
        usage = UsageSet([], table)
        daily = usage.daily_costs(7)
        assert len(daily) == 7
        assert all(cost == Decimal("0") for _, cost in daily)

    def test_between_rejects_bad_range(self, table):
        usage = UsageSet([], table)
        now = datetime(2026, 3, 10, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            usage.between(now, now)

    def test_last_days_rejects_zero(self, table):
        with pytest.raises(ValueError):
            UsageSet([], table).last_days(0)

    def test_summary_shape(self, table):
        s = UsageSet([rec(), rec(team="", project="")], table).summary()
        assert s["calls"] == 2
        assert s["unattributed_calls"] == 1
        assert "total_cost" in s

    def test_project_only_still_counts_as_attributed(self, table):
        """Some orgs only use projects — no team is not the same as no owner."""
        s = UsageSet([rec(team="")], table).summary()
        assert s["unattributed_calls"] == 0

    def test_rollup_cache_ratio(self, table):
        usage = UsageSet([rec(inp=100, cached=50), rec(inp=100, cached=0)], table)
        assert usage.by_model()["gpt-4o"].cache_hit_ratio == 0.25

    def test_avg_cost_per_call_zero_safe(self, table):
        usage = UsageSet([], table)
        assert usage.by_team() == {}


# --- budgets ----------------------------------------------------------------

class TestBudgets:
    def test_burn_rate(self):
        assert burn_rate(Decimal("100"), 10) == Decimal("10")

    def test_burn_rate_zero_days(self):
        assert burn_rate(Decimal("100"), 0) == Decimal("0")

    def test_projection_linear(self):
        # April has 30 days: on the 15th, elapsed == remaining, so the
        # projection is exactly double the spend so far.
        p = project_month(Decimal("100"), today=date(2026, 4, 15))
        assert p.linear == Decimal("200")

    def test_projection_formula(self):
        p = project_month(Decimal("100"), today=date(2026, 3, 11))
        # 11 days elapsed of 31: 100 + (100/11)*20
        expected = Decimal("100") + (Decimal("100") / 11) * 20
        assert abs(p.linear - expected) < Decimal("0.01")

    def test_projection_weekday_weighted(self):
        # weekdays cost 20, weekends cost 0
        costs = [Decimal(20) if date(2026, 3, 1 + i).weekday() < 5 else Decimal(0)
                 for i in range(10)]
        p = project_month(Decimal("140"), today=date(2026, 3, 11), daily_costs=costs)
        assert p.weekday_weighted != p.linear

    def test_projection_range_is_ordered(self):
        p = project_month(Decimal("50"), today=date(2026, 3, 5))
        assert p.low <= p.high

    def test_budget_validation(self):
        with pytest.raises(ValueError):
            Budget("x", Decimal("0"))
        with pytest.raises(ValueError):
            Budget("", Decimal("100"))
        with pytest.raises(ValueError):
            Budget("x", Decimal("100"), thresholds=())

    def test_threshold_crossed(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("85"),
                              today=date(2026, 3, 10))
        assert status.worst_alert is not None
        assert status.worst_alert.level == "critical"

    def test_no_alert_when_under(self):
        status = check_budget(Budget("t", Decimal("1000")), Decimal("100"),
                              today=date(2026, 3, 10))
        assert status.worst_alert is None
        assert status.on_track

    def test_exceeded(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("150"),
                              today=date(2026, 3, 10))
        assert status.worst_alert.level == "exceeded"
        assert not status.on_track

    def test_projected_overspend_flagged_early(self):
        """Under every threshold today, but on track to blow the limit."""
        # 40% used on day 10 of 31 -> no threshold fires, but 40 x 31/10 = 124
        status = check_budget(Budget("t", Decimal("100")), Decimal("40"),
                              today=date(2026, 3, 10))
        assert not [a for a in status.alerts if a.kind == "threshold"]
        projection_alerts = [a for a in status.alerts if a.kind == "projection"]
        assert projection_alerts
        assert "on track to overspend" in projection_alerts[0].message
        assert not status.on_track

    def test_threshold_wins_tie_against_projection(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("90"),
                              today=date(2026, 3, 10))
        assert status.worst_alert.kind == "threshold"
        assert status.worst_alert.level == "critical"

    def test_daily_allowance(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("50"),
                              today=date(2026, 3, 21))
        # 50 left over 10 days
        assert status.daily_allowance == Decimal("5.00")

    def test_daily_allowance_at_month_end(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("50"),
                              today=date(2026, 3, 31))
        assert status.daily_allowance == Decimal("0")

    def test_check_all_sorted_worst_first(self):
        budgets = [Budget("a", Decimal("100")), Budget("b", Decimal("100"))]
        statuses = check_all(budgets, {"a": Decimal("10"), "b": Decimal("90")},
                             today=date(2026, 3, 10))
        assert statuses[0].budget.key == "b"

    def test_status_to_dict(self):
        status = check_budget(Budget("t", Decimal("100")), Decimal("10"),
                              today=date(2026, 3, 10))
        d = status.to_dict()
        assert d["key"] == "t"
        assert "projection" in d


# --- waste ------------------------------------------------------------------

class TestWaste:
    def test_duplicate_prompts_detected(self, table):
        ts = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=ts + timedelta(minutes=i), inp=20_000, out=100, key="same")
             for i in range(5)], table)
        findings = detect_duplicate_prompts(usage)
        assert len(findings) == 1
        assert findings[0].estimated_saving > 0

    def test_duplicates_below_threshold_ignored(self, table):
        usage = UsageSet([rec(inp=20_000, key="same") for _ in range(2)], table)
        assert detect_duplicate_prompts(usage) == []

    def test_duplicates_outside_window_ignored(self, table):
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(days=i), inp=20_000, key="same")
             for i in range(5)], table)
        assert detect_duplicate_prompts(usage) == []

    def test_no_prompt_key_means_no_duplicate_finding(self, table):
        usage = UsageSet([rec(inp=20_000) for _ in range(10)], table)
        assert detect_duplicate_prompts(usage) == []

    def test_low_cache_hit_detected(self, table):
        usage = UsageSet(
            [rec("claude-sonnet-4", inp=30_000, out=500, cached=0)
             for _ in range(5)], table)
        findings = detect_low_cache_hit(usage)
        assert findings and findings[0].estimated_saving > 0

    def test_good_cache_hit_not_flagged(self, table):
        usage = UsageSet(
            [rec("claude-sonnet-4", inp=30_000, out=500, cached=25_000)
             for _ in range(5)], table)
        assert detect_low_cache_hit(usage) == []

    def test_models_without_cache_rate_skipped(self, table):
        usage = UsageSet([rec("gpt-4-turbo", inp=30_000, out=500) for _ in range(5)],
                         table)
        assert detect_low_cache_hit(usage) == []

    def test_oversized_context_detected(self, table):
        usage = UsageSet([rec(inp=50_000, out=30) for _ in range(10)], table)
        findings = detect_oversized_context(usage)
        assert findings and findings[0].estimated_saving > 0

    def test_normal_ratio_not_flagged(self, table):
        usage = UsageSet([rec(inp=2_000, out=1_000) for _ in range(10)], table)
        assert detect_oversized_context(usage) == []

    def test_zero_output_skipped(self, table):
        """Division by zero guard."""
        usage = UsageSet([rec(inp=50_000, out=0) for _ in range(10)], table)
        assert detect_oversized_context(usage) == []

    def test_cheap_work_on_flagship(self, table):
        usage = UsageSet(
            [rec("claude-opus-4", inp=1_000, out=5) for _ in range(20)], table)
        findings = detect_cheap_work_on_flagship(usage)
        assert findings and findings[0].estimated_saving > 0

    def test_long_output_on_flagship_not_flagged(self, table):
        usage = UsageSet(
            [rec("claude-opus-4", inp=1_000, out=2_000) for _ in range(20)], table)
        assert detect_cheap_work_on_flagship(usage) == []

    def test_mini_models_not_flagged(self, table):
        usage = UsageSet([rec("gpt-4o-mini", inp=1_000, out=5) for _ in range(20)],
                         table)
        assert detect_cheap_work_on_flagship(usage) == []

    def test_retry_storm_detected(self, table):
        base = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(seconds=i * 30), inp=5_000, out=200,
                 key="storm", cost="0.10") for i in range(6)], table)
        findings = detect_retry_storms(usage)
        assert findings
        assert findings[0].affected_calls == 5   # all but the first

    def test_spread_out_calls_not_a_storm(self, table):
        base = datetime(2026, 3, 10, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(hours=i), inp=5_000, out=200, key="k")
             for i in range(6)], table)
        assert detect_retry_storms(usage) == []

    def test_unattributed_spend(self, table):
        usage = UsageSet([rec(team="", project="")], table)
        findings = detect_unattributed_spend(usage)
        assert findings and findings[0].estimated_saving == 0

    def test_attributed_spend_not_flagged(self, table):
        assert detect_unattributed_spend(UsageSet([rec()], table)) == []

    def test_unpriced_models_flagged(self, table):
        usage = UsageSet([rec("gpt-9000")], table)
        findings = detect_unpriced_models(usage)
        assert findings and findings[0].severity == "high"

    def test_find_waste_sorted_by_saving(self, table):
        base = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(minutes=i), inp=40_000, out=30, key="dup")
             for i in range(6)]
            + [rec("gpt-9000")], table)
        findings = find_waste(usage)
        savings = [f.estimated_saving for f in findings]
        assert savings == sorted(savings, reverse=True)

    def test_total_potential_saving(self, table):
        base = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(minutes=i), inp=40_000, out=30, key="dup")
             for i in range(6)], table)
        assert total_potential_saving(find_waste(usage)) > 0

    def test_empty_usage_no_crash(self, table):
        assert find_waste(UsageSet([], table)) == []

    def test_every_finding_explains_itself(self, table):
        """A saving you cannot audit is a saving nobody will act on."""
        base = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        usage = UsageSet(
            [rec(ts=base + timedelta(minutes=i), inp=40_000, out=30, key="dup")
             for i in range(6)]
            + [rec("gpt-9000"), rec(team="", project="")], table)
        for f in find_waste(usage):
            assert f.how_computed, f.rule
            assert f.title and f.detail


# --- store ------------------------------------------------------------------

class TestStore:
    def test_add_and_read(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.add(rec())
            assert store.count() == 1
            assert store.all()[0].model == "gpt-4o"

    def test_cost_stored_as_text_not_float(self, tmp_path):
        """REAL would reintroduce exactly the float error this project avoids."""
        with UsageStore(tmp_path / "t.db") as store:
            store.add(rec(cost="0.1"))
            store.add(rec(cost="0.2"))
            rows = store.all()
        assert sum((r.cost for r in rows), Decimal("0")) == Decimal("0.3")

    def test_add_many(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            assert store.add_many([rec() for _ in range(100)]) == 100
            assert store.count() == 100

    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "t.db"
        with UsageStore(path) as store:
            store.add(rec())
        with UsageStore(path) as store:
            assert store.count() == 1

    def test_since_filters(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.add(rec(ts=datetime(2026, 3, 1, tzinfo=timezone.utc)))
            store.add(rec(ts=datetime(2026, 3, 20, tzinfo=timezone.utc)))
            rows = store.since(datetime(2026, 3, 10, tzinfo=timezone.utc))
        assert len(rows) == 1

    def test_budget_roundtrip(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.set_budget("team-a", "500", scope="team")
            budgets = store.budgets()
        assert budgets[0]["key"] == "team-a"
        assert budgets[0]["limit_usd"] == Decimal("500")

    def test_budget_upsert(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.set_budget("team-a", "500")
            store.set_budget("team-a", "800")
            assert len(store.budgets()) == 1
            assert store.budgets()[0]["limit_usd"] == Decimal("800")

    def test_delete_budget(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.set_budget("team-a", "500")
            assert store.delete_budget("team-a")
            assert not store.delete_budget("team-a")

    def test_clear(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.add_many([rec() for _ in range(5)])
            store.clear()
            assert store.count() == 0

    def test_jsonl_export_reimportable(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            store.add(rec(cost="1.50"))
            dump = store.to_jsonl()
        result = import_jsonl(dump)
        assert result.imported == 1
        assert result.records[0].cost == Decimal("1.50")

    def test_in_memory(self):
        with UsageStore(":memory:") as store:
            store.add(rec())
            assert store.count() == 1

    def test_schema_version(self, tmp_path):
        with UsageStore(tmp_path / "t.db") as store:
            assert store.schema_version == 1


# --- importers --------------------------------------------------------------

class TestImporters:
    def test_jsonl_import(self):
        line = json.dumps(rec().to_dict())
        result = import_jsonl(f"{line}\n{line}")
        assert result.imported == 2
        assert result.failures == 0

    def test_jsonl_skips_bad_lines_with_reason(self):
        good = json.dumps(rec().to_dict())
        result = import_jsonl(f"{good}\nnot json\n")
        assert result.imported == 1
        assert result.failures == 1
        assert "JSON" in result.skipped[0][1]

    def test_jsonl_reports_invalid_records(self):
        """A record missing 'ts' is reported, not silently dropped."""
        result = import_jsonl('{"model": "x"}\n')
        assert result.imported == 0
        assert result.failures == 1

    def test_jsonl_skips_blank_lines(self):
        assert import_jsonl("\n\n").imported == 0

    def test_openai_import(self):
        payload = json.dumps({
            "id": "chatcmpl-1", "model": "gpt-4o", "created": 1770000000,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                      "prompt_tokens_details": {"cached_tokens": 300}},
        })
        result = import_openai(payload)
        assert result.imported == 1
        r = result.records[0]
        assert r.input_tokens == 1000
        assert r.cached_input_tokens == 300
        assert r.request_id == "chatcmpl-1"

    def test_openai_skips_missing_usage(self):
        result = import_openai('{"model": "gpt-4o"}')
        assert result.imported == 0
        assert "no usage" in result.skipped[0][1]

    def test_openai_skips_missing_model(self):
        result = import_openai('{"usage": {"prompt_tokens": 1}}')
        assert result.imported == 0

    def test_anthropic_cache_split(self):
        """Anthropic's input_tokens is the FRESH count — summing is required
        or cache-hit ratios come out above 100%."""
        payload = json.dumps({
            "id": "msg_1", "model": "claude-sonnet-4",
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "cache_read_input_tokens": 900,
                      "cache_creation_input_tokens": 0},
        })
        r = import_anthropic(payload).records[0]
        assert r.input_tokens == 1000
        assert r.cached_input_tokens == 900
        assert r.cache_hit_ratio == 0.9

    def test_litellm_keeps_reported_cost(self):
        payload = json.dumps({
            "model": "gpt-4o", "prompt_tokens": 500, "completion_tokens": 50,
            "total_cost": 0.0123, "team": "team-a", "end_user": "proj-x",
            "request_id": "req-1", "response_time": 850,
        })
        r = import_litellm(payload).records[0]
        assert r.team == "team-a"
        assert r.project == "proj-x"
        assert r.latency_ms == 850

    def test_sniffing_anthropic(self):
        payload = json.dumps({"model": "claude-sonnet-4",
                              "usage": {"input_tokens": 1,
                                        "cache_read_input_tokens": 5}})
        assert import_any(payload).source == "anthropic"

    def test_sniffing_openai(self):
        payload = json.dumps({"model": "gpt-4o",
                              "usage": {"prompt_tokens": 1}})
        assert import_any(payload).source == "openai"

    def test_sniffing_litellm(self):
        payload = json.dumps({"model": "gpt-4o", "total_cost": 0.01})
        assert import_any(payload).source == "litellm"

    def test_sniffing_falls_back_to_jsonl(self):
        line = json.dumps(rec().to_dict())
        assert import_any(line).source == "jsonl"

    def test_millisecond_timestamps(self):
        payload = json.dumps({"model": "gpt-4o", "created": 1770000000000,
                              "usage": {"prompt_tokens": 1}})
        r = import_openai(payload).records[0]
        assert r.ts.year >= 2026


# --- redaction --------------------------------------------------------------

class TestRedaction:
    def test_pan(self):
        assert "[PAN REDACTED]" in redact("pan AAAPL1234C here")

    def test_email(self):
        assert redact("mail ravi@example.com").count("[EMAIL REDACTED]") == 1

    def test_aadhaar(self):
        assert "[AADHAAR REDACTED]" in redact("aadhaar 2345 6789 0124")

    def test_phone(self):
        assert "[PHONE REDACTED]" in redact("call 9876543210 now")

    def test_url_with_token(self):
        assert "[URL REDACTED]" in redact("see https://x.com/a?token=abc")

    def test_api_key(self):
        assert "[TOKEN REDACTED]" in redact("api_key=sk-1234567890")

    def test_long_digits(self):
        assert "[NUMBER REDACTED]" in redact("order 12345678901")

    def test_empty_is_safe(self):
        assert redact("") == ""

    def test_prompt_key_stable(self):
        assert prompt_key("hello") == prompt_key("hello")

    def test_prompt_key_differs(self):
        assert prompt_key("hello") != prompt_key("hello!")

    def test_prompt_key_short(self):
        assert len(prompt_key("x")) == 16

    def test_prompt_key_salted(self):
        assert prompt_key("x", salt="a") != prompt_key("x", salt="b")

    def test_prompt_key_empty(self):
        assert prompt_key("") == ""

    def test_find_sensitive(self):
        assert "pan" in find_sensitive("AAAPL1234C")
        assert find_sensitive("nothing here") == []

    def test_redaction_does_not_eat_normal_text(self):
        assert redact("the quick brown fox") == "the quick brown fox"


# --- reports ----------------------------------------------------------------

class TestReports:
    @pytest.fixture
    def usage(self, table):
        base = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        records = [rec("gpt-4o", ts=base + timedelta(hours=i), inp=2_000,
                       out=300, team="team-a", project="chat")
                   for i in range(10)]
        records += [rec("claude-opus-4", ts=base, inp=5_000, out=1_000,
                        team="team-b", project="reports")]
        records += [rec("gpt-9000", ts=base, team="team-c", project="x")]
        return UsageSet(records, table)

    def test_text_report_contains_totals(self, usage):
        out = text_report(usage)
        assert "Total spend" in out
        assert "By team" in out
        assert "By model" in out

    def test_text_report_warns_about_unpriced(self, usage):
        assert "no price" in text_report(usage)
        assert "gpt-9000" in text_report(usage)

    def test_text_report_inr(self, usage):
        assert "\u20b9" in text_report(usage, inr_rate=Decimal("83"))

    def test_markdown_report_valid(self, usage):
        out = markdown_report(usage)
        assert out.startswith("# ")
        assert "| " in out
        assert "## By team" in out

    def test_text_and_markdown_agree(self, usage):
        """A dashboard and a report that disagree is how you lose the argument."""
        total = usage.total_cost()
        assert format_usd(total) in text_report(usage)
        assert format_usd(total) in markdown_report(usage)

    def test_empty_usage_reports_cleanly(self, table):
        assert "Total spend" in text_report(UsageSet([], table))

    def test_sparkline(self):
        assert sparkline([Decimal(1), Decimal(5), Decimal(3)]) == "\u2581\u2588\u2584"
        assert sparkline([]) == ""
        assert sparkline([Decimal(2), Decimal(2)]) == "\u2584\u2584"

    def test_table_alignment(self):
        out = render_table([["a", "1"], ["bb", "22"]], ["key", "n"], align="lr")
        assert "key" in out
        # right-aligned numbers line up on their last digit
        lines = out.splitlines()
        assert lines[-1].endswith("22")
        assert lines[-2].endswith("1")


# --- CLI --------------------------------------------------------------------

class TestCLI:
    def run(self, argv, db):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--db", str(db)] + argv)
        return code, buf.getvalue()

    def test_demo_then_report(self, tmp_path):
        db = tmp_path / "t.db"
        code, _ = self.run(["demo", "--days", "5"], db)
        assert code == 0
        code, out = self.run(["report"], db)
        assert code == 0
        assert "Total spend" in out

    def test_report_on_empty_db(self, tmp_path):
        code, out = self.run(["report"], tmp_path / "e.db")
        assert code == 1
        assert "no usage data" in out

    def test_waste_finds_things(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["demo", "--days", "7"], db)
        code, out = self.run(["waste"], db)
        assert code == 0
        assert "avoidable" in out

    def test_waste_json(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["demo", "--days", "7"], db)
        code, out = self.run(["waste", "--json"], db)
        assert code == 0
        assert "findings" in json.loads(out)

    def test_budget_set_and_list(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["demo", "--days", "5"], db)
        code, _ = self.run(["budget", "set", "team-a", "500"], db)
        assert code == 0
        code, out = self.run(["budget", "list"], db)
        assert code == 0
        assert "team-a" in out

    def test_budget_rm(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["budget", "set", "team-a", "500"], db)
        code, _ = self.run(["budget", "rm", "team-a"], db)
        assert code == 0
        code, _ = self.run(["budget", "rm", "team-a"], db)
        assert code == 1

    def test_budget_list_empty(self, tmp_path):
        code, out = self.run(["budget", "list"], tmp_path / "e.db")
        assert code == 0
        assert "no budgets" in out

    def test_import_file(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text(json.dumps(rec().to_dict()) + "\n")
        code, out = self.run(["import", str(path)], tmp_path / "t.db")
        assert code == 0
        assert "imported 1" in out

    def test_import_missing_file(self, tmp_path):
        code, _ = self.run(["import", str(tmp_path / "nope.jsonl")],
                           tmp_path / "t.db")
        assert code == 2

    def test_report_markdown(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["demo", "--days", "3"], db)
        code, out = self.run(["report", "--markdown"], db)
        assert code == 0
        assert out.startswith("# ")

    def test_report_inr(self, tmp_path):
        db = tmp_path / "t.db"
        self.run(["demo", "--days", "3"], db)
        code, out = self.run(["report", "--inr-rate", "83.5"], db)
        assert code == 0
        assert "\u20b9" in out


class TestWindowsConsoleEncoding:
    """Windows consoles default to cp1252, which cannot encode \u20b9 or \u2192.

    This crashed the CLI smoke test on windows-latest in CI, and would crash
    for any Windows user running the same command.
    """

    def test_force_utf8_stdio_does_not_raise(self):
        """pytest's capture objects lack reconfigure(); the guard must cope."""
        from kharcha.cli import _force_utf8_stdio
        _force_utf8_stdio()

    def test_guard_fixes_a_cp1252_stream(self):
        import io
        import sys
        from kharcha.cli import _force_utf8_stdio

        # the failure mode is real: cp1252 cannot encode a rupee sign
        broken = io.TextIOWrapper(open(2, "wb", closefd=False),
                                  encoding="cp1252", newline="")
        with pytest.raises(UnicodeEncodeError):
            broken.write("\u20b9")
            broken.flush()

        guarded = io.TextIOWrapper(open(2, "wb", closefd=False),
                                   encoding="cp1252", newline="")
        original = sys.stdout
        sys.stdout = guarded
        try:
            _force_utf8_stdio()
            sys.stdout.write("TOTAL \u20b96165.00 \u2192 paid")
            sys.stdout.flush()
        finally:
            sys.stdout = original
