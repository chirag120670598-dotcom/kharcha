<div align="center">

# 💸 kharcha

### *"AI ka bill kahan ja raha hai?"*

**Self-hosted AI spend tracking. One SQLite file, budgets that warn you
*before* you overspend, and a waste detector that tells you which calls you
should not have made.**

Zero dependencies · INR + USD · GST-ready numbers · MIT

[![CI](https://github.com/chirag120670598-dotcom/kharcha/actions/workflows/ci.yml/badge.svg)](https://github.com/chirag120670598-dotcom/kharcha/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![Tests](https://img.shields.io/badge/tests-143%20passing-success)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Try it in 30 seconds

```bash
git clone https://github.com/chirag120670598-dotcom/kharcha
cd kharcha

python -m kharcha.cli demo --days 20      # no API keys needed
python -m kharcha.cli report --inr-rate 83.5
python -m kharcha.cli waste
```

Real output:

```
==============================================================
KHARCHA — AI spend report
==============================================================
Total spend        $49.30   (₹4,116.55)
Calls              8,970
Tokens             60,988,750  (59,689,087 in / 1,299,663 out)
Cache hit rate     2.9%

Last 30 days       ▁▁▁▁▁▁▁▁▁▁▅▁▂▅▆█▆▅▁▁▇▇▇▆▇▁▂▆▆▅

By team
key             calls  cost  avg/call  tokens      cache
──────────────  ─────  ────  ────────  ──────────  ─────
team-b            498   $25  0.050450   3,256,271     3%
team-a          5,174   $22  0.004330  35,023,623     3%

Budgets
key     spent / limit       progress      used  projected  status
──────  ────────────────  ────────────  ──────  ─────────  ──────
team-b  $25.12 / $200.00  ██░░░░░░░░░░    13%     $37.08  OK

Waste — estimated $25.63 avoidable
  [medium]      $7.36  claude-sonnet-4: cache hit rate 3%
  [medium]      $4.21  claude-sonnet-4: avg 46,790 in / 701 out
  [ low  ]      $0.15  6 calls with the same prompt in 5 min

WARNING: these models have no price and are counted as $0:
  mystery-model-x (228 calls)
```

---

## Why another one

There are excellent LLM gateways. They are not what this is:

| | Gateways / SaaS | **kharcha** |
|---|---|---|
| Where your data lives | their cloud | **one file on your disk** |
| Setup | proxy every call | **tail a log file** |
| Currency | USD only | **USD *and* ₹ with Indian digit grouping** |
| Budgets | hard limits | **burn-rate projection with a range** |
| The question it answers | "what did I spend?" | **"what did I waste?"** |
| Dependencies | a runtime | **`dependencies = []`** |

`kharcha` is not a proxy. It does not sit in your request path, so it cannot
add latency, cannot go down and take your product with it, and cannot see your
prompts. You point it at logs you already have.

### The waste detector is the point

A pie chart tells you the bill was big. These rules tell you *why*:

| Rule | What it catches | Typical cause |
|---|---|---|
| `duplicate-prompts` | same prompt hash sent N times in 24h | retry loop, no cache, re-running per page load |
| `low-cache-hit` | 20k+ input tokens, <20% from cache | system prompt re-sent in full every call |
| `oversized-context` | 40,000 in → 30 out | whole file stuffed in when an excerpt would do |
| `cheap-work-on-flagship` | Opus/o1 producing ≤40 tokens | classification on a $75/1M model |
| `retry-storm` | same prompt ≥4 times in 5 min | no backoff, or a double-clicked button |
| `unattributed-spend` | calls with no team and no project | money nobody owns, so nobody cuts |
| `unpriced-model` | models missing from the rate card | **your dashboard is under-reporting** |
| `off-hours` | big share of spend 00:00–06:00 UTC | unplanned nightly loop |

Every finding prints **how it computed the number**, so you can argue with it:

```
[medium]      $7.36  claude-sonnet-4: cache hit rate 3%
       3,405,809 input tokens across 614 calls with only 3% served from cache.
       how: assumed 80% of input is cacheable, priced at 3.00 vs 0.30 per 1M tokens
```

A saving you cannot audit is a saving nobody acts on.

---

## Install

```bash
pip install kharcha
```

Or run from a clone — nothing to build, nothing to install.

## Usage

```bash
# import logs you already have
kharcha import logs.jsonl
kharcha import openai-responses.jsonl --format openai
kharcha import anthropic-messages.jsonl --format anthropic
kharcha import litellm-spend.jsonl   --format litellm

# reports
kharcha report
kharcha report --inr-rate 83.5 --days 30
kharcha report --markdown > spend.md

# waste
kharcha waste
kharcha waste --json

# budgets
kharcha budget set team-a 500
kharcha budget set project-chatbot 200 --scope project
kharcha budget list
kharcha budget rm team-a
```

### As a library

```python
from kharcha.core.usage import UsageSet
from kharcha.core.budgets import Budget, check_budget
from kharcha.core.waste import find_waste

usage = UsageSet.from_store(...)          # or UsageSet(records)
for finding in find_waste(usage):
    print(finding.title, finding.estimated_saving)

status = check_budget(Budget("team-a", 500), usage.by_team()["team-a"].cost)
print(status.on_track, status.projected_overage)
```

---

## How it stays honest

### Costs are `Decimal`, never `float`

Token bills are tiny per call and enormous in aggregate. A per-call rounding
error of 1e-15 across 40,000 calls a month is exactly what makes a dashboard
disagree with the provider's invoice — and the moment that happens, nobody
trusts the dashboard.

Costs are stored in SQLite as **TEXT**, not REAL. REAL is an IEEE float and
would silently reintroduce the error this project exists to prevent. There is
a test that sums `0.1 + 0.2` from the database and asserts it equals exactly
`0.3`.

### Unknown models are never priced at zero

If a model is not in the rate card and has no override, its cost is `None`,
the dashboard reports `$0`, **and a loud warning lists every such model**.
Quietly under-reporting is the one failure mode that destroys trust.

### Prices are checked, not assumed

The bundled rate card carries a `verified_on` date, and
`staleness_warning()` tells you when it is old enough to check. Override
anything with a `pricing.json`:

```json
{ "verified_on": "2026-01-15",
  "models": [{ "model": "your-internal-model", "input": "1.50", "output": "6.00" }] }
```

**Check the bundled rates against provider price pages before you rely on
them.** They drift.

### Budgets project a range, not a false point

AI spend is spiky — a batch job on Tuesday, nothing at the weekend. So
`kharcha` reports both a linear projection and a weekday-weighted one, and
shows the span between them:

```
team-b  $25.12 / $200.00  ██░░░░░░░░░░  13%   $37.08–$41.20  OK
```

And it warns *before* you overspend: 40% used on day 10 of 31 is under every
threshold, but it projects to 124% of the limit.

### Prompts are never stored

`prompt_key` is a SHA-256 hash. Duplicate detection works on the hash. If you
keep samples for debugging, `kharcha.core.redact` scrubs PAN, Aadhaar, GSTIN,
UPI VPA, IFSC, email, phone, card-shaped numbers, URLs and bearer tokens
first — and its own docstring says plainly that a regex redactor is a
best-effort defence, not a guarantee.

---

## Architecture

```
kharcha/
├── core/                  ← the engine. ZERO dependencies.
│   ├── money.py           Decimal, INR grouping, formatting
│   ├── pricing.py         rate card, overrides, prefix matching
│   ├── usage.py           records + rollups by team/model/day/provider
│   ├── budgets.py         burn rate, projection, alerts
│   ├── waste.py           8 detection rules
│   ├── store.py           SQLite (costs as TEXT, not REAL)
│   ├── redact.py          PII scrubbing
│   └── report.py          text + Markdown rendering
├── adapters/
│   └── importers.py       ← the ONLY file that knows provider log shapes
└── cli.py
```

`core/` imports only the standard library, and CI AST-parses it to enforce
that. Provider log formats change constantly; when they do, exactly one file
changes.

### Importers

| Format | Detects | Notes |
|---|---|---|
| `jsonl` | the interchange format | round-trips with `kharcha`'s own export |
| `openai` | `usage.prompt_tokens` | reads `prompt_tokens_details.cached_tokens` |
| `anthropic` | `cache_read_input_tokens` | **sums** fresh + cache_read + cache_creation, because Anthropic's `input_tokens` is the *fresh* count — summing wrong puts cache-hit ratios above 100% |
| `litellm` | `total_cost` / `spend` | keeps the proxy's cost as authoritative |

Format is sniffed from field names, not file extensions, because logs get
renamed. Unparseable lines are reported with a line number and a reason — never
silently dropped.

---

## Tests

```bash
pip install -e ".[dev]"
pytest            # 143 tests, ~1s
```

Things locked down:

- **`0.1 + 0.2 == 0.3`** after a round trip through SQLite
- **cached tokens are not billed twice** — removed from full-price input, then
  added at the cache rate
- **cached tokens can never exceed input tokens** — enforced at the record,
  the rate card and the store
- **unknown models return `None`, not `0`**
- **Anthropic's cache split is summed correctly**
- **`₹12,34,567.50` not `$1,234,567.50`** — Indian digit grouping
- **half-up rounding, not banker's**
- **naive datetimes are rejected** — "today" must not be ambiguous
- **every waste finding explains how it computed its number**
- **zero-division guards** on cache ratios, burn rates and per-call averages

---

## What is NOT here yet (honest roadmap)

| | |
|---|---|
| **Web UI** | CLI and Markdown only. `UsageSet` is shaped for a Flask/React front end, but none ships. |
| **Live log tailing** | You import files. There is no `--follow`. |
| **Alert delivery** | `BudgetStatus.alerts` is computed; nothing emails or Slacks them. |
| **Multi-tenant** | One file, one org. |
| **Rate-card freshness** | Bundled prices are dated `2026-01-15` and **have not been re-verified**. Check them. |
| **Non-OpenAI-shaped providers** | Bedrock, Vertex and Azure response shapes are not parsed (Azure works via the OpenAI path). |

## Roadmap

- [ ] `kharcha import --follow` for live tailing
- [ ] Slack/email budget alerts
- [ ] Flask + React dashboard
- [ ] Per-user chargeback export
- [ ] Postgres backend behind the same `UsageStore` interface
- [ ] Bedrock and Vertex importers
- [ ] Anomaly detection on daily spend (no ML — just robust statistics)

## Contributing

The most useful contributions are **importers** for providers not covered yet,
and **new waste rules**. A rule needs three things: what it detects, how the
saving is computed, and a test.

## License

MIT.

---

<div align="center">

**Made in Rajkot, Gujarat 🇮🇳** — *kharcha* means "expense". The bill should
fit in one file you own.

</div>
