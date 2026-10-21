"""Log importers.

Every provider and gateway logs usage differently. These parsers turn the
common shapes into :class:`~kharcha.core.usage.UsageRecord` so the engine
never has to know where the data came from.

Supported:

* **JSONL** — one usage object per line (the interchange format)
* **OpenAI** — ``{"usage": {"prompt_tokens": …, "completion_tokens": …}}``
* **Anthropic** — ``{"usage": {"input_tokens": …, "output_tokens": …,
  "cache_read_input_tokens": …}}``
* **LiteLLM / generic proxy** — ``{"model", "prompt_tokens", "total_cost"}``

Anything unrecognised is reported in :attr:`ImportResult.skipped` with the
reason. Silently dropping lines is how a dashboard starts under-reporting.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..core.money import usd
from ..core.usage import UsageRecord

__all__ = ["ImportResult", "import_jsonl", "import_openai", "import_anthropic",
           "import_litellm", "import_any", "import_file"]


@dataclass
class ImportResult:
    """What an import produced, including what it could not use."""

    records: List[UsageRecord] = field(default_factory=list)
    skipped: List[Tuple[int, str]] = field(default_factory=list)
    source: str = ""

    @property
    def imported(self) -> int:
        return len(self.records)

    @property
    def failures(self) -> int:
        return len(self.skipped)

    def summary(self) -> str:
        parts = [f"{self.imported} imported"]
        if self.skipped:
            parts.append(f"{self.failures} skipped")
        return ", ".join(parts)


def _parse_ts(value: Any, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # unix seconds or milliseconds — 13+ digits is milliseconds
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return default
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _dec(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return usd(value)
    except Exception:
        return None


def import_jsonl(raw: Union[str, Iterable[str]], *, source: str = "jsonl") -> ImportResult:
    """Parse the interchange format: one :class:`UsageRecord` dict per line."""
    lines = raw.splitlines() if isinstance(raw, str) else list(raw)
    result = ImportResult(source=source)
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            result.records.append(UsageRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            result.skipped.append((number, f"{type(exc).__name__}: {exc}"))
    return result


def import_openai(raw: Union[str, Iterable[str]], *, default_team: str = "",
                  default_project: str = "", source: str = "openai") -> ImportResult:
    """Parse OpenAI chat-completion response bodies.

    Expects the shape returned by ``POST /v1/chat/completions``, optionally
    wrapped in a log line with ``created`` / ``model`` at the top level.
    """
    lines = raw.splitlines() if isinstance(raw, str) else list(raw)
    result = ImportResult(source=source)
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            result.skipped.append((number, f"bad JSON: {exc}"))
            continue

        usage = data.get("usage") or {}
        model = data.get("model", "")
        if not model:
            result.skipped.append((number, "no model field"))
            continue
        if not usage:
            result.skipped.append((number, "no usage field"))
            continue

        try:
            result.records.append(UsageRecord(
                ts=_parse_ts(data.get("created"), datetime.now(timezone.utc)),
                model=model,
                input_tokens=_int(usage.get("prompt_tokens")),
                output_tokens=_int(usage.get("completion_tokens")),
                cached_input_tokens=_int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                team=default_team,
                project=default_project,
                cost=_dec(data.get("total_cost") or data.get("cost_usd")),
                request_id=data.get("id", ""),
            ))
        except ValueError as exc:
            result.skipped.append((number, str(exc)))
    return result


def import_anthropic(raw: Union[str, Iterable[str]], *, default_team: str = "",
                     default_project: str = "", source: str = "anthropic") -> ImportResult:
    """Parse Anthropic message response bodies.

    Anthropic splits cache usage into ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens``. Only the *read* portion is billed at the
    discounted rate, and ``input_tokens`` is the **fresh** count — so the
    record's ``input_tokens`` must be the sum, or cache-hit ratios come out
    above 100%.
    """
    lines = raw.splitlines() if isinstance(raw, str) else list(raw)
    result = ImportResult(source=source)
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            result.skipped.append((number, f"bad JSON: {exc}"))
            continue

        usage = data.get("usage") or {}
        model = data.get("model", "")
        if not model:
            result.skipped.append((number, "no model field"))
            continue
        if not usage:
            result.skipped.append((number, "no usage field"))
            continue

        fresh = _int(usage.get("input_tokens"))
        cached = _int(usage.get("cache_read_input_tokens"))
        created = _int(usage.get("cache_creation_input_tokens"))
        try:
            result.records.append(UsageRecord(
                ts=_parse_ts(data.get("created_at"), datetime.now(timezone.utc)),
                model=model,
                input_tokens=fresh + cached + created,
                output_tokens=_int(usage.get("output_tokens")),
                cached_input_tokens=cached,
                team=default_team,
                project=default_project,
                cost=_dec(data.get("total_cost")),
                request_id=data.get("id", ""),
            ))
        except ValueError as exc:
            result.skipped.append((number, str(exc)))
    return result


def import_litellm(raw: Union[str, Iterable[str]], *, source: str = "litellm") -> ImportResult:
    """Parse LiteLLM proxy spend logs.

    LiteLLM already computed the cost, so it is kept as the authoritative
    number rather than recomputed from the rate card.
    """
    lines = raw.splitlines() if isinstance(raw, str) else list(raw)
    result = ImportResult(source=source)
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            result.skipped.append((number, f"bad JSON: {exc}"))
            continue

        model = data.get("model") or data.get("response_model") or ""
        if not model:
            result.skipped.append((number, "no model field"))
            continue
        try:
            result.records.append(UsageRecord(
                ts=_parse_ts(data.get("startTime") or data.get("call_time"),
                             datetime.now(timezone.utc)),
                model=model,
                input_tokens=_int(data.get("prompt_tokens")),
                output_tokens=_int(data.get("completion_tokens")),
                team=data.get("team", "") or "",
                project=data.get("end_user", "") or data.get("project", "") or "",
                cost=_dec(data.get("total_cost") or data.get("spend")),
                request_id=data.get("request_id") or data.get("id") or "",
                latency_ms=_int(data.get("response_time")) or None,
            ))
        except ValueError as exc:
            result.skipped.append((number, str(exc)))
    return result


_PARSERS = (
    ("openai", import_openai),
    ("anthropic", import_anthropic),
    ("litellm", import_litellm),
    ("jsonl", import_jsonl),
)


def import_any(raw: Union[str, Iterable[str]], *, source: str = "auto") -> ImportResult:
    """Sniff the format and use the right parser.

    Detection is by field names on the first non-empty line, not by file
    extension, because logs get renamed.
    """
    text = raw if isinstance(raw, str) else "\n".join(raw)
    first = next((l for l in text.splitlines() if l.strip()), "")
    try:
        sample = json.loads(first)
    except json.JSONDecodeError:
        return import_jsonl(text, source="jsonl")

    keys = set(sample)
    usage = sample.get("usage") if isinstance(sample.get("usage"), dict) else {}

    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        return import_anthropic(text, source="anthropic")
    if "prompt_tokens" in usage:
        return import_openai(text, source="openai")
    if "total_cost" in keys or "spend" in keys or "prompt_tokens" in keys:
        return import_litellm(text, source="litellm")
    return import_jsonl(text, source="jsonl")


def import_file(path: Union[str, Path], *, format: str = "auto") -> ImportResult:
    """Import from a file on disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    raw = path.read_text(encoding="utf-8")
    parser = _PARSERS[-1][1] if format == "auto" else dict(_PARSERS)[format]
    if format == "auto":
        return import_any(raw)
    return parser(raw, source=format)
