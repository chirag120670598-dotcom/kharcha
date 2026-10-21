"""Model pricing.

Rates are per **1M tokens**, which is how every provider publishes them.
Storing per-token rates would need 18 significant digits to be exact.

.. warning::
   Prices change often. The bundled table below carries a ``verified_on``
   date and :func:`PricingTable.staleness_warning` will tell you when it is
   old enough that you should check it. You can override any model with a
   local ``pricing.json`` — do that rather than editing this file, so an
   upgrade does not lose your numbers.

   If a model is not in the table and has no override, the cost is reported
   as ``None`` and flagged — **never silently zero**. A dashboard that quietly
   under-reports is worse than one that says "I don't know".

Pure standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from .money import usd

__all__ = [
    "ModelRate",
    "PricingTable",
    "PRICING_VERIFIED_ON",
    "STALE_AFTER_DAYS",
    "DEFAULT_RATES",
]

#: When the bundled rates were last checked against provider price pages.
PRICING_VERIFIED_ON = date(2026, 1, 15)

#: After this many days, warn that the bundled table may be stale.
STALE_AFTER_DAYS = 90


@dataclass(frozen=True)
class ModelRate:
    """Cost per 1M tokens for one model.

    Attributes:
        model: canonical model id, e.g. ``gpt-4o``.
        input: cost per 1M input tokens.
        output: cost per 1M output tokens.
        cached_input: cost per 1M tokens served from a prompt cache. Many
            providers charge ~10% of the input rate here, which is why
            measuring cache hits is worth real money.
        provider: vendor, used for grouping in reports.
        family: coarse grouping (``flagship``, ``mid``, ``mini``) used by the
            waste detector to spot expensive models doing cheap work.
    """

    model: str
    input: Decimal
    output: Decimal
    cached_input: Optional[Decimal] = None
    provider: str = "unknown"
    family: str = "mid"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model id is required")
        for label, value in (("input", self.input), ("output", self.output)):
            if value < 0:
                raise ValueError(f"{label} rate for {self.model} cannot be negative")
        if self.cached_input is not None and self.cached_input < 0:
            raise ValueError(f"cached_input for {self.model} cannot be negative")
        if self.cached_input is not None and self.cached_input > self.input:
            raise ValueError(
                f"cached_input ({self.cached_input}) exceeds input ({self.input}) "
                f"for {self.model} — a cache should never cost more"
            )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input": str(self.input),
            "output": str(self.output),
            "cached_input": None if self.cached_input is None else str(self.cached_input),
            "provider": self.provider,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelRate":
        return cls(
            model=data["model"],
            input=usd(data["input"]),
            output=usd(data["output"]),
            cached_input=None if data.get("cached_input") is None
            else usd(data["cached_input"]),
            provider=data.get("provider", "unknown"),
            family=data.get("family", "mid"),
        )


def _r(model, provider, family, inp, out, cached=None) -> ModelRate:
    return ModelRate(
        model=model, provider=provider, family=family,
        input=usd(inp), output=usd(out),
        cached_input=None if cached is None else usd(cached),
    )


#: Bundled rate card. **Check these against provider price pages** — see the
#: module docstring. Keys are lowercase canonical ids.
DEFAULT_RATES: Dict[str, ModelRate] = {
    r.model: r for r in (
        # OpenAI
        _r("gpt-4o", "openai", "flagship", "2.50", "10.00", "1.25"),
        _r("gpt-4o-mini", "openai", "mini", "0.15", "0.60", "0.075"),
        _r("gpt-4-turbo", "openai", "flagship", "10.00", "30.00"),
        _r("gpt-3.5-turbo", "openai", "mini", "0.50", "1.50"),
        _r("o1", "openai", "reasoning", "15.00", "60.00", "7.50"),
        _r("o1-mini", "openai", "reasoning", "3.00", "12.00", "1.50"),
        _r("text-embedding-3-large", "openai", "embedding", "0.13", "0.13"),
        _r("text-embedding-3-small", "openai", "embedding", "0.02", "0.02"),
        # Anthropic
        _r("claude-opus-4", "anthropic", "flagship", "15.00", "75.00"),
        _r("claude-sonnet-4", "anthropic", "mid", "3.00", "15.00", "0.30"),
        _r("claude-haiku-3.5", "anthropic", "mini", "0.80", "4.00", "0.08"),
        # Google
        _r("gemini-1.5-pro", "google", "flagship", "1.25", "5.00"),
        _r("gemini-1.5-flash", "google", "mini", "0.075", "0.30"),
        # Mistral
        _r("mistral-large", "mistral", "flagship", "2.00", "6.00"),
        _r("mistral-small", "mistral", "mini", "0.20", "0.60"),
        # Local / self-hosted
        _r("llama-3.1-70b", "local", "flagship", "0.00", "0.00"),
        _r("llama-3.1-8b", "local", "mini", "0.00", "0.00"),
    )
}


class PricingTable:
    """Rate card with overrides and prefix matching.

    Model ids drift between providers and gateways — ``gpt-4o-2024-08-06``,
    ``openai/gpt-4o``, ``azure/gpt-4o``. Rather than enumerate every variant,
    lookup strips known prefixes and falls back to the longest matching prefix,
    so dated snapshots still price correctly.
    """

    def __init__(
        self,
        rates: Optional[Dict[str, ModelRate]] = None,
        *,
        verified_on: date = PRICING_VERIFIED_ON,
    ) -> None:
        self.rates: Dict[str, ModelRate] = dict(rates if rates is not None else DEFAULT_RATES)
        self.verified_on = verified_on

    # -- construction ------------------------------------------------------

    @classmethod
    def default(cls) -> "PricingTable":
        return cls(DEFAULT_RATES)

    @classmethod
    def from_file(cls, path: Union[str, Path], *, base: Optional["PricingTable"] = None) -> "PricingTable":
        """Load overrides from JSON, layered on top of ``base``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"pricing file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("models", data) if isinstance(data, dict) else data
        if not isinstance(entries, (dict, list)):
            raise ValueError("pricing file must contain a list or an object of models")

        table = cls(dict(base.rates) if base else None,
                    verified_on=date.fromisoformat(data["verified_on"])
                    if isinstance(data, dict) and data.get("verified_on")
                    else (base.verified_on if base else PRICING_VERIFIED_ON))
        items = entries.values() if isinstance(entries, dict) else entries
        for item in items:
            rate = ModelRate.from_dict(item)
            table.rates[rate.model.lower()] = rate
        return table

    def to_file(self, path: Union[str, Path]) -> None:
        payload = {
            "verified_on": self.verified_on.isoformat(),
            "models": [r.to_dict() for r in self.rates.values()],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- mutation ----------------------------------------------------------

    def set(self, rate: ModelRate) -> "PricingTable":
        self.rates[rate.model.lower()] = rate
        return self

    def set_price(self, model: str, *, input: Union[str, float, Decimal],
                  output: Union[str, float, Decimal],
                  cached_input: Optional[Union[str, float, Decimal]] = None,
                  provider: str = "custom", family: str = "mid") -> "PricingTable":
        """Convenience wrapper for the common case."""
        return self.set(ModelRate(
            model=model, provider=provider, family=family,
            input=usd(input), output=usd(output),
            cached_input=None if cached_input is None else usd(cached_input),
        ))

    # -- lookup ------------------------------------------------------------

    def _normalise(self, model: str) -> str:
        name = (model or "").strip().lower()
        # strip gateway prefixes: "openai/gpt-4o" -> "gpt-4o"
        if "/" in name:
            name = name.rsplit("/", 1)[1]
        return name

    def get(self, model: str) -> Optional[ModelRate]:
        """Exact, then date-stripped, then longest-prefix match."""
        key = self._normalise(model)
        if not key:
            return None
        if key in self.rates:
            return self.rates[key]
        # dated snapshot: gpt-4o-2024-08-06
        for suffix_len in (len("-2024-08-06"),):
            stripped = key[: -suffix_len] if key[-suffix_len:-3].startswith("-2") else None
            if stripped and stripped in self.rates:
                return self.rates[stripped]
        parts = key.split("-")
        for i in range(len(parts) - 1, 0, -1):
            candidate = "-".join(parts[:i])
            if candidate in self.rates:
                return self.rates[candidate]
        return None

    def known_models(self) -> list[str]:
        return sorted(self.rates)

    def staleness_warning(self, *, today: Optional[date] = None) -> Optional[str]:
        """A warning string if the bundled rates are old, else ``None``."""
        today = today or date.today()
        age = (today - self.verified_on).days
        if age >= STALE_AFTER_DAYS:
            return (
                f"bundled pricing was verified on {self.verified_on.isoformat()} "
                f"({age} days ago) — check provider price pages and add a "
                f"pricing.json override"
            )
        return None

    # -- costing -----------------------------------------------------------

    def cost(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> Optional[Decimal]:
        """Total USD cost for one call, or ``None`` if the model is unknown.

        Cached tokens are billed at ``cached_input`` and are *removed* from the
        full-price input count, so they are not charged twice.
        """
        rate = self.get(model)
        if rate is None:
            return None
        if input_tokens < 0 or output_tokens < 0 or cached_input_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if cached_input_tokens > input_tokens:
            raise ValueError(
                f"cached_input_tokens ({cached_input_tokens}) cannot exceed "
                f"input_tokens ({input_tokens})"
            )

        per_m = Decimal(1_000_000)
        billed_input = input_tokens - cached_input_tokens
        total = (
            Decimal(billed_input) * rate.input / per_m
            + Decimal(output_tokens) * rate.output / per_m
        )
        if cached_input_tokens and rate.cached_input is not None:
            total += Decimal(cached_input_tokens) * rate.cached_input / per_m
        return total
