"""Money handling.

Every amount in this project is :class:`decimal.Decimal`. Token bills are
small per call and huge in aggregate — a float rounding error of 1e-15 per
call across 40,000 calls a month is exactly the kind of thing that makes a
dashboard disagree with the provider's invoice by a few dollars, and once that
happens nobody trusts the dashboard.

Costs are stored and computed in **USD** (that is what the providers bill in)
and converted for display. Doing it the other way round would bake an
exchange rate into stored data and make historical reports shift whenever the
rate changes.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Union

__all__ = [
    "CENT",
    "USD_SYMBOL",
    "INR_SYMBOL",
    "Money",
    "usd",
    "round_cents",
    "round_micros",
    "format_usd",
    "format_inr",
    "convert_to_inr",
    "human_amount",
]

Money = Decimal

CENT = Decimal("0.01")
MICRO = Decimal("0.000001")
USD_SYMBOL = "$"
INR_SYMBOL = "\u20b9"

#: Indian digit grouping (lakh/crore) groups, largest first.
_INDIAN_GROUPS = (10_000_000, 100_000, 1_000)


def usd(value: Union[str, int, float, Decimal, None]) -> Decimal:
    """Coerce anything to an exact Decimal.

    Floats go through ``str`` so ``usd(0.1)`` is ``Decimal('0.1')`` and not
    ``Decimal('0.1000000000000000055511151231257827021181583404541015625')``.
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(str(value))


def round_cents(value: Decimal) -> Decimal:
    """Round to 2dp, half-up (not banker's rounding)."""
    return usd(value).quantize(CENT, rounding=ROUND_HALF_UP)


def round_micros(value: Decimal) -> Decimal:
    """Round to 6dp — the precision per-token costs actually need."""
    return usd(value).quantize(MICRO, rounding=ROUND_HALF_UP)


def convert_to_inr(amount: Decimal, rate: Union[str, float, Decimal]) -> Decimal:
    """Convert a USD amount to INR at ``rate`` rupees per dollar."""
    rate_dec = usd(rate)
    if rate_dec <= 0:
        raise ValueError("exchange rate must be positive")
    return round_cents(usd(amount) * rate_dec)


def _group_indian(integer_part: str) -> str:
    """``1234567`` → ``12,34,567`` (Indian grouping, not Western)."""
    if len(integer_part) <= 3:
        return integer_part
    last3 = integer_part[-3:]
    rest = integer_part[:-3]
    out = []
    while len(rest) > 2:
        out.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        out.insert(0, rest)
    return ",".join(out) + "," + last3


def _group_western(integer_part: str) -> str:
    out = []
    while len(integer_part) > 3:
        out.insert(0, integer_part[-3:])
        integer_part = integer_part[:-3]
    if integer_part:
        out.insert(0, integer_part)
    return ",".join(out)


def format_usd(amount: Decimal, *, decimals: int = 2) -> str:
    """>>> format_usd(usd("1234.5678"))
    '$1,234.57'
    """
    value = usd(amount)
    negative = value < 0
    value = abs(value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    integer_part, _, frac = str(value).partition(".")
    body = f"{_group_western(integer_part)}.{frac}" if frac else _group_western(integer_part)
    return ("-" if negative else "") + USD_SYMBOL + body


def format_inr(amount: Decimal, *, decimals: int = 2) -> str:
    """Indian digit grouping, because that is how the number will be read.

    >>> format_inr(usd("1234567.50"))
    '\u20b912,34,567.50'
    """
    value = usd(amount)
    negative = value < 0
    value = abs(value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    integer_part, _, frac = str(value).partition(".")
    body = f"{_group_indian(integer_part)}.{frac}" if frac else _group_indian(integer_part)
    return ("-" if negative else "") + INR_SYMBOL + body


def human_amount(amount: Decimal) -> str:
    """Compact form for dashboards: ``$1.2M``, ``$48.3K``, ``$92``."""
    value = usd(amount)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for threshold, suffix in ((Decimal("1e9"), "B"), (Decimal("1e6"), "M"),
                              (Decimal("1e3"), "K")):
        if value >= threshold:
            scaled = (value / threshold).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            return f"{sign}{USD_SYMBOL}{scaled}{suffix}"
    return f"{sign}{USD_SYMBOL}{value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)}"
