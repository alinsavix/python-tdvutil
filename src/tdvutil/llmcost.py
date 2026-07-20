"""LLM model pricing database and cost-calculation utilities.

Re-exported from :mod:`_tdvutil.llmcost`.

Usage::

    from tdvutil.llmcost import get_price, calculate_cost, DeprecatedModelError

    price = get_price("claude-sonnet-5")
    cost = calculate_cost("gpt-4o", input_tokens=10_000, output_tokens=2_000)
"""

from _tdvutil.llmcost import (  # noqa: F401
    ALIASES,
    PRICES,
    PROVIDERS,
    DeprecatedModelError,
    calculate_cost,
    cheapest,
    compare_models,
    find_model,
    get_price,
    list_models,
    search,
)

__all__ = [
    "PRICES",
    "ALIASES",
    "PROVIDERS",
    "DeprecatedModelError",
    "find_model",
    "get_price",
    "list_models",
    "calculate_cost",
    "compare_models",
    "cheapest",
    "search",
]
