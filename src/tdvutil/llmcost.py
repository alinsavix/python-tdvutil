"""LLM model pricing database and cost-calculation utilities.

Re-exported from :mod:`_tdvutil.llmcost`.

Usage::

    from tdvutil.llmcost import get_model, LLM_Model, LLM_Cost, DeprecatedModelError

    model = get_model("gpt-4o")
    cost = model.cost(input_tokens=10_000, output_tokens=2_000)
    print(f"Total: ${cost.total_cost:.4f}")
"""

from _tdvutil.llmcost import (  # noqa: F401
    DeprecatedModelError,
    LLM_Cost,
    LLM_Model,
    LLM_Provider,
    get_model,
    get_models,
    get_providers,
    search_models,
)

__all__ = [
    "DeprecatedModelError",
    "LLM_Provider",
    "LLM_Cost",
    "LLM_Model",
    "get_model",
    "get_models",
    "get_providers",
    "search_models",
]
