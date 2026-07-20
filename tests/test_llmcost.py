"""Tests for tdvutil.llmcost (LLM model pricing utilities)."""

import pytest

from _tdvutil.llmcost import (
    ALIASES,
    PRICES,
    DeprecatedModelError,
    calculate_cost,
    cheapest,
    compare_models,
    find_model,
    get_price,
    list_models,
    search,
)


# ---------------------------------------------------------------------------
# find_model
# ---------------------------------------------------------------------------

class TestFindModel:
    def test_exact_match(self) -> None:
        assert find_model("gpt-4o") == "gpt-4o"

    def test_alias_match(self) -> None:
        assert find_model("gpt4o") == "gpt-4o"

    def test_alias_case_insensitive(self) -> None:
        assert find_model("GPT4O") == "gpt-4o"

    def test_fuzzy_match(self) -> None:
        result = find_model("gemini-flash")
        # either the alias resolves directly or fuzzy finds a flash model
        assert result is not None
        assert "flash" in result

    def test_no_match_returns_none(self) -> None:
        assert find_model("zzz-no-match-zzz") is None

    def test_normalized_spaces(self) -> None:
        result = find_model("gemini flash")
        assert result is not None


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------

class TestGetPrice:
    def test_known_model(self) -> None:
        price = get_price("gpt-4o")
        assert price["model"] == "gpt-4o"
        assert price["provider"] == "openai"
        assert price["input"] == 2.50
        assert price["output"] == 10.00

    def test_alias_resolves(self) -> None:
        price = get_price("claude-sonnet")
        assert price["model"] == "claude-sonnet-5"

    def test_deprecated_raises_by_default(self) -> None:
        with pytest.raises(DeprecatedModelError, match="deprecated"):
            get_price("gpt-4-32k")

    def test_deprecated_allowed_with_flag(self) -> None:
        price = get_price("gpt-4-32k", include_deprecated=True)
        assert price["model"] == "gpt-4-32k"
        assert price["deprecated"] is True

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_price("nonexistent-model-xyz-abc")

    def test_returns_cached_input_if_present(self) -> None:
        price = get_price("claude-sonnet-5")
        assert "cached_input" in price
        assert price["cached_input"] == 0.20

    def test_model_key_in_result(self) -> None:
        price = get_price("gpt-4o")
        assert "model" in price

    def test_gpt35_deprecated(self) -> None:
        with pytest.raises(DeprecatedModelError):
            get_price("gpt-3.5-turbo")


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_list(self) -> None:
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_excludes_deprecated_by_default(self) -> None:
        models = list_models()
        deprecated = [m for m in models if m.get("deprecated")]
        assert len(deprecated) == 0

    def test_includes_deprecated_with_flag(self) -> None:
        all_models = list_models(include_deprecated=True)
        active = list_models(include_deprecated=False)
        assert len(all_models) > len(active)

    def test_provider_filter(self) -> None:
        anthropic = list_models("anthropic")
        assert all(m["provider"] == "anthropic" for m in anthropic)
        assert len(anthropic) > 0

    def test_provider_filter_case_insensitive(self) -> None:
        upper = list_models("ANTHROPIC")
        lower = list_models("anthropic")
        assert len(upper) == len(lower)

    def test_unknown_provider_returns_empty(self) -> None:
        result = list_models("no-such-provider")
        assert result == []

    def test_each_entry_has_model_key(self) -> None:
        for m in list_models():
            assert "model" in m


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:
    def test_basic_cost(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost["currency"] == "USD"
        assert cost["input_cost"] == pytest.approx(2.50, rel=1e-6)
        assert cost["output_cost"] == pytest.approx(10.00, rel=1e-6)
        assert cost["total_cost"] == pytest.approx(12.50, rel=1e-6)

    def test_zero_tokens(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost["total_cost"] == 0.0

    def test_cached_input_tokens(self) -> None:
        # gpt-4o cached_input = $1.25/1M
        cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0,
                               cached_input_tokens=1_000_000)
        assert cost["cached_cost"] == pytest.approx(1.25, rel=1e-6)

    def test_cache_write_tokens_anthropic(self) -> None:
        # claude-sonnet-5 cache_write = $2.50/1M
        cost = calculate_cost("claude-sonnet-5", input_tokens=0, output_tokens=0,
                               cache_write_tokens=1_000_000)
        assert cost["cache_write_cost"] == pytest.approx(2.50, rel=1e-6)

    def test_batch_pricing_anthropic(self) -> None:
        standard = calculate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
        batch = calculate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000,
                                use_batch=True)
        assert batch["total_cost"] < standard["total_cost"]

    def test_breakdown_string(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=100_000, output_tokens=20_000)
        assert "input" in cost["breakdown"]
        assert "output" in cost["breakdown"]
        assert "Total" in cost["breakdown"]

    def test_deprecated_model_raises(self) -> None:
        with pytest.raises(DeprecatedModelError):
            calculate_cost("gpt-4-32k", input_tokens=1_000, output_tokens=1_000)

    def test_deprecated_model_allowed_with_flag(self) -> None:
        cost = calculate_cost("gpt-4-32k", input_tokens=1_000_000, output_tokens=1_000_000,
                               include_deprecated=True)
        assert cost["total_cost"] > 0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            calculate_cost("zzz-no-such-model-zzz", input_tokens=1_000, output_tokens=1_000)

    def test_small_token_count(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=100, output_tokens=50)
        # 100/1M * 2.50 + 50/1M * 10.0 = 0.00025 + 0.0005 = 0.00075
        assert cost["total_cost"] == pytest.approx(0.00075, rel=1e-4)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_sorted_cheapest_first(self) -> None:
        results = compare_models(["gpt-4o", "gpt-4o-mini"], input_tokens=1_000_000, output_tokens=1_000_000)
        assert len(results) == 2
        assert results[0]["total_cost"] <= results[1]["total_cost"]

    def test_invalid_model_gets_inf(self) -> None:
        results = compare_models(["gpt-4o", "fake-xyz-model"])
        inf_entry = next(r for r in results if r.get("model") == "fake-xyz-model")
        assert inf_entry["total_cost"] == float("inf")

    def test_empty_list(self) -> None:
        assert compare_models([]) == []

    def test_single_model(self) -> None:
        results = compare_models(["gpt-4o"])
        assert len(results) == 1
        assert results[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# cheapest
# ---------------------------------------------------------------------------

class TestCheapest:
    def test_returns_n_results(self) -> None:
        results = cheapest(n=3)
        assert len(results) <= 3

    def test_sorted_cheapest_first(self) -> None:
        results = cheapest(n=5)
        costs = [r["total_cost"] for r in results]
        assert costs == sorted(costs)

    def test_provider_filter(self) -> None:
        results = cheapest(provider="openai", n=3)
        assert all(PRICES[str(r["model"])]["provider"] == "openai" for r in results)

    def test_excludes_deprecated_by_default(self) -> None:
        results = cheapest(n=20)
        for r in results:
            assert not PRICES[str(r["model"])].get("deprecated")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_finds_flash_models(self) -> None:
        results = search("flash")
        assert len(results) > 0
        assert all("flash" in str(r["model"]).lower() or "flash" in str(r.get("notes", "")).lower()
                   for r in results)

    def test_case_insensitive(self) -> None:
        lower = search("flash")
        upper = search("FLASH")
        assert len(lower) == len(upper)

    def test_no_results(self) -> None:
        results = search("zzznomatchzzz")
        assert results == []

    def test_includes_deprecated(self) -> None:
        # "2.0" only matches deprecated Gemini 2.0 models
        results = search("2.0-flash")
        assert len(results) > 0
        deprecated = [r for r in results if r.get("deprecated")]
        assert len(deprecated) > 0

    def test_searches_notes_field(self) -> None:
        # "Long-ctx" appears in notes of several gpt-5 models
        results = search("Long-ctx")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Data integrity checks
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_all_prices_have_provider(self) -> None:
        for key, data in PRICES.items():
            assert "provider" in data, f"{key} missing 'provider'"

    def test_all_prices_have_input_or_none(self) -> None:
        for key, data in PRICES.items():
            assert "input" in data, f"{key} missing 'input' key"

    def test_all_aliases_resolve_to_existing_keys(self) -> None:
        for alias, target in ALIASES.items():
            assert target in PRICES, f"Alias {alias!r} -> {target!r} not in PRICES"

    def test_deprecated_field_is_bool_when_present(self) -> None:
        for key, data in PRICES.items():
            if "deprecated" in data:
                assert data["deprecated"] is True, f"{key}: deprecated should be True"

    def test_known_deprecated_models_flagged(self) -> None:
        for m in ["gpt-4-32k", "gpt-3.5-turbo", "gemini-2.0-flash", "claude-opus-4.1"]:
            assert PRICES[m].get("deprecated") is True, f"{m} should be deprecated"

    def test_known_active_models_not_flagged(self) -> None:
        for m in ["gpt-4o", "claude-sonnet-5", "gemini-2.5-flash", "grok-4.5"]:
            assert not PRICES[m].get("deprecated"), f"{m} should not be deprecated"
