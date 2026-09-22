"""Tests for tdvutil.llmcost (LLM model pricing utilities)."""

import pytest

from _tdvutil.llmcost import (
    _ALIASES,
    _PRICES,
    DeprecatedModelError,
    LLM_Cost,
    LLM_Model,
    LLM_Provider,
    get_model,
    get_models,
    get_providers,
    search_models,
)


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_exact_match(self) -> None:
        m = get_model("gpt-4o")
        assert m.name == "gpt-4o"

    def test_alias_match(self) -> None:
        m = get_model("gpt4o")
        assert m.name == "gpt-4o"

    def test_alias_case_insensitive(self) -> None:
        m = get_model("GPT4O")
        assert m.name == "gpt-4o"

    def test_returns_llm_model(self) -> None:
        m = get_model("gpt-4o")
        assert isinstance(m, LLM_Model)

    def test_provider_is_llm_provider(self) -> None:
        m = get_model("gpt-4o")
        assert isinstance(m.provider, LLM_Provider)
        assert m.provider.id == "openai"
        assert m.provider.name == "OpenAI"
        assert "openai" in m.provider.url.lower()

    def test_pricing_fields(self) -> None:
        m = get_model("gpt-4o")
        assert m.input == 2.50
        assert m.output == 10.00
        assert m.cached_input == 1.25

    @pytest.mark.parametrize(
        ("name", "input_price", "cached_price", "write_price", "output_price"),
        [
            ("gpt-5.6-sol", 4.00, 0.40, 5.00, 20.00),
            ("gpt-6-astra", 10.00, 1.00, 12.50, 50.00),
            ("copilot/gpt-5.6-sol", 4.00, 0.40, 5.00, 20.00),
            ("copilot/gpt-6-astra", 10.00, 1.00, 12.50, 50.00),
        ],
    )
    def test_updated_gpt_5_6_prices(
        self,
        name: str,
        input_price: float,
        cached_price: float,
        write_price: float,
        output_price: float,
    ) -> None:
        m = get_model(name)
        assert m.input == input_price
        assert m.cached_input == cached_price
        assert m.cache_write == write_price
        assert m.output == output_price

    @pytest.mark.parametrize(
        ("name", "input_price", "cached_price", "output_price"),
        [
            ("gemini-3.6-flash", 0.75, 0.075, 3.75),
            ("gemini-3.7-flash", 0.75, 0.075, 3.75),
            ("gemini-3.8-flash", 0.75, 0.075, 3.75),
            ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
            ("claude-mythos-5", 10.00, 1.00, 50.00),
            ("claude-fable-5.1", 10.00, 0.25, 50.00),
            ("claude-opus-5.5", 4.00, 0.20, 20.00),
            ("gpt-6-sol", 2.00, 0.20, 10.00),
            ("gpt-6-luna", 0.10, 0.01, 0.50),
            ("grok-4.6", 2.00, 0.50, 6.00),
            ("grok-4.7", 2.00, 0.50, 6.00),
            ("zai-glm-5-3", 1.40, 0.14, 4.40),
            ("together/ternary-bonsai-27b", 0.00, None, 0.00),
            ("groq/openai/gpt-oss-20b", 0.075, 0.0375, 0.30),
        ],
    )
    def test_new_model_prices(
        self,
        name: str,
        input_price: float,
        cached_price: float,
        output_price: float,
    ) -> None:
        m = get_model(name)
        assert m.input == input_price
        assert m.cached_input == cached_price
        assert m.output == output_price

    def test_no_fuzzy_match(self) -> None:
        with pytest.raises(KeyError):
            get_model("gpt-4")

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_model("zzz-no-such-model-zzz")

    def test_fuzzy_hint_in_key_error(self) -> None:
        with pytest.raises(KeyError, match="Did you mean"):
            get_model("gpt-4o-mini-preview")

    def test_deprecated_raises_by_default(self) -> None:
        with pytest.raises(DeprecatedModelError, match="deprecated"):
            get_model("gpt-4-32k")

    def test_deprecated_allowed_with_flag(self) -> None:
        m = get_model("gpt-4-32k", include_deprecated=True)
        assert m.name == "gpt-4-32k"
        assert m.deprecated is True

    def test_none_fields_are_none(self) -> None:
        # gpt-5.5-pro has no cached_input
        m = get_model("gpt-5.5-pro")
        assert m.cached_input is None

    def test_known_alias(self) -> None:
        assert get_model("claude-sonnet").name == "claude-sonnet-5"

    @pytest.mark.parametrize("alias", ["gpt6", "chatgpt-6", "astra"])
    def test_gpt_6_aliases(self, alias: str) -> None:
        assert get_model(alias).name == "gpt-6-astra"

    def test_notes_field(self) -> None:
        m = get_model("claude-sonnet-5")
        assert m.notes is not None
        assert "pricing" in m.notes.lower() or "intro" in m.notes.lower()


# ---------------------------------------------------------------------------
# LLM_Model.cost()
# ---------------------------------------------------------------------------

class TestModelCost:
    def test_basic_cost(self) -> None:
        m = get_model("gpt-4o")
        cost = m.cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert isinstance(cost, LLM_Cost)
        assert cost.currency == "USD"
        assert cost.input_cost == pytest.approx(2.50, rel=1e-6)
        assert cost.output_cost == pytest.approx(10.00, rel=1e-6)
        assert cost.total_cost == pytest.approx(12.50, rel=1e-6)

    def test_model_name_in_result(self) -> None:
        m = get_model("gpt-4o")
        cost = m.cost(input_tokens=100, output_tokens=100)
        assert cost.model == "gpt-4o"

    def test_zero_tokens(self) -> None:
        m = get_model("gpt-4o")
        cost = m.cost(input_tokens=0, output_tokens=0)
        assert cost.total_cost == 0.0

    def test_cache_hit_tokens(self) -> None:
        m = get_model("gpt-4o")  # cached_input = $1.25/1M
        cost = m.cost(input_tokens=0, output_tokens=0, cache_hit_tokens=1_000_000)
        assert cost.cached_cost == pytest.approx(1.25, rel=1e-6)

    def test_deprecated_cached_input_tokens_alias(self) -> None:
        m = get_model("gpt-4o")
        with pytest.warns(DeprecationWarning, match="cache_hit_tokens"):
            cost = m.cost(
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=1_000_000,
            )
        assert cost.cached_cost == pytest.approx(1.25, rel=1e-6)

    def test_cache_token_arguments_are_mutually_exclusive(self) -> None:
        m = get_model("gpt-4o")
        with pytest.raises(TypeError, match="only one"):
            m.cost(
                input_tokens=0,
                output_tokens=0,
                cache_hit_tokens=1,
                cached_input_tokens=1,
            )

    def test_cache_write_tokens_anthropic(self) -> None:
        m = get_model("claude-sonnet-5")  # cache_write = $2.50/1M
        cost = m.cost(input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000)
        assert cost.cache_write_cost == pytest.approx(2.50, rel=1e-6)

    def test_batch_pricing(self) -> None:
        m = get_model("claude-sonnet-5")
        standard = m.cost(input_tokens=1_000_000, output_tokens=1_000_000)
        batch = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, use_batch=True)
        assert batch.total_cost < standard.total_cost

    def test_current_google_batch_pricing(self) -> None:
        m = get_model("gemini-2.5-flash")
        batch = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, use_batch=True)
        assert batch.pricing_tier == "batch"
        assert batch.input_rate == pytest.approx(0.15)
        assert batch.output_rate == pytest.approx(1.25)
        assert batch.total_cost == pytest.approx(1.40)

    def test_breakdown_string(self) -> None:
        m = get_model("gpt-4o")
        cost = m.cost(input_tokens=100_000, output_tokens=20_000)
        assert "input" in cost.breakdown
        assert "output" in cost.breakdown
        assert "Total" in cost.breakdown

    def test_long_context_pricing(self) -> None:
        m = get_model("gpt-5.5")  # long_input=10.00, long_output=45.00
        standard = m.cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            long_context=False,
        )
        long = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, long_context=True)
        assert long.total_cost > standard.total_cost
        assert long.input_cost == pytest.approx(10.00, rel=1e-6)
        assert long.output_cost == pytest.approx(45.00, rel=1e-6)

    def test_long_context_no_pricing_falls_back(self) -> None:
        # gpt-4o has no long_input/long_output — should silently use standard rates
        m = get_model("gpt-4o")
        standard = m.cost(input_tokens=1_000_000, output_tokens=1_000_000)
        long = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, long_context=True)
        assert long.total_cost == standard.total_cost

    def test_long_context_uses_long_cache_prices(self) -> None:
        m = get_model("gpt-5.6-terra")
        cost = m.cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_hit_tokens=1_000_000,
            cache_write_tokens=1_000_000,
            long_context=True,
        )
        assert cost.input_cost == pytest.approx(4.00, rel=1e-6)
        assert cost.cached_cost == pytest.approx(0.40, rel=1e-6)
        assert cost.cache_write_cost == pytest.approx(5.00, rel=1e-6)
        assert cost.output_cost == pytest.approx(18.00, rel=1e-6)
        assert cost.total_cost == pytest.approx(27.40, rel=1e-6)

    def test_long_context_is_selected_automatically(self) -> None:
        m = get_model("gpt-5.6-terra")
        cost = m.cost(input_tokens=272_001, output_tokens=1)
        assert cost.pricing_tier == "long_context"
        assert cost.input_rate == 4.00
        assert cost.cache_hit_rate == 0.40
        assert cost.cache_write_rate == 5.00
        assert cost.output_rate == 18.00

    def test_standard_context_can_be_forced(self) -> None:
        m = get_model("gpt-5.6-terra")
        cost = m.cost(input_tokens=272_001, output_tokens=1, long_context=False)
        assert cost.pricing_tier == "standard"
        assert cost.input_rate == 2.00

    def test_all_prompt_token_types_count_toward_threshold(self) -> None:
        m = get_model("gpt-5.6-terra")
        cost = m.cost(
            input_tokens=200_000,
            output_tokens=1,
            cache_hit_tokens=50_000,
            cache_write_tokens=22_001,
        )
        assert cost.pricing_tier == "long_context"

    def test_gpt_5_4_long_context_pricing(self) -> None:
        m = get_model("gpt-5.4")
        assert m.long_context_threshold == 272_000
        cost = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, long_context=True)
        assert cost.total_cost == pytest.approx(27.50, rel=1e-6)

    def test_gpt_6_astra_long_context_and_batch_pricing(self) -> None:
        m = get_model("gpt-6-astra")
        long = m.cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_hit_tokens=1_000_000,
            cache_write_tokens=1_000_000,
            long_context=True,
        )
        batch = m.cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            use_batch=True,
        )
        assert long.total_cost == pytest.approx(122.00, rel=1e-6)
        assert batch.total_cost == pytest.approx(30.00, rel=1e-6)

    def test_xai_long_context_pricing(self) -> None:
        m = get_model("grok-4.3")
        long = m.cost(input_tokens=1_000_000, output_tokens=1_000_000, long_context=True)
        assert long.total_cost == pytest.approx(7.50, rel=1e-6)
        # xAI's current pricing page does not publish Batch API rates.
        batch_requested = m.cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            use_batch=True,
            long_context=False,
        )
        assert batch_requested.pricing_tier == "standard"
        assert batch_requested.total_cost == pytest.approx(3.75, rel=1e-6)

    def test_small_count(self) -> None:
        m = get_model("gpt-4o")
        cost = m.cost(input_tokens=100, output_tokens=50)
        # 100/1M * 2.50 + 50/1M * 10.0 = 0.00025 + 0.0005 = 0.00075
        assert cost.total_cost == pytest.approx(0.00075, rel=1e-4)


# ---------------------------------------------------------------------------
# get_models
# ---------------------------------------------------------------------------

class TestGetModels:
    def test_returns_list_of_llm_model(self) -> None:
        models = get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, LLM_Model) for m in models)

    def test_excludes_deprecated_by_default(self) -> None:
        models = get_models()
        assert all(not m.deprecated for m in models)

    def test_includes_deprecated_with_flag(self) -> None:
        all_models = get_models(include_deprecated=True)
        active = get_models(include_deprecated=False)
        assert len(all_models) > len(active)

    def test_provider_filter(self) -> None:
        anthropic = get_models("anthropic")
        assert all(m.provider.id == "anthropic" for m in anthropic)
        assert len(anthropic) > 0

    def test_provider_filter_case_insensitive(self) -> None:
        upper = get_models("ANTHROPIC")
        lower = get_models("anthropic")
        assert len(upper) == len(lower)

    def test_unknown_provider_returns_empty(self) -> None:
        assert get_models("no-such-provider") == []

    def test_provider_on_each_model(self) -> None:
        for m in get_models():
            assert isinstance(m.provider, LLM_Provider)


# ---------------------------------------------------------------------------
# get_providers
# ---------------------------------------------------------------------------

class TestGetProviders:
    def test_returns_list(self) -> None:
        providers = get_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_all_are_llm_provider(self) -> None:
        for p in get_providers():
            assert isinstance(p, LLM_Provider)

    def test_known_providers_present(self) -> None:
        ids = {p.id for p in get_providers()}
        for expected in ["openai", "anthropic", "google", "mistral"]:
            assert expected in ids

    def test_each_has_name_and_url(self) -> None:
        for p in get_providers():
            assert p.name
            assert p.url

    def test_openai_url(self) -> None:
        by_id = {p.id: p for p in get_providers()}
        assert "openai.com" in by_id["openai"].url


# ---------------------------------------------------------------------------
# search_models
# ---------------------------------------------------------------------------

class TestSearchModels:
    def test_finds_flash_models(self) -> None:
        results = search_models("flash")
        assert len(results) > 0
        assert all(isinstance(r, LLM_Model) for r in results)
        assert all("flash" in r.name.lower() or (r.notes and "flash" in r.notes.lower())
                   for r in results)

    def test_case_insensitive(self) -> None:
        lower = search_models("flash")
        upper = search_models("FLASH")
        assert len(lower) == len(upper)

    def test_no_results(self) -> None:
        assert search_models("zzznomatchzzz") == []

    def test_includes_deprecated(self) -> None:
        results = search_models("2.0-flash")
        assert len(results) > 0
        deprecated = [r for r in results if r.deprecated]
        assert len(deprecated) > 0

    def test_searches_notes(self) -> None:
        results = search_models("Long-ctx")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_all_prices_have_provider(self) -> None:
        for key, data in _PRICES.items():
            assert "provider" in data, f"{key} missing 'provider'"

    def test_all_aliases_resolve(self) -> None:
        for alias, target in _ALIASES.items():
            assert target in _PRICES, f"Alias {alias!r} -> {target!r} not in _PRICES"

    def test_deprecated_field_is_bool(self) -> None:
        for key, data in _PRICES.items():
            if "deprecated" in data:
                assert data["deprecated"] is True, f"{key}: deprecated should be True"

    def test_known_deprecated_flagged(self) -> None:
        for name in ["gpt-4-32k", "gpt-3.5-turbo", "gemini-2.0-flash", "claude-opus-4.1"]:
            m = get_model(name, include_deprecated=True)
            assert m.deprecated, f"{name} should be deprecated"

    def test_known_active_not_flagged(self) -> None:
        for name in ["gpt-4o", "claude-sonnet-5", "gemini-2.5-flash", "grok-4.5"]:
            m = get_model(name)
            assert not m.deprecated, f"{name} should not be deprecated"

    def test_get_models_providers_match_get_providers(self) -> None:
        provider_ids = {p.id for p in get_providers()}
        for m in get_models():
            assert m.provider.id in provider_ids, \
                f"{m.name} has unknown provider {m.provider.id!r}"
