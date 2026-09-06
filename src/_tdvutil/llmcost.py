"""
LLM model pricing database and cost-calculation utilities.

Prices last updated: 2026-09-06
All token prices are USD per 1,000,000 tokens unless noted otherwise.

Usage::

    from tdvutil.llmcost import get_model, LLM_Model, DeprecatedModelError

    model = get_model("gpt-4o")
    print(f"${model.input} in / ${model.output} out per 1M tokens")
    cost = model.cost(input_tokens=10_000, output_tokens=2_000)
    print(f"Total: ${cost['total_cost']:.4f}")

Sources:
    OpenAI:    https://developers.openai.com/api/docs/pricing
    Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
    Google:    https://ai.google.dev/gemini-api/docs/pricing
    Mistral:   https://mistral.ai/pricing/api
    xAI:       https://docs.x.ai/docs/pricing
    Together:  https://www.together.ai/pricing
    Groq:      https://groq.com/pricing/
    Copilot:   https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
"""

from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

# ---------------------------------------------------------------------------
# Price database
# Each entry: dict with at minimum:
#   provider : str
#   input    : float | None   — USD per 1M input tokens
#   output   : float | None   — USD per 1M output tokens
# Optional keys (all USD per 1M tokens unless noted):
#   cached_input             — price for cache-hit input tokens
#   cache_write              — price to write tokens into cache
#   batch_input / batch_output
#   long_input / long_output — price above the long-context threshold
#   long_cached_input / long_cache_write — long-context cache prices
#   longctx_threshold        — token count where long-context pricing kicks in
#   deprecated : bool        — model is no longer active/recommended
#   notes : str
# ---------------------------------------------------------------------------

# Public reference (read-only; do not modify)
_PRICES: Dict[str, Dict[str, object]] = {

    # =========================================================================
    # OPENAI
    # =========================================================================

    # --- Flagship / latest ---
    "gpt-6-astra":      {"provider": "openai", "input": 10.00,  "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00,
                          "batch_input": 5.00, "batch_output": 25.00,
                          "long_input": 20.00, "long_cached_input": 2.00, "long_cache_write": 25.00, "long_output": 75.00, "longctx_threshold": 272_000},
    "gpt-5.6-sol":      {"provider": "openai", "input": 4.00,   "cached_input": 0.40,  "cache_write": 5.00, "output": 20.00,
                          "long_input": 8.00,  "long_cached_input": 0.80, "long_cache_write": 10.00, "long_output": 30.00, "longctx_threshold": 272_000,
                          "notes": "Promotional pricing through at least 2026-11-21"},
    "gpt-5.6-terra":    {"provider": "openai", "input": 2.00,   "cached_input": 0.20,  "cache_write": 2.50, "output": 12.00,
                          "long_input": 4.00,  "long_cached_input": 0.40, "long_cache_write": 5.00,  "long_output": 18.00, "longctx_threshold": 272_000},
    "gpt-5.6-luna":     {"provider": "openai", "input": 0.20,   "cached_input": 0.02,  "cache_write": 0.25, "output": 1.20,
                          "long_input": 0.40,  "long_cached_input": 0.04, "long_cache_write": 0.50,  "long_output": 1.80,  "longctx_threshold": 272_000},
    "gpt-5.5":          {"provider": "openai", "input": 5.00,   "cached_input": 0.50,  "output": 30.00,
                          "long_input": 10.00, "long_cached_input": 1.00, "long_output": 45.00, "longctx_threshold": 272_000},
    "gpt-5.5-pro":      {"provider": "openai", "input": 30.00,  "cached_input": None,  "output": 180.00,
                          "long_input": 60.00, "long_output": 270.00, "longctx_threshold": 272_000},
    "gpt-5.4":          {"provider": "openai", "input": 2.50,   "cached_input": 0.25,  "output": 15.00,
                          "long_input": 5.00, "long_cached_input": 0.50, "long_output": 22.50, "longctx_threshold": 272_000},
    "gpt-5.4-mini":     {"provider": "openai", "input": 0.75,   "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano":     {"provider": "openai", "input": 0.20,   "cached_input": 0.02,  "output": 1.25},
    "gpt-5.4-pro":      {"provider": "openai", "input": 30.00,  "cached_input": None,  "output": 180.00,
                          "long_input": 60.00, "long_output": 270.00, "longctx_threshold": 272_000},
    "gpt-5.2":          {"provider": "openai", "input": 1.75,   "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-pro":      {"provider": "openai", "input": 21.00,  "cached_input": None,  "output": 168.00},
    "gpt-5.1":          {"provider": "openai", "input": 1.25,   "cached_input": 0.125, "output": 10.00},
    "gpt-5":            {"provider": "openai", "input": 1.25,   "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini":       {"provider": "openai", "input": 0.25,   "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano":       {"provider": "openai", "input": 0.05,   "cached_input": 0.005, "output": 0.40},
    "gpt-5-pro":        {"provider": "openai", "input": 15.00,  "cached_input": None,  "output": 120.00},

    # --- GPT-4.x ---
    "gpt-4.1":          {"provider": "openai", "input": 2.00,   "cached_input": 0.50,  "output": 8.00},
    "gpt-4.1-mini":     {"provider": "openai", "input": 0.40,   "cached_input": 0.10,  "output": 1.60},
    "gpt-4.1-nano":     {"provider": "openai", "input": 0.10,   "cached_input": 0.025, "output": 0.40},
    "gpt-4o":           {"provider": "openai", "input": 2.50,   "cached_input": 1.25,  "output": 10.00},
    "gpt-4o-mini":      {"provider": "openai", "input": 0.15,   "cached_input": 0.075, "output": 0.60},

    # --- Reasoning (o-series) ---
    "o4-mini":          {"provider": "openai", "input": 1.10,   "cached_input": 0.275, "output": 4.40},
    "o3":               {"provider": "openai", "input": 2.00,   "cached_input": 0.50,  "output": 8.00},
    "o3-mini":          {"provider": "openai", "input": 1.10,   "cached_input": 0.55,  "output": 4.40,  "deprecated": True},
    "o3-pro":           {"provider": "openai", "input": 20.00,  "cached_input": None,  "output": 80.00},
    "o1":               {"provider": "openai", "input": 15.00,  "cached_input": 7.50,  "output": 60.00},
    "o1-mini":          {"provider": "openai", "input": 1.10,   "cached_input": 0.55,  "output": 4.40,  "deprecated": True},
    "o1-pro":           {"provider": "openai", "input": 150.00, "cached_input": None,  "output": 600.00},

    # --- Legacy GPT-4 (all deprecated) ---
    "gpt-4o-2024-05-13":         {"provider": "openai", "input": 5.00,  "output": 15.00, "deprecated": True},
    "gpt-4-turbo-2024-04-09":    {"provider": "openai", "input": 10.00, "output": 30.00,  "deprecated": True},
    "gpt-4-0125-preview":        {"provider": "openai", "input": 10.00, "output": 30.00,  "deprecated": True},
    "gpt-4-1106-preview":        {"provider": "openai", "input": 10.00, "output": 30.00,  "deprecated": True},
    "gpt-4-0613":                {"provider": "openai", "input": 30.00, "output": 60.00,  "deprecated": True},
    "gpt-4-32k":                 {"provider": "openai", "input": 60.00, "output": 120.00, "deprecated": True},

    # --- GPT-3.5 (all deprecated) ---
    "gpt-3.5-turbo":             {"provider": "openai", "input": 0.50, "output": 1.50, "deprecated": True},
    "gpt-3.5-turbo-0125":        {"provider": "openai", "input": 0.50, "output": 1.50, "deprecated": True},
    "gpt-3.5-turbo-instruct":    {"provider": "openai", "input": 1.50, "output": 2.00, "deprecated": True},
    "gpt-3.5-turbo-16k-0613":    {"provider": "openai", "input": 3.00, "output": 4.00, "deprecated": True},

    # --- Specialized ---
    "o3-deep-research":          {"provider": "openai", "input": 10.00, "cached_input": 2.50,  "output": 40.00},
    "o4-mini-deep-research":     {"provider": "openai", "input": 2.00,  "cached_input": 0.50,  "output": 8.00},
    "gpt-4o-search-preview":     {"provider": "openai", "input": 2.50,  "output": 10.00},
    "gpt-4o-mini-search-preview":{"provider": "openai", "input": 0.15,  "output": 0.60},
    "computer-use-preview":      {"provider": "openai", "input": 3.00,  "output": 12.00},
    "codex-mini-latest":         {"provider": "openai", "input": 1.50,  "cached_input": 0.375, "output": 6.00},

    # --- Embeddings ---
    "text-embedding-3-small":    {"provider": "openai", "input": 0.02,  "output": None, "notes": "Embedding only"},
    "text-embedding-3-large":    {"provider": "openai", "input": 0.13,  "output": None, "notes": "Embedding only"},
    "text-embedding-ada-002":    {"provider": "openai", "input": 0.10,  "output": None, "notes": "Embedding only"},

    # =========================================================================
    # ANTHROPIC
    # =========================================================================

    "claude-fable-5.1":          {"provider": "anthropic", "input": 10.00, "output": 50.00,
                                  "cached_input": 0.25,   "cache_write": 12.50,
                                  "batch_input": 5.00,    "batch_output": 25.00,
                                  "notes": "Cache hits priced at 0.025x base input"},
    "claude-mythos-5.1":         {"provider": "anthropic", "input": 10.00, "output": 50.00,
                                  "cached_input": 0.25,   "cache_write": 12.50,
                                  "batch_input": 5.00,    "batch_output": 25.00,
                                  "notes": "Limited availability; cache hits priced at 0.025x base input"},
    "claude-fable-5":            {"provider": "anthropic", "input": 10.00, "output": 50.00,
                                  "cached_input": 1.00,   "cache_write": 12.50,
                                  "batch_input": 5.00,    "batch_output": 25.00,
                                  "notes": "Next-gen intelligence for long-running agents"},
    "claude-mythos-5":           {"provider": "anthropic", "input": 10.00, "output": 50.00,
                                  "cached_input": 1.00,   "cache_write": 12.50,
                                  "batch_input": 5.00,    "batch_output": 25.00,
                                  "notes": "Limited availability"},
    "claude-opus-5":             {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50},
    "claude-opus-4.8":           {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50,
                                  "notes": "Fast mode available at 2x pricing"},
    "claude-sonnet-5":           {"provider": "anthropic", "input": 2.00,  "output": 10.00,
                                  "cached_input": 0.20,   "cache_write": 2.50,
                                  "batch_input": 1.00,    "batch_output": 5.00,
                                  "notes": "Launch pricing retained as standard pricing"},
    "claude-haiku-4.5":          {"provider": "anthropic", "input": 1.00,  "output": 5.00,
                                  "cached_input": 0.10,   "cache_write": 1.25,
                                  "batch_input": 0.50,    "batch_output": 2.50},
    "claude-sonnet-4.6":         {"provider": "anthropic", "input": 3.00,  "output": 15.00,
                                  "cached_input": 0.30,   "cache_write": 3.75,
                                  "batch_input": 1.50,    "batch_output": 7.50},
    "claude-opus-4.7":           {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50},
    "claude-opus-4.6":           {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50},
    "claude-sonnet-4.5":         {"provider": "anthropic", "input": 3.00,  "output": 15.00,
                                  "cached_input": 0.30,   "cache_write": 3.75,
                                  "batch_input": 1.50,    "batch_output": 7.50},
    "claude-opus-4.5":           {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50},
    "claude-opus-4.1":           {"provider": "anthropic", "input": 15.00, "output": 75.00,
                                  "cached_input": 1.50,   "cache_write": 18.75,
                                  "batch_input": 7.50,    "batch_output": 37.50, "deprecated": True},

    # =========================================================================
    # GOOGLE GEMINI  (Gemini Developer API / Google AI Studio paid tier)
    # =========================================================================

    "gemini-2.5-pro":        {"provider": "google", "input": 1.25,  "output": 10.00,
                              "cached_input": 0.125,
                              "long_input": 2.50, "long_output": 15.00, "longctx_threshold": 200_000,
                              "notes": "Tiered: >200k ctx; cache storage $4.50/1M tok/hr"},
    "gemini-2.5-flash":      {"provider": "google", "input": 0.30,  "output": 2.50,
                              "cached_input": 0.03,
                              "notes": "Audio input: $1.00/1M; 1M ctx window"},
    "gemini-2.5-flash-lite": {"provider": "google", "input": 0.10,  "output": 0.40,
                              "cached_input": 0.01,
                              "notes": "Audio input: $0.30/1M"},
    "gemini-3.8-flash":      {"provider": "google", "input": 0.75,  "output": 3.75,
                              "cached_input": 0.075,
                              "notes": "GA; promotional pricing through 2026-12-31; thinking tokens billed in output"},
    "gemini-3.7-flash":      {"provider": "google", "input": 0.75,  "output": 3.75,
                              "cached_input": 0.075,
                              "notes": "GA; promotional pricing through 2026-12-31; thinking tokens billed in output"},
    "gemini-3.6-flash":      {"provider": "google", "input": 0.75,  "output": 3.75,
                              "cached_input": 0.075,
                              "notes": "GA; promotional pricing through 2026-12-31; thinking tokens billed in output"},
    "gemini-3.5-flash":      {"provider": "google", "input": 1.50,  "output": 9.00,
                              "cached_input": 0.15,
                              "notes": "GA; thinking tokens billed in output"},
    "gemini-3.5-flash-lite": {"provider": "google", "input": 0.30,  "output": 2.50,
                              "cached_input": 0.03,
                              "notes": "GA; thinking tokens billed in output"},
    "gemini-3.1-pro-preview":{"provider": "google", "input": 2.00,  "output": 12.00,
                              "cached_input": 0.20,
                              "long_input": 4.00, "long_output": 18.00, "longctx_threshold": 200_000,
                              "notes": "Preview; >200k ctx; cache storage $4.50/1M tok/hr"},
    "gemini-3.1-flash-lite": {"provider": "google", "input": 0.25,  "output": 1.50,
                              "cached_input": 0.025,
                              "notes": "GA; audio input: $0.50/1M"},
    "gemini-3-flash-preview":{"provider": "google", "input": 0.50,  "output": 3.00,
                              "cached_input": 0.05,
                              "notes": "Preview; audio input: $1.00/1M"},
    "gemini-2.0-flash":      {"provider": "google", "input": 0.10,  "output": 0.40,  "deprecated": True},
    "gemini-2.0-flash-lite": {"provider": "google", "input": 0.075, "output": 0.30,  "deprecated": True},
    "gemini-embedding-2":    {"provider": "google", "input": 0.20,  "output": None,
                              "notes": "Multimodal embeddings; image $0.45/1M, audio $6.50/1M"},
    "gemini-embedding-001":  {"provider": "google", "input": 0.15,  "output": None,
                              "notes": "Text embeddings only"},

    # =========================================================================
    # MISTRAL
    # =========================================================================

    "mistral-medium-latest":  {"provider": "mistral", "input": 1.50, "output": 7.50,
                               "notes": "Mistral Medium 3.5; batch -50%, cache 10% of input"},
    "mistral-large-latest":   {"provider": "mistral", "input": 0.50, "output": 1.50,
                               "notes": "Mistral Large 3"},
    "mistral-small-latest":   {"provider": "mistral", "input": 0.15, "output": 0.60,
                               "notes": "Mistral Small 4"},
    "devstral-medium-latest": {"provider": "mistral", "input": 0.40, "output": 2.00,
                               "notes": "Devstral 2; coding/agentic"},
    "devstral-small-latest":  {"provider": "mistral", "input": 0.10, "output": 0.30,
                               "notes": "Devstral Small 2"},
    "codestral-latest":       {"provider": "mistral", "input": 0.30, "output": 0.90,
                               "notes": "Codestral; coding"},
    "magistral-medium-latest":{"provider": "mistral", "input": 2.00, "output": 5.00,
                               "notes": "Magistral Medium; reasoning"},
    "magistral-small-latest": {"provider": "mistral", "input": 0.50, "output": 1.50,
                               "notes": "Magistral Small; reasoning"},
    "ministral-3b-latest":    {"provider": "mistral", "input": 0.10, "output": 0.10},
    "ministral-8b-latest":    {"provider": "mistral", "input": 0.15, "output": 0.15},
    "ministral-14b-latest":   {"provider": "mistral", "input": 0.20, "output": 0.20},
    "open-mistral-nemo":      {"provider": "mistral", "input": 0.15, "output": 0.15},
    "open-mixtral-8x7b":      {"provider": "mistral", "input": 0.70, "output": 0.70},
    "open-mixtral-8x22b":     {"provider": "mistral", "input": 2.00, "output": 6.00},
    "mistral-embed":          {"provider": "mistral", "input": 0.10, "output": None,
                               "notes": "Embedding only"},

    # =========================================================================
    # xAI (Grok)
    # =========================================================================

    "grok-4.5":                    {"provider": "xai", "input": 2.00,  "cached_input": 0.30,  "output": 6.00,
                                    "long_input": 4.00, "long_cached_input": 0.60, "long_output": 12.00, "longctx_threshold": 200_000,
                                    "notes": "500k context"},
    "grok-4.6":                    {"provider": "xai", "input": 2.00,  "cached_input": 0.50,  "output": 6.00,
                                    "long_input": 4.00, "long_cached_input": 1.00, "long_output": 12.00, "longctx_threshold": 200_000,
                                    "notes": "500k context"},
    "grok-4.3":                    {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "batch_input": 1.00, "batch_output": 2.00,
                                    "long_input": 2.50, "long_cached_input": 0.40, "long_output": 5.00, "longctx_threshold": 200_000,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-multi-agent-0309":  {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "batch_input": 1.00, "batch_output": 2.00,
                                    "long_input": 2.50, "long_cached_input": 0.40, "long_output": 5.00, "longctx_threshold": 200_000,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-0309-reasoning":    {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "batch_input": 1.00, "batch_output": 2.00,
                                    "long_input": 2.50, "long_cached_input": 0.40, "long_output": 5.00, "longctx_threshold": 200_000,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-0309-non-reasoning":{"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "batch_input": 1.00, "batch_output": 2.00,
                                    "long_input": 2.50, "long_cached_input": 0.40, "long_output": 5.00, "longctx_threshold": 200_000,
                                    "notes": "1M context; batch -20%"},
    "grok-build-0.1":              {"provider": "xai", "input": 1.00,  "cached_input": 0.20,  "output": 2.00,
                                    "long_input": 2.00, "long_cached_input": 0.40, "long_output": 4.00, "longctx_threshold": 200_000,
                                    "notes": "Code API; 256k context"},

    # =========================================================================
    # TOGETHER AI  (hosted open-source models)
    # =========================================================================

    "together/llama-3.3-70b":          {"provider": "together", "input": 1.04,  "output": 1.04},
    "together/llama-3-8b-instruct-lite":{"provider": "together", "input": 0.14,  "output": 0.14},
    "together/qwen3-235b-a22b":        {"provider": "together", "input": 0.20,  "output": 0.60},
    "together/qwen3.7-plus":           {"provider": "together", "input": 0.32,  "output": 1.28},
    "together/qwen3.7-max":            {"provider": "together", "input": 1.25,  "cached_input": 0.13, "output": 3.75},
    "together/qwen3.8-max":            {"provider": "together", "input": 2.00,  "cached_input": 0.25, "output": 6.00},
    "together/qwen3.8-flash":          {"provider": "together", "input": 0.15,  "output": 0.47},
    "together/qwen3.5-397b":           {"provider": "together", "input": 0.60,  "cached_input": 0.35, "output": 3.60},
    "together/qwen3.5-9b":             {"provider": "together", "input": 0.17,  "output": 0.25},
    "together/qwen2.5-7b-turbo":       {"provider": "together", "input": 0.30,  "output": 0.30},
    "together/gemma-4-31b":            {"provider": "together", "input": 0.39,  "output": 0.97},
    "together/deepseek-v4-pro":        {"provider": "together", "input": 1.32,  "cached_input": 0.13, "output": 3.96},
    "together/deepseek-v4-flash":      {"provider": "together", "input": 0.14,  "cached_input": 0.03, "output": 0.28},
    "together/kimi-k2.7-code":         {"provider": "together", "input": 0.95,  "cached_input": 0.19, "output": 4.00},
    "together/kimi-k2.6":              {"provider": "together", "input": 1.20,  "cached_input": 0.20, "output": 4.50},
    "together/kimi-k3":                {"provider": "together", "input": 3.00,  "cached_input": 0.30, "output": 15.00},
    "together/minimax-m3":             {"provider": "together", "input": 0.30,  "cached_input": 0.06, "output": 1.20},
    "together/gpt-oss-120b":           {"provider": "together", "input": 0.15,  "output": 0.60},
    "together/gpt-oss-20b":            {"provider": "together", "input": 0.05,  "output": 0.20},
    "together/cogito-v2.1-671b":       {"provider": "together", "input": 1.25,  "output": 1.25},
    "together/nvidia-nemotron-3-ultra":{"provider": "together", "input": 0.60,  "cached_input": 0.20, "output": 3.60},
    "together/glm-5.3":                {"provider": "together", "input": 1.40,  "cached_input": 0.26, "output": 4.40},
    "together/glm-5.3-flash":          {"provider": "together", "input": 0.15,  "cached_input": 0.03, "output": 0.50},

    # =========================================================================
    # GITHUB COPILOT  (1 AI credit = $0.01 USD)
    # =========================================================================

    "copilot/gpt-5-mini":          {"provider": "copilot", "input": 0.25,  "cached_input": 0.025, "output": 2.00},
    "copilot/gpt-5.3-codex":       {"provider": "copilot", "input": 1.75,  "cached_input": 0.175, "output": 14.00},
    "copilot/gpt-5.4":             {"provider": "copilot", "input": 2.50,  "cached_input": 0.25,  "output": 15.00,
                                    "long_input": 5.00, "long_output": 22.50, "longctx_threshold": 272_000,
                                    "notes": "Long-ctx >272k tokens"},
    "copilot/gpt-5.4-mini":        {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 4.50},
    "copilot/gpt-5.4-nano":        {"provider": "copilot", "input": 0.20,  "cached_input": 0.02,  "output": 1.25},
    "copilot/gpt-5.5":             {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "output": 30.00,
                                    "long_input": 10.00, "long_output": 45.00, "longctx_threshold": 272_000,
                                    "notes": "Long-ctx >272k tokens"},
    "copilot/gpt-5.6-luna":        {"provider": "copilot", "input": 0.20,  "cached_input": 0.02,  "cache_write": 0.25, "output": 1.20,
                                    "long_input": 0.40, "long_cached_input": 0.04, "long_cache_write": 0.50, "long_output": 1.80, "longctx_threshold": 200_000,
                                    "notes": "Long-ctx >200k tokens"},
    "copilot/gpt-5.6-sol":         {"provider": "copilot", "input": 4.00,  "cached_input": 0.40,  "cache_write": 5.00, "output": 20.00,
                                    "long_input": 8.00, "long_cached_input": 0.80, "long_cache_write": 10.00, "long_output": 30.00, "longctx_threshold": 272_000,
                                    "notes": "Long-ctx >272k tokens"},
    "copilot/gpt-5.6-terra":       {"provider": "copilot", "input": 2.00,  "cached_input": 0.20,  "cache_write": 2.50, "output": 12.00,
                                    "long_input": 4.00, "long_cached_input": 0.40, "long_cache_write": 5.00, "long_output": 18.00, "longctx_threshold": 272_000,
                                    "notes": "Long-ctx >272k tokens"},
    "copilot/gpt-6-astra":         {"provider": "copilot", "input": 10.00, "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00,
                                    "long_input": 20.00, "long_cached_input": 2.00, "long_cache_write": 25.00, "long_output": 75.00, "longctx_threshold": 272_000,
                                    "notes": "Long-ctx >272k tokens"},
    "copilot/claude-haiku-4.5":    {"provider": "copilot", "input": 1.00,  "cached_input": 0.10,  "cache_write": 1.25,  "output": 5.00},
    "copilot/claude-sonnet-4":     {"provider": "copilot", "input": 3.00,  "cached_input": 0.30,  "cache_write": 3.75,  "output": 15.00},
    "copilot/claude-sonnet-4.5":   {"provider": "copilot", "input": 3.00,  "cached_input": 0.30,  "cache_write": 3.75,  "output": 15.00},
    "copilot/claude-sonnet-4.6":   {"provider": "copilot", "input": 3.00,  "cached_input": 0.30,  "cache_write": 3.75,  "output": 15.00},
    "copilot/claude-sonnet-5":     {"provider": "copilot", "input": 2.00,  "cached_input": 0.20,  "cache_write": 2.50,  "output": 10.00,
                                    "notes": "Intro pricing through 2026-08-31"},
    "copilot/claude-opus-4.5":     {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "cache_write": 6.25,  "output": 25.00},
    "copilot/claude-opus-4.6":     {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "cache_write": 6.25,  "output": 25.00},
    "copilot/claude-opus-4.7":     {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "cache_write": 6.25,  "output": 25.00},
    "copilot/claude-opus-4.8":     {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "cache_write": 6.25,  "output": 25.00},
    "copilot/claude-opus-5":       {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "cache_write": 6.25,  "output": 25.00},
    "copilot/claude-opus-4.8-fast":{"provider": "copilot", "input": 10.00, "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00,
                                    "notes": "Fast mode preview; up to 2.5x faster"},
    "copilot/claude-fable-5":      {"provider": "copilot", "input": 10.00, "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00},
    "copilot/claude-fable-5.1":    {"provider": "copilot", "input": 10.00, "cached_input": 0.25,  "cache_write": 12.50, "output": 50.00},
    "copilot/gemini-2.5-pro":      {"provider": "copilot", "input": 1.25,  "cached_input": 0.125, "output": 10.00},
    "copilot/gemini-3-flash":      {"provider": "copilot", "input": 0.50,  "cached_input": 0.05,  "output": 3.00,
                                    "notes": "Public preview"},
    "copilot/gemini-3.1-pro":      {"provider": "copilot", "input": 2.00,  "cached_input": 0.20,  "output": 12.00,
                                    "long_input": 4.00, "long_output": 18.00, "longctx_threshold": 200_000,
                                    "notes": "Public preview; Long-ctx >200k tokens"},
    "copilot/gemini-3.5-flash":    {"provider": "copilot", "input": 1.50,  "cached_input": 0.15,  "output": 9.00},
    "copilot/gemini-3.6-flash":    {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 3.75,
                                    "notes": "Promotional pricing through 2026-12-31"},
    "copilot/gemini-3.7-flash":    {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 3.75,
                                    "notes": "Promotional pricing through 2026-12-31"},
    "copilot/gemini-3.8-flash":    {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 3.75,
                                    "notes": "Promotional pricing through 2026-12-31"},
    "copilot/raptor-mini":         {"provider": "copilot", "input": 0.25,  "cached_input": 0.025, "output": 2.00,
                                    "notes": "GitHub fine-tuned model"},
    "copilot/mai-code-1-flash":    {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 4.50,
                                    "notes": "Microsoft model"},
    "copilot/mai-code-1.1-flash":  {"provider": "copilot", "input": 0.20,  "cached_input": 0.02,  "output": 1.20,
                                    "notes": "Microsoft model"},
    "copilot/kimi-k2.7-code":      {"provider": "copilot", "input": 0.95,  "cached_input": 0.19,  "output": 4.00,
                                    "notes": "Moonshot AI model"},
    "copilot/kimi-k3":             {"provider": "copilot", "input": 3.00,  "cached_input": 0.30,  "output": 15.00,
                                    "notes": "Moonshot AI model"},
    "copilot/grok-4.5":            {"provider": "copilot", "input": 2.00,  "cached_input": 0.50,  "output": 6.00,
                                    "long_input": 4.00, "long_cached_input": 1.00, "long_output": 12.00, "longctx_threshold": 200_000,
                                    "notes": "Long-ctx >200k tokens"},
    "copilot/grok-4.6":            {"provider": "copilot", "input": 2.00,  "cached_input": 0.50,  "output": 6.00,
                                    "long_input": 4.00, "long_cached_input": 1.00, "long_output": 12.00, "longctx_threshold": 200_000,
                                    "notes": "Long-ctx >200k tokens"},

    # =========================================================================
    # GROQ  (ultra-fast inference)
    # =========================================================================

    "groq/llama-4-scout-17bx16e":    {"provider": "groq", "input": 0.11,  "output": 0.34,
                                      "notes": "128k ctx; ~594 tok/s", "deprecated": True},
    "groq/llama-3.3-70b-versatile":  {"provider": "groq", "input": 0.59,  "output": 0.79,
                                      "notes": "128k ctx; ~394 tok/s"},
    "groq/llama-3.1-8b-instant":     {"provider": "groq", "input": 0.05,  "output": 0.08,
                                      "notes": "128k ctx; ~840 tok/s"},
    "groq/qwen3-32b":                {"provider": "groq", "input": 0.29,  "output": 0.59,
                                      "notes": "131k ctx; ~662 tok/s", "deprecated": True},
    "groq/qwen3.6-27b":              {"provider": "groq", "input": 0.60,  "output": 3.00,
                                      "notes": "131k ctx; ~500 tok/s"},
    "groq/qwen3.8-27b":              {"provider": "groq", "input": 0.80,  "output": 4.00,
                                      "notes": "131k ctx; multimodal"},
    "groq/gpt-oss-20b":              {"provider": "groq", "input": 0.075, "cached_input": 0.037, "output": 0.30,
                                      "notes": "128k ctx; ~1000 tok/s"},
    "groq/gpt-oss-120b":             {"provider": "groq", "input": 0.15,  "cached_input": 0.075, "output": 0.60,
                                      "notes": "128k ctx; ~500 tok/s"},
    "groq/kimi-k2-instruct":         {"provider": "groq", "input": 1.00,  "cached_input": 0.50, "output": 3.00},
}

# ---------------------------------------------------------------------------
# Aliases — common shorthand → canonical key
# ---------------------------------------------------------------------------
_ALIASES: Dict[str, str] = {
    # OpenAI shorthands
    "gpt4o":          "gpt-4o",
    "gpt4omini":      "gpt-4o-mini",
    "gpt4":           "gpt-4-0613",
    "gpt4turbo":      "gpt-4-turbo-2024-04-09",
    "gpt35":          "gpt-3.5-turbo",
    "gpt35turbo":     "gpt-3.5-turbo",
    "o1pro":          "o1-pro",
    "o3pro":          "o3-pro",
    "gpt6":           "gpt-6-astra",
    "chatgpt-6":      "gpt-6-astra",
    "astra":          "gpt-6-astra",

    # Anthropic shorthands
    "claude-fable":        "claude-fable-5.1",
    "fable5":              "claude-fable-5.1",
    "claude-opus":         "claude-opus-5",
    "claude-sonnet":       "claude-sonnet-5",
    "claude-haiku":        "claude-haiku-4.5",
    "claude-3.5-sonnet":   "claude-sonnet-4.5",
    "claude-3.7-sonnet":   "claude-sonnet-4.6",
    "claude-opus-4":       "claude-opus-4.8",
    "claude-sonnet-4":     "claude-sonnet-5",

    # Google shorthands
    "gemini-pro":          "gemini-2.5-pro",
    "gemini-flash":        "gemini-2.5-flash",
    "gemini-flash-lite":   "gemini-2.5-flash-lite",
    "gemini-2.5":          "gemini-2.5-pro",

    # Mistral shorthands
    "mistral-medium":  "mistral-medium-latest",
    "mistral-large":   "mistral-large-latest",
    "mistral-small":   "mistral-small-latest",
    "codestral":       "codestral-latest",
    "magistral":       "magistral-medium-latest",
    "mixtral":         "open-mixtral-8x7b",
    "mixtral-8x7b":    "open-mixtral-8x7b",
    "mixtral-8x22b":   "open-mixtral-8x22b",

    # xAI shorthands
    "grok":            "grok-4.5",
    "grok4":           "grok-4.5",
    "grok4.3":         "grok-4.3",
    "grok4.6":         "grok-4.6",

    # Together shorthands
    "llama-3.3-70b":   "together/llama-3.3-70b",
    "llama3":          "together/llama-3.3-70b",
    "deepseek-v4":     "together/deepseek-v4-pro",
    "qwen3":           "together/qwen3-235b-a22b",

    # Groq shorthands
    "llama-4-scout":   "groq/llama-4-scout-17bx16e",
    "llama3-8b":       "groq/llama-3.1-8b-instant",
    "llama3-70b":      "groq/llama-3.3-70b-versatile",

    # Copilot shorthands
    "copilot-sonnet":  "copilot/claude-sonnet-5",
    "copilot-opus":    "copilot/claude-opus-5",
    "copilot-haiku":   "copilot/claude-haiku-4.5",
    "copilot-gpt5":    "copilot/gpt-5.4",
    "copilot-gpt6":    "copilot/gpt-6-astra",
    "raptor-mini":     "copilot/raptor-mini",
    "mai-code":        "copilot/mai-code-1-flash",
}

# Provider display metadata
_PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai":    {"name": "OpenAI",         "url": "https://openai.com"},
    "anthropic": {"name": "Anthropic",      "url": "https://anthropic.com"},
    "google":    {"name": "Google",         "url": "https://ai.google.dev"},
    "mistral":   {"name": "Mistral AI",     "url": "https://mistral.ai"},
    "xai":       {"name": "xAI",            "url": "https://x.ai"},
    "together":  {"name": "Together AI",    "url": "https://together.ai"},
    "groq":      {"name": "Groq",           "url": "https://groq.com"},
    "copilot":   {"name": "GitHub Copilot", "url": "https://github.com/features/copilot"},
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DeprecatedModelError(Exception):
    """Raised when querying a deprecated model without opt-in."""
    pass


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLM_Provider:
    """Metadata for an LLM provider."""
    id: str       # canonical id, e.g. "openai", "anthropic"
    name: str     # display name, e.g. "OpenAI"
    url: str      # pricing page URL


@dataclass
class LLM_Cost:
    """Itemised cost breakdown for a single LLM API call."""
    model: str              # canonical model name
    input_cost: float       # USD cost for non-cached input tokens
    cached_cost: float      # USD cost for cache-hit tokens (0.0 if unused)
    cache_write_cost: float # USD cost for cache-write tokens (0.0 if unused)
    output_cost: float      # USD cost for output tokens
    total_cost: float       # sum of all cost components
    currency: str           # always "USD"
    breakdown: str          # human-readable multi-line cost summary
    pricing_tier: str = "standard"  # standard, batch, or long_context
    input_rate: float = 0.0          # effective USD per 1M input tokens
    cache_hit_rate: float = 0.0      # effective USD per 1M cache-hit tokens
    cache_write_rate: float = 0.0    # effective USD per 1M cache-write tokens
    output_rate: float = 0.0         # effective USD per 1M output tokens


@dataclass
class LLM_Model:
    """Pricing information and cost calculations for a single LLM model."""

    name: str                    # canonical model key, e.g. "gpt-4o"
    provider: LLM_Provider       # provider metadata
    input: Optional[float]       # USD per 1M input tokens; None = not applicable
    output: Optional[float]      # USD per 1M output tokens; None = not applicable
    cached_input: Optional[float] = None  # USD per 1M cache-hit input tokens
    cache_write: Optional[float] = None   # USD per 1M cache-write tokens
    batch_input: Optional[float] = None   # USD per 1M input tokens (batch API)
    batch_output: Optional[float] = None  # USD per 1M output tokens (batch API)
    long_input: Optional[float] = None    # USD per 1M input tokens (long-context tier)
    long_output: Optional[float] = None   # USD per 1M output tokens (long-context tier)
    long_context_threshold: Optional[int] = None  # token count where longctx pricing kicks in
    deprecated: bool = False              # True if model is deprecated/retired
    notes: Optional[str] = None           # free-form notes (context tiers, etc.)
    long_cached_input: Optional[float] = None  # USD per 1M long-context cache-hit tokens
    long_cache_write: Optional[float] = None   # USD per 1M long-context cache-write tokens

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        cache_write_tokens: int = 0,
        use_batch: bool = False,
        long_context: Optional[bool] = None,
        *,
        cached_input_tokens: Optional[int] = None,
    ) -> LLM_Cost:
        """Calculate the USD cost of a single API call with this model.

        :param input_tokens: Number of non-cached input tokens.
        :type input_tokens: int
        :param output_tokens: Number of output tokens.
        :type output_tokens: int
        :param cache_hit_tokens: Cache-hit tokens (billed at
            ``cached_input`` rate, which is cheaper than standard input).
        :type cache_hit_tokens: int
        :param cache_write_tokens: Tokens written into cache this turn.
            Anthropic and current OpenAI models charge separately for these;
            models without a separate write price fall back to the applicable
            input rate.
        :type cache_write_tokens: int
        :param use_batch: If True and the model has batch pricing, use
            ``batch_input`` / ``batch_output`` rates instead of standard.
            Batch requests are asynchronous (up to 24h) at a significant
            discount (e.g. 50% off for Anthropic, 20% off for xAI).
        :type use_batch: bool
        :param long_context: ``None`` (default) automatically selects the
            long-context tier from the total prompt tokens and
            ``long_context_threshold``. ``True`` forces the long-context tier;
            ``False`` forces standard pricing.
        :type long_context: Optional[bool]
        :param cached_input_tokens: Deprecated keyword alias for
            ``cache_hit_tokens``.
        :type cached_input_tokens: Optional[int]
        :return: Itemised cost breakdown.
        :rtype: LLM_Cost
        """
        m_per = 1_000_000

        if cached_input_tokens is not None:
            if cache_hit_tokens:
                raise TypeError(
                    "Pass only one of cache_hit_tokens and cached_input_tokens"
                )
            warnings.warn(
                "cached_input_tokens is deprecated; use cache_hit_tokens instead",
                DeprecationWarning,
                stacklevel=2,
            )
            cache_hit_tokens = cached_input_tokens

        prompt_tokens = input_tokens + cache_hit_tokens + cache_write_tokens
        use_long_context = (
            prompt_tokens > self.long_context_threshold
            if long_context is None and self.long_context_threshold is not None
            else bool(long_context)
        )

        if use_batch and self.batch_input is not None and self.batch_output is not None:
            in_rate: float = self.batch_input
            out_rate: float = self.batch_output
            pricing_tier = "batch"
            use_long_rates = False
        elif use_long_context and self.long_input is not None and self.long_output is not None:
            in_rate = self.long_input
            out_rate = self.long_output
            pricing_tier = "long_context"
            use_long_rates = True
        else:
            in_rate = self.input or 0.0
            out_rate = self.output or 0.0
            pricing_tier = "standard"
            use_long_rates = False

        if use_long_rates:
            cache_hit_rate: float = (
                self.long_cached_input
                if self.long_cached_input is not None
                else self.cached_input if self.cached_input is not None else in_rate
            )
            cache_write_rate: float = (
                self.long_cache_write
                if self.long_cache_write is not None
                else self.cache_write if self.cache_write is not None else in_rate
            )
        else:
            cache_hit_rate = self.cached_input if self.cached_input is not None else in_rate
            cache_write_rate = self.cache_write if self.cache_write is not None else in_rate

        input_cost: float       = (input_tokens        / m_per) * in_rate
        cached_cost: float      = (cache_hit_tokens    / m_per) * cache_hit_rate
        cache_write_cost: float = (cache_write_tokens  / m_per) * cache_write_rate
        output_cost: float      = (output_tokens       / m_per) * out_rate
        total: float            = input_cost + cached_cost + cache_write_cost + output_cost

        breakdown = f"{input_tokens:,} input @ ${in_rate}/1M = ${input_cost:.4f}"
        if cache_write_tokens:
            breakdown += f"\n{cache_write_tokens:,} cache write @ ${cache_write_rate}/1M = ${cache_write_cost:.4f}"
        if cache_hit_tokens:
            breakdown += f"\n{cache_hit_tokens:,} cache hit  @ ${cache_hit_rate}/1M = ${cached_cost:.4f}"
        breakdown += f"\n{output_tokens:,} output @ ${out_rate}/1M = ${output_cost:.4f}"
        breakdown += f"\nTotal = ${total:.4f}"

        return LLM_Cost(
            model=self.name,
            input_cost=round(input_cost, 6),
            cached_cost=round(cached_cost, 6),
            cache_write_cost=round(cache_write_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total, 6),
            currency="USD",
            breakdown=breakdown,
            pricing_tier=pricing_tier,
            input_rate=in_rate,
            cache_hit_rate=cache_hit_rate,
            cache_write_rate=cache_write_rate,
            output_rate=out_rate,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(model: str) -> Optional[str]:
    """Resolve a model name to its canonical key via exact match or alias.
    Returns None if not found — no fuzzy matching.
    """
    if model in _PRICES:
        return model
    normalized = model.lower().strip().replace(" ", "-")
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    if normalized in _PRICES:
        return normalized
    alias_key = _ALIASES.get(normalized)
    if alias_key and alias_key in _PRICES:
        return alias_key
    return None


def _fuzzy_candidates(model: str, n: int = 5) -> List[str]:
    """Return up to n fuzzy-matched canonical keys (used only for error hints)."""
    normalized = model.lower().strip().replace(" ", "-")
    all_keys = list(_PRICES.keys()) + list(_ALIASES.keys())
    matches = difflib.get_close_matches(normalized, all_keys, n=n, cutoff=0.4)
    resolved: List[str] = []
    seen: set = set()
    for m in matches:
        key = _ALIASES.get(m, m)
        if key in _PRICES and key not in seen:
            resolved.append(key)
            seen.add(key)
    return resolved


def _make_model(key: str) -> LLM_Model:
    """Construct an LLM_Model from a canonical price key."""
    data = _PRICES[key]
    provider_id = str(data.get("provider", ""))
    prov_data = _PROVIDERS.get(provider_id, {"name": provider_id, "url": ""})
    provider = LLM_Provider(id=provider_id, name=prov_data["name"], url=prov_data["url"])
    return LLM_Model(
        name=key,
        provider=provider,
        input=data.get("input"),  # type: ignore[arg-type]
        output=data.get("output"),  # type: ignore[arg-type]
        cached_input=data.get("cached_input"),  # type: ignore[arg-type]
        cache_write=data.get("cache_write"),  # type: ignore[arg-type]
        batch_input=data.get("batch_input"),  # type: ignore[arg-type]
        batch_output=data.get("batch_output"),  # type: ignore[arg-type]
        long_input=data.get("long_input"),  # type: ignore[arg-type]
        long_output=data.get("long_output"),  # type: ignore[arg-type]
        long_context_threshold=data.get("longctx_threshold"),  # type: ignore[arg-type]
        deprecated=bool(data.get("deprecated", False)),
        notes=data.get("notes"),  # type: ignore[arg-type]
        long_cached_input=data.get("long_cached_input"),  # type: ignore[arg-type]
        long_cache_write=data.get("long_cache_write"),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(model: str, include_deprecated: bool = False) -> LLM_Model:
    """Look up a model by exact name or alias, returning an :class:`LLM_Model`.

    :param model: Model name (exact match or registered alias).
    :type model: str
    :param include_deprecated: If False (default), raises
        :exc:`DeprecatedModelError` for deprecated models.
    :type include_deprecated: bool
    :return: Model pricing and metadata.
    :rtype: LLM_Model
    :raises KeyError: If the model is not found. The error message includes
        fuzzy suggestions when close matches exist.
    :raises DeprecatedModelError: If the model is deprecated and
        ``include_deprecated=False``.

    Example::

        model = get_model("gpt-4o")
        print(f"${model.input} in / ${model.output} out per 1M tokens")
        cost = model.cost(input_tokens=10_000, output_tokens=2_000)
    """
    key = _resolve(model)
    if key:
        data = _PRICES[key]
        if data.get("deprecated") and not include_deprecated:
            raise DeprecatedModelError(
                f"Model {key!r} is deprecated. "
                f"Pass include_deprecated=True to retrieve it anyway."
            )
        return _make_model(key)

    candidates = _fuzzy_candidates(model, n=5)
    if candidates:
        raise KeyError(
            f"Model {model!r} not found. Did you mean one of: {candidates}?"
        )
    raise KeyError(f"Model {model!r} not found.")


def get_models(
    provider: Optional[str] = None,
    include_deprecated: bool = False,
) -> List[LLM_Model]:
    """Return a list of known models, optionally filtered by provider.

    :param provider: Provider id (e.g. ``'openai'``, ``'anthropic'``),
        or ``None`` for all providers.
    :type provider: Optional[str]
    :param include_deprecated: If False (default), deprecated models are
        excluded.
    :type include_deprecated: bool
    :return: List of models matching the filter.
    :rtype: List[LLM_Model]

    Example::

        for m in get_models("anthropic"):
            print(m.name, m.input, m.output)
    """
    results: List[LLM_Model] = []
    for key, data in _PRICES.items():
        if provider and data.get("provider") != provider.lower():
            continue
        if not include_deprecated and data.get("deprecated"):
            continue
        results.append(_make_model(key))
    return results


def get_providers() -> List[LLM_Provider]:
    """Return metadata for all known providers.

    :return: List of provider objects.
    :rtype: List[LLM_Provider]

    Example::

        for p in get_providers():
            print(p.id, p.name, p.url)

        # Build a lookup by id if needed:
        by_id = {p.id: p for p in get_providers()}
    """
    return [
        LLM_Provider(id=k, name=v["name"], url=v["url"])
        for k, v in _PROVIDERS.items()
    ]


def search_models(query: str) -> List[LLM_Model]:
    """Search for models by substring match against model names and notes.

    The search is case-insensitive. Deprecated models are included in
    results (check ``model.deprecated`` to filter if needed).

    :param query: Substring to search for in model names and notes fields.
    :type query: str
    :return: All matching models.
    :rtype: List[LLM_Model]

    Example::

        search_models("flash")  # -> all Gemini Flash variants
        search_models("mini")   # -> all mini/small models across providers
    """
    q = query.lower()
    return [
        _make_model(key)
        for key, data in _PRICES.items()
        if q in key.lower() or q in str(data.get("notes", "")).lower()
    ]
