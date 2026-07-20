"""
LLM model pricing database and cost-calculation utilities.

Prices last updated: 2026-07-20
All token prices are USD per 1,000,000 tokens unless noted otherwise.

Usage::

    from tdvutil import get_price, list_models, calculate_cost, find_model, compare_models
    from tdvutil import DeprecatedModelError

    price = get_price("gpt-4o")
    print(f"${price['input']} in / ${price['output']} out per 1M tokens")

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
from typing import Dict, List, Optional

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
#   deprecated : bool        — model is no longer active/recommended
#   notes : str
# ---------------------------------------------------------------------------

PRICES: Dict[str, Dict[str, object]] = {

    # =========================================================================
    # OPENAI
    # =========================================================================

    # --- Flagship / latest ---
    "gpt-5.6-sol":      {"provider": "openai", "input": 5.00,   "cached_input": 0.50,  "output": 30.00,  "notes": "Long-ctx: $10/$45 in/out"},
    "gpt-5.6-terra":    {"provider": "openai", "input": 2.50,   "cached_input": 0.25,  "output": 15.00,  "notes": "Long-ctx: $5/$22.50 in/out"},
    "gpt-5.6-luna":     {"provider": "openai", "input": 1.00,   "cached_input": 0.10,  "output": 6.00,   "notes": "Long-ctx: $2/$9 in/out"},
    "gpt-5.5":          {"provider": "openai", "input": 5.00,   "cached_input": 0.50,  "output": 30.00,  "notes": "Long-ctx: $10/$45 in/out"},
    "gpt-5.5-pro":      {"provider": "openai", "input": 30.00,  "cached_input": None,  "output": 180.00, "notes": "Long-ctx: $60/$270 in/out"},
    "gpt-5.4":          {"provider": "openai", "input": 2.50,   "cached_input": 0.25,  "output": 15.00},
    "gpt-5.4-mini":     {"provider": "openai", "input": 0.75,   "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano":     {"provider": "openai", "input": 0.20,   "cached_input": 0.02,  "output": 1.25},
    "gpt-5.4-pro":      {"provider": "openai", "input": 30.00,  "cached_input": None,  "output": 180.00},
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

    "claude-fable-5":            {"provider": "anthropic", "input": 10.00, "output": 50.00,
                                  "cached_input": 1.00,   "cache_write": 12.50,
                                  "batch_input": 5.00,    "batch_output": 25.00,
                                  "notes": "Next-gen intelligence for long-running agents"},
    "claude-opus-4.8":           {"provider": "anthropic", "input": 5.00,  "output": 25.00,
                                  "cached_input": 0.50,   "cache_write": 6.25,
                                  "batch_input": 2.50,    "batch_output": 12.50,
                                  "notes": "Fast mode available at 2x pricing"},
    "claude-sonnet-5":           {"provider": "anthropic", "input": 2.00,  "output": 10.00,
                                  "cached_input": 0.20,   "cache_write": 2.50,
                                  "batch_input": 1.00,    "batch_output": 5.00,
                                  "notes": "Intro pricing until 2026-08-31; rises to $3/$15"},
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
                              "notes": "Tiered: >200k ctx: $2.50/$15.00; cache storage $4.50/1M tok/hr"},
    "gemini-2.5-flash":      {"provider": "google", "input": 0.30,  "output": 2.50,
                              "cached_input": 0.03,
                              "notes": "Audio input: $1.00/1M; 1M ctx window"},
    "gemini-2.5-flash-lite": {"provider": "google", "input": 0.10,  "output": 0.40,
                              "cached_input": 0.01,
                              "notes": "Audio input: $0.30/1M"},
    "gemini-3.5-flash":      {"provider": "google", "input": 1.50,  "output": 9.00,
                              "cached_input": 0.15,
                              "notes": "GA; thinking tokens billed in output"},
    "gemini-3.1-pro-preview":{"provider": "google", "input": 2.00,  "output": 12.00,
                              "cached_input": 0.20,
                              "notes": "Preview; >200k ctx: $4/$18; cache storage $4.50/1M tok/hr"},
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
                                    "notes": "500k context"},
    "grok-4.3":                    {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-multi-agent-0309":  {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-0309-reasoning":    {"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "notes": "1M context; batch -20%"},
    "grok-4.20-0309-non-reasoning":{"provider": "xai", "input": 1.25,  "cached_input": 0.20,  "output": 2.50,
                                    "notes": "1M context; batch -20%"},
    "grok-build-0.1":              {"provider": "xai", "input": 1.00,  "cached_input": 0.20,  "output": 2.00,
                                    "notes": "Code API; 256k context"},

    # =========================================================================
    # TOGETHER AI  (hosted open-source models)
    # =========================================================================

    "together/llama-3.3-70b":          {"provider": "together", "input": 1.04,  "output": 1.04},
    "together/llama-3-8b-instruct-lite":{"provider": "together", "input": 0.14,  "output": 0.14},
    "together/qwen3-235b-a22b":        {"provider": "together", "input": 0.20,  "output": 0.60},
    "together/qwen3.7-plus":           {"provider": "together", "input": 0.32,  "output": 1.28},
    "together/qwen3.7-max":            {"provider": "together", "input": 1.25,  "cached_input": 0.13, "output": 3.75},
    "together/qwen3.5-397b":           {"provider": "together", "input": 0.60,  "cached_input": 0.35, "output": 3.60},
    "together/qwen3.5-9b":             {"provider": "together", "input": 0.17,  "output": 0.25},
    "together/qwen2.5-7b-turbo":       {"provider": "together", "input": 0.30,  "output": 0.30},
    "together/gemma-4-31b":            {"provider": "together", "input": 0.39,  "output": 0.97},
    "together/deepseek-v4-pro":        {"provider": "together", "input": 1.74,  "cached_input": 0.20, "output": 3.48},
    "together/kimi-k2.7-code":         {"provider": "together", "input": 0.95,  "cached_input": 0.19, "output": 4.00},
    "together/kimi-k2.6":              {"provider": "together", "input": 1.20,  "cached_input": 0.20, "output": 4.50},
    "together/gpt-oss-120b":           {"provider": "together", "input": 0.15,  "output": 0.60},
    "together/gpt-oss-20b":            {"provider": "together", "input": 0.05,  "output": 0.20},
    "together/cogito-v2.1-671b":       {"provider": "together", "input": 1.25,  "output": 1.25},
    "together/nvidia-nemotron-3-ultra":{"provider": "together", "input": 0.60,  "cached_input": 0.20, "output": 3.60},

    # =========================================================================
    # GITHUB COPILOT  (1 AI credit = $0.01 USD)
    # =========================================================================

    "copilot/gpt-5-mini":          {"provider": "copilot", "input": 0.25,  "cached_input": 0.025, "output": 2.00},
    "copilot/gpt-5.3-codex":       {"provider": "copilot", "input": 1.75,  "cached_input": 0.175, "output": 14.00},
    "copilot/gpt-5.4":             {"provider": "copilot", "input": 2.50,  "cached_input": 0.25,  "output": 15.00,
                                    "notes": "Long-ctx (>272k): $5/$22.50 in/out"},
    "copilot/gpt-5.4-mini":        {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 4.50},
    "copilot/gpt-5.4-nano":        {"provider": "copilot", "input": 0.20,  "cached_input": 0.02,  "output": 1.25},
    "copilot/gpt-5.5":             {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "output": 30.00,
                                    "notes": "Long-ctx (>272k): $10/$45 in/out"},
    "copilot/gpt-5.6-luna":        {"provider": "copilot", "input": 1.00,  "cached_input": 0.10,  "output": 6.00,
                                    "notes": "Long-ctx (>200k): $2/$9 in/out"},
    "copilot/gpt-5.6-sol":         {"provider": "copilot", "input": 5.00,  "cached_input": 0.50,  "output": 30.00,
                                    "notes": "Long-ctx (>272k): $10/$45 in/out"},
    "copilot/gpt-5.6-terra":       {"provider": "copilot", "input": 2.50,  "cached_input": 0.25,  "output": 15.00,
                                    "notes": "Long-ctx (>272k): $5/$22.50 in/out"},
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
    "copilot/claude-opus-4.8-fast":{"provider": "copilot", "input": 10.00, "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00,
                                    "notes": "Fast mode preview; up to 2.5x faster"},
    "copilot/claude-fable-5":      {"provider": "copilot", "input": 10.00, "cached_input": 1.00,  "cache_write": 12.50, "output": 50.00},
    "copilot/gemini-2.5-pro":      {"provider": "copilot", "input": 1.25,  "cached_input": 0.125, "output": 10.00},
    "copilot/gemini-3-flash":      {"provider": "copilot", "input": 0.50,  "cached_input": 0.05,  "output": 3.00,
                                    "notes": "Public preview"},
    "copilot/gemini-3.1-pro":      {"provider": "copilot", "input": 2.00,  "cached_input": 0.20,  "output": 12.00,
                                    "notes": "Public preview; Long-ctx (>200k): $4/$18 in/out"},
    "copilot/gemini-3.5-flash":    {"provider": "copilot", "input": 1.50,  "cached_input": 0.15,  "output": 9.00},
    "copilot/raptor-mini":         {"provider": "copilot", "input": 0.25,  "cached_input": 0.025, "output": 2.00,
                                    "notes": "GitHub fine-tuned model"},
    "copilot/mai-code-1-flash":    {"provider": "copilot", "input": 0.75,  "cached_input": 0.075, "output": 4.50,
                                    "notes": "Microsoft model"},
    "copilot/kimi-k2.7-code":      {"provider": "copilot", "input": 0.95,  "cached_input": 0.19,  "output": 4.00,
                                    "notes": "Moonshot AI model"},

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
    "groq/gpt-oss-20b":              {"provider": "groq", "input": 0.075, "output": 0.30,
                                      "notes": "128k ctx; ~1000 tok/s"},
    "groq/gpt-oss-120b":             {"provider": "groq", "input": 0.15,  "output": 0.60,
                                      "notes": "128k ctx; ~500 tok/s"},
    "groq/kimi-k2-instruct":         {"provider": "groq", "input": 1.00,  "cached_input": 0.50, "output": 3.00},
}

# ---------------------------------------------------------------------------
# Aliases — common shorthand → canonical key
# ---------------------------------------------------------------------------
ALIASES: Dict[str, str] = {
    # OpenAI shorthands
    "gpt4o":          "gpt-4o",
    "gpt4omini":      "gpt-4o-mini",
    "gpt4":           "gpt-4-0613",
    "gpt4turbo":      "gpt-4-turbo-2024-04-09",
    "gpt35":          "gpt-3.5-turbo",
    "gpt35turbo":     "gpt-3.5-turbo",
    "o1pro":          "o1-pro",
    "o3pro":          "o3-pro",

    # Anthropic shorthands
    "claude-fable-5":      "claude-fable-5",
    "fable5":              "claude-fable-5",
    "claude-opus":         "claude-opus-4.8",
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
    "copilot-opus":    "copilot/claude-opus-4.8",
    "copilot-haiku":   "copilot/claude-haiku-4.5",
    "copilot-gpt5":    "copilot/gpt-5.4",
    "raptor-mini":     "copilot/raptor-mini",
    "mai-code":        "copilot/mai-code-1-flash",
}

# Provider display metadata
PROVIDERS: Dict[str, Dict[str, str]] = {
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
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(model: str) -> Optional[str]:
    """Resolve a model name string to a canonical key, or None if not found."""
    if model in PRICES:
        return model
    normalized = model.lower().strip().replace(" ", "-")
    if normalized in ALIASES:
        return ALIASES[normalized]
    if normalized in PRICES:
        return normalized
    alias_key = ALIASES.get(normalized)
    if alias_key and alias_key in PRICES:
        return alias_key
    return None


def _fuzzy_candidates(model: str, n: int = 5) -> List[str]:
    """Return up to n fuzzy-matched canonical keys for a model string."""
    normalized = model.lower().strip().replace(" ", "-")
    all_keys = list(PRICES.keys()) + list(ALIASES.keys())
    matches = difflib.get_close_matches(normalized, all_keys, n=n, cutoff=0.4)
    resolved: List[str] = []
    seen: set = set()
    for m in matches:
        key = ALIASES.get(m, m)
        if key in PRICES and key not in seen:
            resolved.append(key)
            seen.add(key)
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_model(model: str) -> Optional[str]:
    """
    Resolve a model name (exact, alias, or fuzzy) to its canonical key.
    Returns None if nothing matched.

    :param model: Model name to look up (exact, alias, or approximate).
    :type model: str
    :return: Canonical model key, or None if not found.
    :rtype: Optional[str]

    Example::

        find_model("gpt4o")        # -> "gpt-4o"
        find_model("claude")       # -> "claude-sonnet-5"  (via alias)
        find_model("gemini flash") # -> "gemini-2.5-flash" (fuzzy)
    """
    exact = _resolve(model)
    if exact:
        return exact
    candidates = _fuzzy_candidates(model, n=1)
    return candidates[0] if candidates else None


def get_price(model: str, include_deprecated: bool = False) -> Dict[str, object]:
    """
    Look up pricing for a model.

    :param model: Model name (exact, alias, or approximate).
    :type model: str
    :param include_deprecated: If False (default), raises DeprecatedModelError
        for deprecated models. Set True to retrieve pricing anyway.
    :type include_deprecated: bool
    :return: Dict with ``model``, ``provider``, ``input``, ``output``, and any
        additional fields (``cached_input``, ``batch_input``, ``notes``, etc.).
    :rtype: dict
    :raises KeyError: If the model can't be found even with fuzzy matching.
    :raises DeprecatedModelError: If the model is deprecated and
        ``include_deprecated=False``.

    Example::

        price = get_price("gpt-4o")
        print(f"${price['input']} in / ${price['output']} out per 1M tokens")
    """
    key = _resolve(model)
    if key:
        data = PRICES[key]
        if data.get("deprecated") and not include_deprecated:
            raise DeprecatedModelError(
                f"Model {key!r} is deprecated. "
                f"Pass include_deprecated=True to get its pricing anyway."
            )
        return {"model": key, **data}

    candidates = _fuzzy_candidates(model, n=5)
    if candidates:
        raise KeyError(
            f"Model {model!r} not found. Did you mean one of: {candidates}?"
        )
    raise KeyError(f"Model {model!r} not found and no similar models matched.")


def list_models(
    provider: Optional[str] = None,
    include_deprecated: bool = False,
) -> List[Dict[str, object]]:
    """
    List all known models, optionally filtered by provider.

    :param provider: One of ``'openai'``, ``'anthropic'``, ``'google'``,
        ``'mistral'``, ``'xai'``, ``'together'``, ``'groq'``, ``'copilot'``,
        or None for all providers.
    :type provider: Optional[str]
    :param include_deprecated: If False (default), deprecated models are
        excluded.
    :type include_deprecated: bool
    :return: List of dicts, each with model name and pricing info.
    :rtype: list[dict]

    Example::

        for m in list_models("anthropic"):
            print(m["model"], m["input"], m["output"])
    """
    results = []
    for key, data in PRICES.items():
        if provider and data.get("provider") != provider.lower():
            continue
        if not include_deprecated and data.get("deprecated"):
            continue
        results.append({"model": key, **data})
    return results


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    use_batch: bool = False,
    include_deprecated: bool = False,
) -> Dict[str, object]:
    """
    Calculate the USD cost of a single API call.

    :param model: Model name (exact, alias, or approximate).
    :type model: str
    :param input_tokens: Number of non-cached input tokens.
    :type input_tokens: int
    :param output_tokens: Number of output tokens.
    :type output_tokens: int
    :param cached_input_tokens: Cache-hit tokens (billed at cached_input rate).
    :type cached_input_tokens: int
    :param cache_write_tokens: Tokens written into cache this turn.
        Anthropic charges ~1.25x input for these; OpenAI/Google have no
        separate write cost.
    :type cache_write_tokens: int
    :param use_batch: If True, use batch_input/batch_output pricing where
        available (Anthropic), otherwise standard.
    :type use_batch: bool
    :param include_deprecated: Allow cost calculation on deprecated models.
    :type include_deprecated: bool
    :return: Dict with ``model``, ``input_cost``, ``cached_cost``,
        ``cache_write_cost``, ``output_cost``, ``total_cost``, ``currency``,
        and a human-readable ``breakdown`` string.
    :rtype: dict
    :raises KeyError: If the model is not found.
    :raises DeprecatedModelError: If the model is deprecated and
        ``include_deprecated=False``.

    Example::

        cost = calculate_cost("claude-sonnet-5",
                              input_tokens=5_000,
                              output_tokens=2_000,
                              cache_write_tokens=50_000,
                              cached_input_tokens=50_000)
        print(f"Total: ${cost['total_cost']:.4f}")
    """
    key = find_model(model)
    if key is None:
        raise KeyError(f"Model {model!r} not found.")
    data = PRICES[key]
    if data.get("deprecated") and not include_deprecated:
        raise DeprecatedModelError(
            f"Model {key!r} is deprecated. "
            f"Pass include_deprecated=True to get its pricing anyway."
        )

    m_per = 1_000_000

    if use_batch and "batch_input" in data:
        in_rate: float = float(data["batch_input"])  # type: ignore[arg-type]
        out_rate: float = float(data["batch_output"])  # type: ignore[arg-type]
    else:
        in_rate = float(data.get("input") or 0.0)  # type: ignore[arg-type]
        out_rate = float(data.get("output") or 0.0)  # type: ignore[arg-type]

    cache_hit_rate: float = float(data["cached_input"]) if data.get("cached_input") is not None else in_rate  # type: ignore[arg-type]
    cache_write_rate: float = float(data["cache_write"]) if data.get("cache_write") is not None else in_rate  # type: ignore[arg-type]

    input_cost: float       = (input_tokens        / m_per) * in_rate
    cached_cost: float      = (cached_input_tokens / m_per) * cache_hit_rate
    cache_write_cost: float = (cache_write_tokens  / m_per) * cache_write_rate
    output_cost: float      = (output_tokens       / m_per) * out_rate
    total: float            = input_cost + cached_cost + cache_write_cost + output_cost

    breakdown = f"{input_tokens:,} input @ ${in_rate}/1M = ${input_cost:.4f}"
    if cache_write_tokens:
        breakdown += f"\n{cache_write_tokens:,} cache write @ ${cache_write_rate}/1M = ${cache_write_cost:.4f}"
    if cached_input_tokens:
        breakdown += f"\n{cached_input_tokens:,} cache hit  @ ${cache_hit_rate}/1M = ${cached_cost:.4f}"
    breakdown += f"\n{output_tokens:,} output @ ${out_rate}/1M = ${output_cost:.4f}"
    breakdown += f"\nTotal = ${total:.4f}"

    return {
        "model":            key,
        "input_cost":       round(input_cost,       6),
        "cached_cost":      round(cached_cost,       6),
        "cache_write_cost": round(cache_write_cost,  6),
        "output_cost":      round(output_cost,       6),
        "total_cost":       round(total,             6),
        "currency":         "USD",
        "breakdown":        breakdown,
    }


def compare_models(
    models: List[str],
    input_tokens: int = 1_000_000,
    output_tokens: int = 1_000_000,
) -> List[Dict[str, object]]:
    """
    Compare costs across multiple models for the same token counts.
    Results are sorted cheapest first.

    :param models: List of model names to compare.
    :type models: list[str]
    :param input_tokens: Input token count for comparison (default 1M).
    :type input_tokens: int
    :param output_tokens: Output token count for comparison (default 1M).
    :type output_tokens: int
    :return: Sorted list of cost dicts (same shape as :func:`calculate_cost`).
    :rtype: list[dict]

    Example::

        results = compare_models(
            ["gpt-4o", "claude-sonnet-5", "gemini-2.5-flash"],
            input_tokens=100_000, output_tokens=20_000,
        )
        for r in results:
            print(r["model"], f"${r['total_cost']:.4f}")
    """
    results = []
    for m in models:
        try:
            cost = calculate_cost(m, input_tokens, output_tokens)
            results.append(cost)
        except (KeyError, DeprecatedModelError) as e:
            results.append({"model": m, "error": str(e), "total_cost": float("inf")})
    return sorted(results, key=lambda x: x.get("total_cost", float("inf")))  # type: ignore[return-value]


def cheapest(
    provider: Optional[str] = None,
    n: int = 5,
    input_tokens: int = 1_000_000,
    output_tokens: int = 1_000_000,
    include_deprecated: bool = False,
) -> List[Dict[str, object]]:
    """
    Find the n cheapest models by combined input+output cost for given token
    counts.

    :param provider: Restrict to one provider, or None for all.
    :type provider: Optional[str]
    :param n: Number of results to return.
    :type n: int
    :param input_tokens: Input tokens for cost estimate.
    :type input_tokens: int
    :param output_tokens: Output tokens for cost estimate.
    :type output_tokens: int
    :param include_deprecated: Include deprecated models in ranking.
    :type include_deprecated: bool
    :return: Up to n cost dicts sorted cheapest first.
    :rtype: list[dict]

    Example::

        for m in cheapest(provider="openai", n=3):
            print(m["model"], f"${m['total_cost']:.4f}")
    """
    model_keys = [m["model"] for m in list_models(provider, include_deprecated)]
    results = [
        calculate_cost(str(m), input_tokens, output_tokens, include_deprecated=include_deprecated)
        for m in model_keys
        if PRICES[str(m)].get("input") is not None
    ]
    return sorted(results, key=lambda x: x["total_cost"])[:n]  # type: ignore[return-value]


def search(query: str) -> List[Dict[str, object]]:
    """
    Search models by partial name match (case-insensitive substring).
    Deprecated models are included in search results.

    :param query: Substring to search for in model names and notes.
    :type query: str
    :return: All matching model dicts.
    :rtype: list[dict]

    Example::

        search("flash")  # -> all Gemini Flash variants
        search("mini")   # -> all mini/small models across providers
    """
    q = query.lower()
    return [
        {"model": key, **data}
        for key, data in PRICES.items()
        if q in key.lower() or q in str(data.get("notes", "")).lower()
    ]
