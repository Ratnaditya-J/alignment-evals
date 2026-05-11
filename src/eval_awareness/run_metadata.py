"""Run-time metadata helpers for cross-protocol analysis.

These helpers let post-processing tools (``rescore_cross_protocol.py``,
``validate_refusal_scorer.py``) recover the information they need to slice
results without re-running the benchmark:

* ``task_family_from_example_id`` — infers task family from an ``EvalAwarenessExample``
  id even when the rollouts.jsonl was produced before we logged task_family
  natively. Matches the prefixes used by ``dataset_loaders``.

* ``ModelTier`` / ``classify_model`` — assigns each model under test to a
  vendor (openai/anthropic/openrouter/...) AND a reasoning tier (reasoning
  vs non-reasoning). This is what the "reasoning vs non-reasoning within
  and across providers" analysis is built on.

* ``reasoning_provenance`` — closed (polished, hidden CoT) vs open
  (full reasoning visible). Lets us compare how visible CoT changes
  measured eval-framing effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .providers import (
    _is_anthropic_reasoning_model,
    _is_open_reasoning_model,
    _is_openai_reasoning_model,
)

# ---------------------------------------------------------------------------
# Task family inference
# ---------------------------------------------------------------------------

# Maps the leading token of an example_id (everything before the first colon)
# to the canonical task_family used downstream. The keys mirror what the
# loaders emit in dataset_loaders.py.
_FAMILY_BY_PREFIX: Dict[str, str] = {
    "mmlu": "knowledge",
    "gpqa": "knowledge",
    "gsm8k": "math",
    "math": "math",
    "truthfulqa": "truthfulness",
    "humaneval": "code",
    "mbpp": "code",
    "beavertails": "safety",
    "synth_safety": "safety",
}


def task_family_from_example_id(example_id: str) -> str:
    """Best-effort inference of the task family from an example id.

    Returns 'unknown' when no prefix matches; callers should preserve that
    bucket in their reporting so we never silently drop rollouts.
    """
    if not example_id:
        return "unknown"
    # First-colon split handles "mmlu:anatomy:test:42" etc.
    head = example_id.split(":", 1)[0]
    if head in _FAMILY_BY_PREFIX:
        return _FAMILY_BY_PREFIX[head]
    # synth_safety_<i> doesn't use a colon - check prefix match.
    for prefix, family in _FAMILY_BY_PREFIX.items():
        if example_id.startswith(prefix):
            return family
    return "unknown"


# ---------------------------------------------------------------------------
# Model tier classification (vendor + reasoning + provenance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Classified metadata for one model under test."""

    label: str  # the report label, e.g. "openai:gpt-5.5"
    vendor: str  # openai / anthropic / openrouter / ollama / unknown
    model_id: str  # the bare model id, e.g. "gpt-5.5"
    is_reasoning: bool
    reasoning_provenance: str  # "closed_polished" / "open_full" / "none"
    is_open_weights: bool  # True for Meta/Mistral/DeepSeek/Qwen/Moonshot/etc.
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "vendor": self.vendor,
            "model_id": self.model_id,
            "is_reasoning": self.is_reasoning,
            "reasoning_provenance": self.reasoning_provenance,
            "is_open_weights": self.is_open_weights,
            "notes": self.notes,
        }


# Closed-source labs ship API-only. Their models can be served through
# OpenRouter or other gateways but they're still 'closed_source' from a
# weights-release standpoint. The first-party vendor strings are detected
# below; routed access via openrouter is detected by model_id prefix.
_CLOSED_SOURCE_VENDORS = {
    "openai",
    "openai_compatible",
    "anthropic",
    "anthropic_sdk",
    "gemini",
    "gemini_sdk",
    "cohere",
}

# When a closed-source model is served via OpenRouter, the model_id keeps
# the originating lab's prefix (e.g. "openai/gpt-4o-mini",
# "anthropic/claude-opus-4-7", "google/gemini-2.5-pro"). Detect those so
# bucketing isn't fooled by the routing layer.
_CLOSED_SOURCE_MODEL_PREFIXES_VIA_ROUTER = (
    "openai/",
    "anthropic/",
    "google/gemini",
    "cohere/",
)


def _is_open_weights_model(vendor: str, model_id: str) -> bool:
    """Detect whether a model's weights are publicly released.

    Open-weights = Meta Llama, DeepSeek, Qwen, Mistral, Moonshot Kimi,
    Zhipu GLM, Allen AI OLMo, 01.AI Yi, Google Gemma (NOT Gemini), etc.
    Closed-source = OpenAI, Anthropic, Gemini (proper), Cohere closed.

    Defaults to True (open-weights) when uncertain - on the rationale
    that an unknown OpenRouter-routed model is more likely an open
    fine-tune than a routed closed-source product. Override by adding
    new vendor strings to ``_CLOSED_SOURCE_VENDORS`` or model-id prefixes
    to ``_CLOSED_SOURCE_MODEL_PREFIXES_VIA_ROUTER``.
    """
    if vendor.lower() in _CLOSED_SOURCE_VENDORS:
        return False
    lower_id = model_id.lower()
    if any(lower_id.startswith(p) for p in _CLOSED_SOURCE_MODEL_PREFIXES_VIA_ROUTER):
        return False
    return True


def classify_model(label: str) -> ModelTier:
    """Classify a model name into vendor + reasoning + provenance.

    ``label`` is the per-model identifier used in reports, typically of the
    form ``"<vendor>:<model_id>"`` (e.g. ``"openai:gpt-5.5"``,
    ``"anthropic:claude-opus-4-7"``, ``"openrouter:deepseek/deepseek-r1"``).
    Labels without a vendor prefix fall back to the model id itself; the
    classifier then tries known prefixes to recover the vendor.
    """
    raw = label.strip()
    vendor = "unknown"
    model_id = raw
    if ":" in raw:
        prefix, _, suffix = raw.partition(":")
        vendor = prefix.lower()
        model_id = suffix

    # Recover vendor for unprefixed labels.
    if vendor == "unknown":
        low = model_id.lower()
        if low.startswith(("gpt-", "o1", "o3", "o4")):
            vendor = "openai"
        elif low.startswith(("claude", "anthropic")):
            vendor = "anthropic"
        elif "/" in low:
            # Hugging Face / OpenRouter style "owner/model"
            vendor = "openrouter"
        elif low.startswith(("gemini", "google")):
            vendor = "gemini"
        elif low.startswith(("llama", "qwen", "deepseek", "mistral", "kimi")):
            vendor = "openrouter"

    is_reasoning = False
    provenance = "none"
    notes = ""

    if vendor in ("openai", "openai_compatible", "litellm"):
        is_reasoning = _is_openai_reasoning_model(model_id)
        provenance = "closed_polished" if is_reasoning else "none"
        if is_reasoning:
            notes = "closed-source reasoning model with polished/hidden CoT"
    elif vendor in ("anthropic", "anthropic_sdk"):
        is_reasoning = _is_anthropic_reasoning_model(model_id)
        provenance = "closed_polished" if is_reasoning else "none"
        if is_reasoning:
            notes = "closed-source reasoning model with polished/hidden CoT"
    elif vendor in ("openrouter", "together", "fireworks", "vllm", "ollama", "llama_cpp"):
        if _is_open_reasoning_model(model_id):
            is_reasoning = True
            provenance = "open_full"
            notes = "open-weight reasoning model with visible CoT"
    elif vendor == "gemini":
        # Gemini reasoning models (e.g. gemini-2.5-pro) have hidden CoT.
        lower = model_id.lower()
        if "thinking" in lower or "2.5" in lower or "3" in lower:
            is_reasoning = True
            provenance = "closed_polished"

    return ModelTier(
        label=label,
        vendor=vendor,
        model_id=model_id,
        is_reasoning=is_reasoning,
        reasoning_provenance=provenance,
        is_open_weights=_is_open_weights_model(vendor, model_id),
        notes=notes,
    )


def reasoning_tier_bucket(tier: ModelTier) -> str:
    """Short human-readable bucket used in cross-tier comparison tables."""
    if not tier.is_reasoning:
        return "non_reasoning"
    if tier.reasoning_provenance == "open_full":
        return "open_full_reasoning"
    if tier.reasoning_provenance == "closed_polished":
        return "closed_polished_reasoning"
    return "reasoning_unknown"


def source_type_bucket(tier: ModelTier) -> str:
    """Open vs closed source bucketing for a separate analysis dimension.

    Returns ``open_source`` for models whose weights are publicly released
    (Meta, Mistral, DeepSeek, Qwen, Moonshot, Zhipu, etc.) and
    ``closed_source`` for API-only models (OpenAI, Anthropic, Gemini,
    Cohere closed). This is orthogonal to the reasoning-tier bucket -
    Anthropic Claude can be closed_source AND reasoning, DeepSeek R1 is
    open_source AND reasoning.
    """
    return "open_source" if tier.is_open_weights else "closed_source"
