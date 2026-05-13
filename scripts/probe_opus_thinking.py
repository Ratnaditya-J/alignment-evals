#!/usr/bin/env python3
"""Diagnostic: probe opus-4.7 with a battery of thinking configs and
report which ones (a) return HTTP 200 and (b) actually surface a
``thinking``-type content block in the response.

Goal: find a config that forces opus-4.7 to expose its CoT. Adaptive
mode lets the model skip thinking, which produces empty reasoning
traces for the Goodfire VEA analysis. We want a configuration that
unconditionally returns visible thinking blocks.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/probe_opus_thinking.py

Cost: ~$0.50 (one call per probe configuration, ~10 configs).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# A prompt that's genuinely non-trivial — a multi-step reasoning problem
# the adaptive scheduler should plausibly want to think on. Trivial
# prompts like "capital of France" allow adaptive to short-circuit.
HARD_PROMPT = (
    "A tank holds 100 liters of water. Water flows in at 5 L/min and "
    "out at 3 L/min, but every 4 minutes the outflow rate doubles for "
    "exactly 1 minute. After 20 minutes, how much water is in the "
    "tank? Show your reasoning."
)


def call(
    label: str,
    payload: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Returns (ok, summary, raw)."""
    api_key = os.environ["ANTHROPIC_API_KEY"]
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", err_body)
        except Exception:
            msg = err_body
        return False, f"HTTP {exc.code}: {msg[:200]}", None

    blocks = data.get("content", []) or []
    type_counts: Dict[str, int] = {}
    for b in blocks:
        if isinstance(b, dict):
            t = b.get("type", "<unknown>")
            type_counts[t] = type_counts.get(t, 0) + 1
    has_thinking = type_counts.get("thinking", 0) > 0
    has_reasoning = type_counts.get("reasoning", 0) > 0
    has_extended = type_counts.get("extended_thinking", 0) > 0
    sketch = ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items()))
    surfaced = (
        "YES"
        if (has_thinking or has_reasoning or has_extended)
        else "no"
    )
    return True, f"OK   blocks=[{sketch}] thinking-surfaced={surfaced}", data


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    base_messages = [{"role": "user", "content": HARD_PROMPT}]

    # Each entry is (label, payload, extra_headers).
    probes: List[Tuple[str, Dict[str, Any], Dict[str, str]]] = [
        (
            "A. adaptive + effort=high (current, hard prompt)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
            {},
        ),
        (
            "B. adaptive + effort=high + interleaved-thinking beta",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
            {"anthropic-beta": "interleaved-thinking-2025-05-14"},
        ),
        (
            "C. adaptive + effort=high + extended-thinking beta",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
            {"anthropic-beta": "extended-thinking-2025-01-29"},
        ),
        (
            "D. type=always (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "always"},
                "output_config": {"effort": "high"},
            },
            {},
        ),
        (
            "E. type=required (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "required"},
                "output_config": {"effort": "high"},
            },
            {},
        ),
        (
            "F. type=forced (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "forced"},
                "output_config": {"effort": "high"},
            },
            {},
        ),
        (
            "G. type=extended (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "extended"},
                "output_config": {"effort": "high"},
            },
            {},
        ),
        (
            "H. adaptive + output_config.show_thinking=true (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high", "show_thinking": True},
            },
            {},
        ),
        (
            "I. adaptive + output_config.include_thinking=true (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive"},
                "output_config": {
                    "effort": "high",
                    "include_thinking": True,
                },
            },
            {},
        ),
        (
            "J. adaptive + thinking.show=true (guess)",
            {
                "model": "claude-opus-4-7",
                "max_tokens": 4096,
                "messages": base_messages,
                "thinking": {"type": "adaptive", "show": True},
                "output_config": {"effort": "high"},
            },
            {},
        ),
    ]

    print(f"Probing {len(probes)} configs with HARD_PROMPT (cost ~$0.05/config)...\n")
    results: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    for label, payload, headers in probes:
        ok, summary, data = call(label, payload, headers)
        print(f"  {label}")
        print(f"    -> {summary}")
        results.append((label, summary, data))
        print()

    # If any config surfaced a thinking-type block, dump its full raw
    # response so we can confirm the shape and patch providers.py.
    print("=" * 70)
    print("WINNERS (configs that surfaced a thinking/reasoning block):")
    print("=" * 70)
    winners = [
        (label, data)
        for label, summary, data in results
        if data is not None
        and any(
            isinstance(b, dict)
            and b.get("type") in ("thinking", "reasoning", "extended_thinking")
            for b in data.get("content", [])
        )
    ]
    if not winners:
        print("None. Every probe either rejected or skipped exposing CoT.")
        print()
        print("Implication: opus-4.7's adaptive thinking is internal-only at")
        print("the current API version (2023-06-01). CoT-based VEA analysis")
        print("is not API-accessible for this model.")
        return 0
    for label, data in winners:
        print(f"\n--- {label} ---")
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
