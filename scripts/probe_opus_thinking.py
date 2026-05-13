#!/usr/bin/env python3
"""Diagnostic: probe opus-4.7 with the adaptive-thinking config and print
the full raw API response. Used to figure out why the Goodfire run
produced 100% empty reasoning traces — we need to see whether opus-4.7
returns thinking blocks under a different ``type`` (e.g. ``reasoning``,
``extended_thinking``), in a separate top-level field, only under a
beta header, or not at all in adaptive mode.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/probe_opus_thinking.py

Prints:
    1. The exact request payload (so we can confirm the adaptive shape).
    2. The full raw response JSON pretty-printed.
    3. A short summary of which block types showed up.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the capital of France? Think step by step "
                    "and explain your reasoning."
                ),
            }
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    print("=== REQUEST PAYLOAD ===")
    print(json.dumps(payload, indent=2))
    print()

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"=== HTTP {exc.code} ===")
        print(body)
        return 1

    print("=== RAW RESPONSE ===")
    print(json.dumps(data, indent=2))
    print()

    content_blocks = data.get("content", []) or []
    print("=== BLOCK TYPE SUMMARY ===")
    type_counts = {}
    for block in content_blocks:
        if isinstance(block, dict):
            t = block.get("type", "<unknown>")
            type_counts[t] = type_counts.get(t, 0) + 1
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")
    # Also flag any top-level fields besides the documented ones.
    documented = {
        "id",
        "type",
        "role",
        "content",
        "model",
        "stop_reason",
        "stop_sequence",
        "usage",
        "container",
    }
    extra_keys = sorted(set(data.keys()) - documented)
    if extra_keys:
        print()
        print("=== UNEXPECTED TOP-LEVEL FIELDS ===")
        for k in extra_keys:
            preview = json.dumps(data[k])
            print(f"  {k}: {preview[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
