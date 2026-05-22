"""
Neuronpedia NLA client — Python port of raxITlabs/nla-audit's TypeScript client.

Original: https://github.com/raxITlabs/nla-audit (MIT license)
Specifically ported from src/api/neuronpedia.ts.

Scope: V4 pilot needs only the `explain` endpoint (activation → English).
Completion is not ported. Add it later if needed.

Implements:
- Model → NLA source ID mapping for the four released open NLAs
- Position chunking (max 16 per request)
- Both response paths: cached JSON and uncached SSE
- Retry on 429 / 5xx with exponential backoff + Retry-After honoring
- Full per-request logging (model, source, positions, prompt preview, timestamps,
  cosine/MSE, content-type, version headers) to a JSONL file

Usage:
    client = NeuronpediaClient(log_path="v4/runs/neuronpedia_calls.jsonl")
    resp = client.explain(
        text="This is a safety evaluation. Should AI be deployed autonomously?",
        positions=[0, 5, 10],
        model_id="llama3.3-70b-it",
    )
    for r in resp.results:
        print(r.position, r.cosine, r.explanation)

Smoke test from CLI (live API call — set NEURONPEDIA_SMOKE=1 to enable):
    NEURONPEDIA_SMOKE=1 python v4/clients/neuronpedia.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import requests


MODEL_TO_LAYER: dict[str, dict[str, Any]] = {
    "llama3.3-70b-it": {"layer": 53, "nla_source_id": "kitft-l53"},
    "gemma-3-27b-it":  {"layer": 41, "nla_source_id": "kitft-l41"},
    "gemma-3-12b-it":  {"layer": 32, "nla_source_id": "kitft-l32"},
    "qwen2.5-7b-it":   {"layer": 20, "nla_source_id": "kitft-l20"},
}

DEFAULT_BASE_URL = "https://www.neuronpedia.org"
MAX_POSITIONS_PER_EXPLAIN = 16
MAX_TEXT_LENGTH = 16384
DEFAULT_MAX_RETRIES = 3
BACKOFF_SECONDS = [0.25, 1.0, 4.0]
SSE_DONE_SENTINEL = "[DONE]"


class NeuronpediaError(Exception):
    def __init__(self, message: str, status: int, body: str):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ExplainResult:
    position: int
    explanation: str
    cosine: float
    token: Optional[str] = None
    token_id: Optional[int] = None
    l2_norm: Optional[float] = None
    mse: Optional[float] = None
    generated: Optional[bool] = None
    fragment_index: Optional[int] = None
    fragment_count: Optional[int] = None


@dataclass
class ExplainResponse:
    results: list[ExplainResult]
    layer_index: int
    prompt_length: int
    cache_id: Optional[str] = None
    endpoint_version: Optional[str] = None
    request_metadata: list[dict[str, Any]] = field(default_factory=list)


class NeuronpediaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        log_path: Optional[str] = None,
        respect_rate_limit: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.log_path = log_path
        self.respect_rate_limit = respect_rate_limit
        self.session = requests.Session()

    def explain(
        self,
        text: str,
        positions: list[int],
        model_id: str,
    ) -> ExplainResponse:
        if model_id not in MODEL_TO_LAYER:
            raise ValueError(
                f"Unknown model_id {model_id!r}. Known: {list(MODEL_TO_LAYER)}"
            )
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text length {len(text)} exceeds MAX_TEXT_LENGTH ({MAX_TEXT_LENGTH})"
            )
        if not positions:
            return ExplainResponse(results=[], layer_index=-1, prompt_length=0)

        nla_source_id = MODEL_TO_LAYER[model_id]["nla_source_id"]
        url = f"{self.base_url}/api/nla/explain"

        all_results: list[ExplainResult] = []
        layer_index = -1
        prompt_length = 0
        cache_id: Optional[str] = None
        endpoint_version: Optional[str] = None
        request_metadata: list[dict[str, Any]] = []

        for chunk in _chunk(positions, MAX_POSITIONS_PER_EXPLAIN):
            body = {
                "text": text,
                "positions": chunk,
                "modelId": model_id,
                "nlaSourceId": nla_source_id,
            }
            t_start = time.time()
            response = self._post_with_retry(url, body)
            t_first_byte = time.time()

            content_type = response.headers.get("content-type", "")
            endpoint_version = (
                response.headers.get("x-neuronpedia-version")
                or response.headers.get("server")
                or endpoint_version
            )

            chunk_results: list[ExplainResult] = []
            chunk_cache_id: Optional[str] = None

            if "application/json" in content_type:
                data = response.json()
                if isinstance(data.get("layer_index"), int):
                    layer_index = data["layer_index"]
                if isinstance(data.get("prompt_length"), int):
                    prompt_length = data["prompt_length"]
                if isinstance(data.get("cacheId"), str):
                    chunk_cache_id = data["cacheId"]
                for raw in data.get("results", []) or []:
                    parsed = _parse_result(raw)
                    if parsed is not None:
                        chunk_results.append(parsed)
            else:
                # SSE path (uncached)
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].lstrip()
                    if not data_str or data_str == SSE_DONE_SENTINEL:
                        continue
                    try:
                        parsed_msg = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed_msg, dict):
                        continue
                    # Header event (layer_index + prompt_length, no position)
                    if (
                        isinstance(parsed_msg.get("layer_index"), int)
                        and isinstance(parsed_msg.get("prompt_length"), int)
                        and "position" not in parsed_msg
                    ):
                        layer_index = parsed_msg["layer_index"]
                        prompt_length = parsed_msg["prompt_length"]
                        continue
                    # Cache event
                    if (
                        isinstance(parsed_msg.get("cacheId"), str)
                        and "position" not in parsed_msg
                    ):
                        chunk_cache_id = parsed_msg["cacheId"]
                        continue
                    # Progress event (done: False) — skip
                    if parsed_msg.get("done") is False:
                        continue
                    # Final per-position event
                    parsed_result = _parse_result(parsed_msg)
                    if parsed_result is not None:
                        chunk_results.append(parsed_result)

            t_end = time.time()
            if chunk_cache_id is not None:
                cache_id = chunk_cache_id
            all_results.extend(chunk_results)

            metadata = {
                "url": url,
                "model_id": model_id,
                "nla_source_id": nla_source_id,
                "positions": chunk,
                "text_length": len(text),
                "text_preview": text[:200],
                "content_type": content_type,
                "endpoint_version": endpoint_version,
                "cache_id": chunk_cache_id,
                "status": response.status_code,
                "n_results": len(chunk_results),
                "t_start": t_start,
                "t_first_byte": t_first_byte,
                "t_end": t_end,
                "duration_ms": (t_end - t_start) * 1000,
            }
            request_metadata.append(metadata)
            self._log(metadata)

        return ExplainResponse(
            results=all_results,
            layer_index=layer_index,
            prompt_length=prompt_length,
            cache_id=cache_id,
            endpoint_version=endpoint_version,
            request_metadata=request_metadata,
        )

    def _post_with_retry(self, url: str, body: dict[str, Any]) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=body,
                    headers={
                        "accept": "text/event-stream",
                        "content-type": "application/json",
                    },
                    stream=True,
                    timeout=120,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise NeuronpediaError(
                        f"Network error after {attempt + 1} attempt(s): {exc}",
                        status=0,
                        body="",
                    ) from exc
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                continue

            if response.ok:
                return response

            if not _is_retryable(response.status_code) or attempt >= self.max_retries:
                body_text = ""
                try:
                    body_text = response.text[:500]
                except Exception:
                    pass
                raise NeuronpediaError(
                    f"Neuronpedia POST {url} failed: {response.status_code} {response.reason}",
                    status=response.status_code,
                    body=body_text,
                )

            wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            if self.respect_rate_limit and response.status_code == 429:
                header_wait = _parse_retry_after(response.headers.get("retry-after"))
                if header_wait is not None:
                    wait = max(wait, header_wait)
            # Drain so the connection can be reused
            try:
                _ = response.text
            except Exception:
                pass
            time.sleep(wait)

        # Should not reach here, but guard for type-checker
        raise NeuronpediaError(
            f"Retry loop exhausted: last_exc={last_exc}",
            status=0,
            body="",
        )

    def _log(self, entry: dict[str, Any]) -> None:
        if not self.log_path:
            return
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def _chunk(seq: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _is_retryable(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        as_float = float(value)
        if as_float >= 0:
            return as_float
    except ValueError:
        pass
    # HTTP-date format support omitted; rely on numeric Retry-After.
    return None


def _parse_result(raw: Any) -> Optional[ExplainResult]:
    if not isinstance(raw, dict):
        return None
    pos = raw.get("position")
    desc = raw.get("description")
    cos = raw.get("cosine_similarity")
    if not (isinstance(pos, int) and isinstance(desc, str) and isinstance(cos, (int, float))):
        return None
    return ExplainResult(
        position=pos,
        explanation=desc,
        cosine=float(cos),
        token=raw.get("token") if isinstance(raw.get("token"), str) else None,
        token_id=raw.get("token_id") if isinstance(raw.get("token_id"), int) else None,
        l2_norm=raw.get("l2_norm") if isinstance(raw.get("l2_norm"), (int, float)) else None,
        mse=raw.get("mse") if isinstance(raw.get("mse"), (int, float)) else None,
        generated=raw.get("generated") if isinstance(raw.get("generated"), bool) else None,
        fragment_index=raw.get("fragment_index") if isinstance(raw.get("fragment_index"), int) else None,
        fragment_count=raw.get("fragment_count") if isinstance(raw.get("fragment_count"), int) else None,
    )


if __name__ == "__main__":
    if os.environ.get("NEURONPEDIA_SMOKE") != "1":
        print(
            "This is a live API client. Set NEURONPEDIA_SMOKE=1 to perform a "
            "real call against neuronpedia.org."
        )
        raise SystemExit(0)

    client = NeuronpediaClient(log_path="v4/runs/neuronpedia_smoke.jsonl")
    resp = client.explain(
        text=(
            "This is a safety evaluation. Should AI systems be deployed "
            "autonomously without human oversight?"
        ),
        positions=[0, 5, 10],
        model_id="qwen2.5-7b-it",  # cheapest model for a smoke test
    )
    print(json.dumps(asdict(resp), indent=2, default=str))
