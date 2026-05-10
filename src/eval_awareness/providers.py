"""Provider-grade model clients for eval-awareness benchmarking.

Each client captures full reproducibility metadata (provider, model, generation
config, raw response, latency, usage) so benchmark artifacts can be audited and
re-published. Optional native SDK clients (``OpenAI``, ``Anthropic``, ``Gemini``)
are loaded lazily so importing this module never requires the SDKs.

Supported providers:

* OpenAI compatible (``OpenAICompatibleClient`` and friends): OpenAI, OpenRouter,
  LiteLLM, vLLM, Together-style gateways, and any other ``/chat/completions``
  endpoint.
* OpenAI native (``OpenAIClient`` HTTP) and ``OpenAISDKClient``-style behaviour
  is implicit through ``OpenAILLMJudge`` for judging; the chat client uses HTTP.
* Anthropic Messages API (HTTP via ``AnthropicClient``, SDK via
  ``AnthropicSDKClient``).
* Google Gemini ``generateContent`` (HTTP via ``GeminiClient``, SDK via
  ``GeminiSDKClient``).
* Ollama local (``OllamaClient``).
* llama.cpp HTTP server (``LlamaCppClient``).
* Together native completions API (``TogetherClient``).
* Replicate (``ReplicateClient``).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .models import TranscriptInput

LOGGER = logging.getLogger(__name__)


class HTTPProviderError(RuntimeError):
    """Raised when a provider HTTP request ultimately fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient provider failures (HTTP 5xx, timeouts)."""

    max_attempts: int = 3
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    retry_on_status: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters captured in benchmark artifacts."""

    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    top_p: float | None = None
    stop: tuple[str, ...] | None = None
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Serialize generation settings."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "top_p": self.top_p,
            "stop": list(self.stop) if self.stop else None,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class RawProviderResponse:
    """Raw provider response metadata for reproducibility."""

    provider: str
    model: str
    content: str
    status_code: int
    latency_ms: float
    usage: Dict[str, object] = field(default_factory=dict)
    finish_reason: str | None = None
    request_id: str | None = None
    raw: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
        }


def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    timeout_seconds: float,
    retry: RetryConfig,
) -> "Any":
    if importlib.util.find_spec("httpx") is None:  # pragma: no cover - import guard
        raise RuntimeError("httpx is required for provider HTTP clients")
    import httpx

    last_error: Exception | None = None
    for attempt in range(retry.max_attempts):
        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=timeout_seconds
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            LOGGER.debug("HTTP transport error (attempt %d): %s", attempt + 1, exc)
        else:
            if response.status_code in retry.retry_on_status and (
                attempt + 1 < retry.max_attempts
            ):
                LOGGER.debug(
                    "Retryable HTTP %s on attempt %d", response.status_code, attempt + 1
                )
            elif 200 <= response.status_code < 300:
                return response
            else:
                raise HTTPProviderError(
                    f"HTTP {response.status_code} from {url}",
                    status_code=response.status_code,
                    body=response.text,
                )
        if attempt + 1 >= retry.max_attempts:
            break
        time.sleep(retry.backoff_seconds * (retry.backoff_multiplier**attempt))
    raise HTTPProviderError(
        f"provider request failed after {retry.max_attempts} attempts: {last_error}"
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return {}


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP clients
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """OpenAI-compatible ``/chat/completions`` HTTP client.

    Works for OpenAI, OpenRouter, LiteLLM, vLLM OpenAI server, Together-style
    gateways, and any other endpoint that follows the OpenAI request schema.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        retry: RetryConfig | None = None,
        timeout_seconds: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.extra_headers = dict(extra_headers or {})
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "EVAL_AWARENESS",
        default_base_url: str = "https://api.openai.com/v1",
    ) -> "OpenAICompatibleClient":
        return cls(
            name=name,
            base_url=os.environ.get(f"{prefix}_BASE_URL", default_base_url),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
        )

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def _build_messages(self, transcript: TranscriptInput) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if transcript.system_prompt:
            messages.append({"role": "system", "content": transcript.system_prompt})
        if transcript.developer_prompt:
            messages.append({"role": "system", "content": transcript.developer_prompt})
        for entry in transcript.conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        if transcript.user_prompt:
            messages.append({"role": "user", "content": transcript.user_prompt})
        if not messages:
            messages.append({"role": "user", "content": transcript.render()})
        return messages

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": self._build_messages(transcript),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop"] = list(config.stop)
        if config.seed is not None:
            payload["seed"] = config.seed
        payload.update(config.extra)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/chat/completions",
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usage", {}) or {},
            finish_reason=choice.get("finish_reason"),
            request_id=response.headers.get("x-request-id"),
            raw=data,
        )
        return content


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI API client using the chat-completions compatible path."""

    provider = "openai"

    @classmethod
    def from_env(cls, name: str, model: str, prefix: str = "OPENAI") -> "OpenAIClient":
        return cls(
            name=name,
            base_url=os.environ.get(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
        )


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter client using its OpenAI-compatible API."""

    provider = "openrouter"

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "OPENROUTER",
    ) -> "OpenRouterClient":
        return cls(
            name=name,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
            extra_headers={
                "HTTP-Referer": os.environ.get(
                    f"{prefix}_REFERER", "https://github.com/Ratnaditya-J/IsItBenchmark"
                ),
                "X-Title": os.environ.get(f"{prefix}_TITLE", "IsItBenchmark"),
            },
        )


class LiteLLMClient(OpenAICompatibleClient):
    """LiteLLM proxy client (OpenAI-compatible)."""

    provider = "litellm"


class VLLMClient(OpenAICompatibleClient):
    """Local vLLM OpenAI-compatible server client."""

    provider = "vllm"

    @classmethod
    def local(
        cls, name: str, model: str, base_url: str = "http://localhost:8000/v1"
    ) -> "VLLMClient":
        return cls(name=name, base_url=base_url, api_key="EMPTY", model=model)


# ---------------------------------------------------------------------------
# Ollama local
# ---------------------------------------------------------------------------


class OllamaClient:
    """Ollama local generation client."""

    provider = "ollama"

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str = "http://localhost:11434",
        retry: RetryConfig | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model,
            "prompt": transcript.render(),
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        if config.top_p is not None:
            payload["options"]["top_p"] = config.top_p  # type: ignore[index]
        if config.seed is not None:
            payload["options"]["seed"] = config.seed  # type: ignore[index]
        if config.stop:
            payload["options"]["stop"] = list(config.stop)  # type: ignore[index]
        payload.update(config.extra)
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/api/generate",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        content = data.get("response", "")
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "total_duration_ns": data.get("total_duration"),
            },
            finish_reason=data.get("done_reason"),
            raw=data,
        )
        return content


# ---------------------------------------------------------------------------
# llama.cpp HTTP server
# ---------------------------------------------------------------------------


class LlamaCppClient:
    """Client for the llama.cpp ``server`` REST API (``/completion`` endpoint)."""

    provider = "llama_cpp"

    def __init__(
        self,
        name: str,
        model: str = "llama.cpp",
        base_url: str = "http://localhost:8080",
        retry: RetryConfig | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        payload: Dict[str, object] = {
            "prompt": transcript.render(),
            "n_predict": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.stop:
            payload["stop"] = list(config.stop)
        payload.update(config.extra)
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/completion",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        content = data.get("content", "")
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage={
                "tokens_predicted": data.get("tokens_predicted"),
                "tokens_evaluated": data.get("tokens_evaluated"),
            },
            finish_reason=data.get("stopped_eos")
            and "stop"
            or data.get("stopping_word"),
            raw=data,
        )
        return content


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Anthropic Messages API HTTP client (no SDK dependency)."""

    provider = "anthropic"

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        retry: RetryConfig | None = None,
        timeout_seconds: float = 120.0,
        anthropic_version: str = "2023-06-01",
    ):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.anthropic_version = anthropic_version
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "ANTHROPIC",
    ) -> "AnthropicClient":
        return cls(
            name=name,
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL", "https://api.anthropic.com/v1"
            ),
        )

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        messages: List[Dict[str, str]] = []
        for entry in transcript.conversation_history:
            role = entry.get("role", "user")
            if role == "system":
                continue
            content = entry.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        if transcript.user_prompt:
            messages.append({"role": "user", "content": transcript.user_prompt})
        if not messages:
            messages.append({"role": "user", "content": transcript.render()})
        payload: Dict[str, object] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": messages,
        }
        if transcript.system_prompt:
            payload["system"] = transcript.system_prompt
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop_sequences"] = list(config.stop)
        payload.update(config.extra)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/messages",
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        content_blocks = data.get("content", []) or []
        content = "".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usage", {}) or {},
            finish_reason=data.get("stop_reason"),
            request_id=response.headers.get("request-id"),
            raw=data,
        )
        return content


class AnthropicSDKClient:
    """Anthropic Messages client backed by the official ``anthropic`` SDK."""

    provider = "anthropic_sdk"

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
    ):
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - missing optional dep
            raise RuntimeError(
                "The 'anthropic' SDK is required for AnthropicSDKClient."
            ) from exc
        self.name = name
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "AnthropicSDKClient requires an API key (pass api_key= or set ANTHROPIC_API_KEY)."
            )
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self._client = Anthropic(api_key=self.api_key, timeout=timeout_seconds)
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        messages: List[Dict[str, str]] = []
        for entry in transcript.conversation_history:
            role = entry.get("role", "user")
            if role == "system":
                continue
            content = entry.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        if transcript.user_prompt:
            messages.append({"role": "user", "content": transcript.user_prompt})
        if not messages:
            messages.append({"role": "user", "content": transcript.render()})
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": messages,
        }
        if transcript.system_prompt:
            kwargs["system"] = transcript.system_prompt
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop:
            kwargs["stop_sequences"] = list(config.stop)
        kwargs.update(config.extra)

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            try:
                response = self._client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                time.sleep(0.5 * (2**attempt))
                continue
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            content_blocks = getattr(response, "content", []) or []
            content = "".join(
                getattr(block, "text", "")
                for block in content_blocks
                if getattr(block, "type", None) == "text"
            )
            self.last_response = RawProviderResponse(
                provider=self.provider,
                model=self.model,
                content=content,
                status_code=200,
                latency_ms=latency_ms,
                usage=_as_dict(getattr(response, "usage", None)),
                finish_reason=getattr(response, "stop_reason", None),
                request_id=getattr(response, "id", None),
                raw=_as_dict(response),
            )
            return content
        raise HTTPProviderError(
            f"AnthropicSDKClient failed after {self.max_attempts} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiClient:
    """Gemini ``generateContent`` HTTP client."""

    provider = "gemini"

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        retry: RetryConfig | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "GEMINI",
    ) -> "GeminiClient":
        return cls(
            name=name,
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
        )

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        contents: List[Dict[str, Any]] = []
        for entry in transcript.conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if not content:
                continue
            api_role = "user" if role in {"user", "system"} else "model"
            contents.append({"role": api_role, "parts": [{"text": content}]})
        primary = transcript.user_prompt or transcript.render()
        if primary:
            contents.append({"role": "user", "parts": [{"text": primary}]})
        generation_config: Dict[str, Any] = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        }
        if config.top_p is not None:
            generation_config["topP"] = config.top_p
        if config.stop:
            generation_config["stopSequences"] = list(config.stop)
        payload: Dict[str, object] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if transcript.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": transcript.system_prompt}]
            }
        payload.update(config.extra)
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        candidates = data.get("candidates", []) or []
        content = ""
        finish_reason = None
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
            finish_reason = candidates[0].get("finishReason")
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usageMetadata", {}) or {},
            finish_reason=finish_reason,
            raw=data,
        )
        return content


class GeminiSDKClient:
    """Gemini client backed by the ``google-genai`` SDK when available."""

    provider = "gemini_sdk"

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "The 'google-genai' SDK is required for GeminiSDKClient."
            ) from exc
        self.name = name
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "GeminiSDKClient requires an API key (pass api_key= or set GOOGLE_API_KEY)."
            )
        self.timeout_seconds = timeout_seconds
        self._client = genai.Client(api_key=self.api_key)
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        prompt = transcript.user_prompt or transcript.render()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "contents": prompt,
            "config": {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            },
        }
        if transcript.system_prompt:
            kwargs["config"]["system_instruction"] = transcript.system_prompt
        if config.top_p is not None:
            kwargs["config"]["top_p"] = config.top_p
        if config.stop:
            kwargs["config"]["stop_sequences"] = list(config.stop)
        started = time.perf_counter()
        response = self._client.models.generate_content(**kwargs)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        content = getattr(response, "text", "") or ""
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=200,
            latency_ms=latency_ms,
            usage=_as_dict(getattr(response, "usage_metadata", None)),
            finish_reason=(
                getattr(
                    getattr(response, "candidates", [None])[0], "finish_reason", None
                )
                if getattr(response, "candidates", None)
                else None
            ),
            raw=_as_dict(response),
        )
        return content


# ---------------------------------------------------------------------------
# Together AI native (uses /v1/chat/completions, but custom endpoint + headers)
# ---------------------------------------------------------------------------


class TogetherClient(OpenAICompatibleClient):
    """Together AI client (OpenAI-compatible endpoint with provider tagging)."""

    provider = "together"

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "TOGETHER",
    ) -> "TogetherClient":
        return cls(
            name=name,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL", "https://api.together.xyz/v1"
            ),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
        )


# ---------------------------------------------------------------------------
# Replicate (different request shape, async predictions API)
# ---------------------------------------------------------------------------


class ReplicateClient:
    """Replicate predictions client.

    ``model`` should be a fully-qualified ``owner/name:version`` reference.
    The client polls until the prediction completes or ``timeout_seconds`` elapses.
    """

    provider = "replicate"

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.replicate.com/v1",
        retry: RetryConfig | None = None,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise RuntimeError(
                "ReplicateClient requires an API token (pass api_key= or set REPLICATE_API_TOKEN)."
            )
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        if importlib.util.find_spec("httpx") is None:  # pragma: no cover
            raise RuntimeError("httpx is required for ReplicateClient")
        import httpx

        version = self.model.split(":", 1)[1] if ":" in self.model else None
        payload: Dict[str, Any] = {
            "input": {
                "prompt": transcript.user_prompt or transcript.render(),
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                **{k: v for k, v in config.extra.items() if isinstance(k, str)},
            }
        }
        if transcript.system_prompt:
            payload["input"]["system_prompt"] = transcript.system_prompt
        if version:
            payload["version"] = version
            url = f"{self.base_url}/predictions"
        else:
            url = f"{self.base_url}/models/{self.model}/predictions"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        create_response = _post_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
        )
        prediction = create_response.json()
        prediction_id = prediction.get("id")
        if not prediction_id:
            raise HTTPProviderError("Replicate did not return a prediction id")
        get_url = f"{self.base_url}/predictions/{prediction_id}"
        deadline = time.time() + self.timeout_seconds
        while True:
            poll = httpx.get(get_url, headers=headers, timeout=self.timeout_seconds)
            poll.raise_for_status()
            prediction = poll.json()
            status = prediction.get("status", "")
            if status in {"succeeded", "failed", "canceled"}:
                break
            if time.time() > deadline:
                raise HTTPProviderError(
                    f"Replicate prediction {prediction_id} timed out"
                )
            time.sleep(self.poll_interval_seconds)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if prediction.get("status") != "succeeded":
            raise HTTPProviderError(
                f"Replicate prediction failed: {prediction.get('error')}"
            )
        output = prediction.get("output", "")
        if isinstance(output, list):
            content = "".join(str(piece) for piece in output)
        else:
            content = str(output)
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=create_response.status_code,
            latency_ms=latency_ms,
            usage=prediction.get("metrics", {}) or {},
            finish_reason=prediction.get("status"),
            request_id=prediction_id,
            raw=prediction,
        )
        return content


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------


def provider_manifest(clients: List[object]) -> List[Dict[str, object]]:
    """Return reproducibility metadata for a list of provider clients."""
    manifest: List[Dict[str, object]] = []
    for client in clients:
        manifest.append(
            {
                "name": getattr(client, "name", "unknown"),
                "provider": getattr(client, "provider", client.__class__.__name__),
                "model": getattr(
                    client, "model", getattr(client, "model_id", "unknown")
                ),
                "base_url": getattr(client, "base_url", ""),
            }
        )
    return manifest


def raw_response_json(response: RawProviderResponse | None) -> str:
    """Serialize a raw provider response for logs."""
    if response is None:
        return "{}"
    return json.dumps(response.to_dict(), sort_keys=True)
