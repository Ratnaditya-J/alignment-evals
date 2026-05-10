"""Provider-grade model clients for eval-awareness benchmarking.

These clients are intentionally thin adapters around common HTTP APIs. They are
safe to import without credentials and become active when environment variables
or explicit keys are supplied by benchmark operators.
"""

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .models import TranscriptInput


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient provider failures."""

    max_attempts: int = 3
    backoff_seconds: float = 0.5


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters captured in benchmark artifacts."""

    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Serialize generation settings."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
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
    raw: Dict[str, object] = field(default_factory=dict)


class OpenAICompatibleClient:
    """OpenAI-compatible chat-completions client.

    Works for OpenAI, OpenRouter, LiteLLM, vLLM OpenAI server, Together-style
    gateways, and other `/chat/completions` compatible endpoints.
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
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.retry = retry or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(
        cls,
        name: str,
        model: str,
        prefix: str = "EVAL_AWARENESS",
        default_base_url: str = "https://api.openai.com/v1",
    ) -> "OpenAICompatibleClient":
        """Create a client from `{prefix}_API_KEY` and optional `{prefix}_BASE_URL`."""
        return cls(
            name=name,
            base_url=os.environ.get(f"{prefix}_BASE_URL", default_base_url),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
        )

    def generate(self, transcript: TranscriptInput) -> str:
        """Generate with default deterministic settings."""
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        """Generate with explicit settings and capture raw metadata."""
        if importlib.util.find_spec("httpx") is None:
            raise RuntimeError("httpx is required for provider HTTP clients")
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": transcript.render()}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **config.extra,
        }
        if config.seed is not None:
            payload["seed"] = config.seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
        content = data["choices"][0]["message"]["content"]
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usage", {}),
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
        cls, name: str, model: str, prefix: str = "OPENROUTER"
    ) -> "OpenRouterClient":
        return cls(
            name=name,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
        )


class LiteLLMClient(OpenAICompatibleClient):
    """LiteLLM proxy client."""

    provider = "litellm"


class VLLMClient(OpenAICompatibleClient):
    """Local vLLM OpenAI-compatible server client."""

    provider = "vllm"

    @classmethod
    def local(
        cls, name: str, model: str, base_url: str = "http://localhost:8000/v1"
    ) -> "VLLMClient":
        return cls(name=name, base_url=base_url, api_key="EMPTY", model=model)


class OllamaClient:
    """Ollama local generation client."""

    provider = "ollama"

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str = "http://localhost:11434",
        retry: RetryConfig | None = None,
    ):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.last_response: RawProviderResponse | None = None

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        if importlib.util.find_spec("httpx") is None:
            raise RuntimeError("httpx is required for provider HTTP clients")
        payload = {
            "model": self.model,
            "prompt": transcript.render(),
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        if config.seed is not None:
            payload["options"]["seed"] = config.seed
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/api/generate",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=120.0,
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
            raw=data,
        )
        return content


class AnthropicClient:
    """Anthropic Messages API client."""

    provider = "anthropic"

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        retry: RetryConfig | None = None,
    ):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(
        cls, name: str, model: str, prefix: str = "ANTHROPIC"
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
        if importlib.util.find_spec("httpx") is None:
            raise RuntimeError("httpx is required for provider HTTP clients")
        payload = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": transcript.render()}],
            **config.extra,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/messages",
            headers=headers,
            payload=payload,
            timeout_seconds=120.0,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if isinstance(block, dict)
        )
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usage", {}),
            raw=data,
        )
        return content


class GeminiClient:
    """Gemini generateContent API client."""

    provider = "gemini"

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        retry: RetryConfig | None = None,
    ):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryConfig()
        self.last_response: RawProviderResponse | None = None

    @classmethod
    def from_env(cls, name: str, model: str, prefix: str = "GEMINI") -> "GeminiClient":
        return cls(
            name=name,
            api_key=os.environ[f"{prefix}_API_KEY"],
            model=model,
            base_url=os.environ.get(
                f"{prefix}_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
            ),
        )

    def generate(self, transcript: TranscriptInput) -> str:
        return self.generate_with_config(transcript, GenerationConfig())

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        if importlib.util.find_spec("httpx") is None:
            raise RuntimeError("httpx is required for provider HTTP clients")
        payload = {
            "contents": [{"parts": [{"text": transcript.render()}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
            },
            **config.extra,
        }
        started = time.perf_counter()
        response = _post_with_retries(
            url=f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=120.0,
            retry=self.retry,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data = response.json()
        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
        self.last_response = RawProviderResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
            usage=data.get("usageMetadata", {}),
            raw=data,
        )
        return content


def provider_manifest(clients: List[object]) -> List[Dict[str, object]]:
    """Return reproducibility metadata for a list of provider clients."""
    manifest = []
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


def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    timeout_seconds: float,
    retry: RetryConfig,
):
    import httpx

    last_error: Exception | None = None
    for attempt in range(retry.max_attempts):
        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=timeout_seconds
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retry.max_attempts:
                break
            time.sleep(retry.backoff_seconds * (2**attempt))
    raise RuntimeError(
        f"provider request failed after {retry.max_attempts} attempts: {last_error}"
    )


def raw_response_json(response: RawProviderResponse | None) -> str:
    """Serialize a raw provider response for logs."""
    if response is None:
        return "{}"
    return json.dumps(response.raw, sort_keys=True)
