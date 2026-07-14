"""Model providers — the thin layer that actually calls an LLM.

A ``ModelClient`` knows how to turn ``(system, prompt)`` into text and how long
it took. Targets and the LLM-judge classifier depend only on this interface, so
swapping Ollama for Anthropic (or adding OpenAI, vLLM, etc.) is a one-file
change that the rest of the pipeline never notices.

``OllamaClient`` is the default and needs no API key. ``AnthropicClient`` imports
the ``anthropic`` SDK *lazily* and is only ever constructed when the Anthropic
path is explicitly enabled, so the core pipeline never depends on it.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Settings


class GenerationError(Exception):
    """A model call failed for a reason worth surfacing to the user."""


class OllamaNotAvailableError(GenerationError):
    """Ollama is not reachable, or the requested model has not been pulled."""


@dataclass
class Generation:
    """The result of one model call."""

    text: str
    latency_s: float
    model: str


class ModelClient(ABC):
    """Common interface for every model backend."""

    provider: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict], *, system: str | None = None) -> Generation:
        """Send a full conversation (``[{"role", "content"}, …]``) and return the reply.

        This is the primitive every backend implements; it carries multi-turn
        attacks. ``generate`` is the single-user-turn convenience wrapper.
        """

    def generate(self, prompt: str, *, system: str | None = None) -> Generation:
        """Send one user turn (with optional system prompt) and return the reply."""
        return self.chat([{"role": "user", "content": prompt}], system=system)

    def health_check(self) -> None:
        """Fail fast with a helpful message if this backend is unusable.

        Default: assume healthy. Backends with a reachable dependency (Ollama)
        override this.
        """
        return None


# ---------------------------------------------------------------------------
# Ollama — local, free, the default for every target and the judge.
# ---------------------------------------------------------------------------
_CONNECTION_HINTS = (
    "connection",
    "connect",
    "refused",
    "max retries",
    "failed to establish",
    "timed out",
    "no route to host",
)


class OllamaClient(ModelClient):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        temperature: float = 0.0,
        num_predict: int = 512,
    ):
        super().__init__(model)
        self.host = host
        self.temperature = temperature
        self.num_predict = num_predict
        self._client = None  # created lazily so importing this module is cheap

    def _get_client(self):
        if self._client is None:
            import ollama

            # A None host lets the library use OLLAMA_HOST / localhost defaults.
            self._client = ollama.Client(host=self.host) if self.host else ollama
        return self._client

    def chat(self, messages: list[dict], *, system: str | None = None) -> Generation:
        full = ([{"role": "system", "content": system}] if system else []) + list(messages)

        client = self._get_client()
        t0 = time.perf_counter()
        try:
            resp = client.chat(
                model=self.model,
                messages=full,
                options={"temperature": self.temperature, "num_predict": self.num_predict},
            )
        except Exception as exc:  # noqa: BLE001 - classify then re-raise with guidance
            raise self._translate_error(exc) from exc
        latency = time.perf_counter() - t0

        # ollama>=0.4 returns an object; older versions return a dict. Support both.
        message = getattr(resp, "message", None)
        if message is not None:
            text = getattr(message, "content", "") or ""
        else:
            text = resp.get("message", {}).get("content", "")  # type: ignore[union-attr]
        return Generation(text=text, latency_s=latency, model=self.model)

    def health_check(self) -> None:
        client = self._get_client()
        try:
            resp = client.list()
        except Exception as exc:  # noqa: BLE001
            raise OllamaNotAvailableError(self._daemon_down_message(exc)) from exc

        names = _extract_model_names(resp)
        # Accept an exact match, or a bare name matching the family (llama3.1
        # matching llama3.1:8b) to be forgiving about the :tag suffix.
        base = self.model.split(":")[0]
        if self.model not in names and not any(n.split(":")[0] == base for n in names):
            raise OllamaNotAvailableError(self._model_missing_message(names))

    def _translate_error(self, exc: Exception) -> GenerationError:
        text = str(exc).lower()
        if any(hint in text for hint in _CONNECTION_HINTS):
            return OllamaNotAvailableError(self._daemon_down_message(exc))
        if "not found" in text or "no such model" in text or "try pulling" in text:
            return OllamaNotAvailableError(self._model_missing_message(None))
        return GenerationError(f"Ollama call failed: {exc}")

    def _daemon_down_message(self, exc: Exception) -> str:
        return (
            "Ollama does not appear to be running.\n\n"
            "  1. Install Ollama:  https://ollama.com/download  (or: brew install ollama)\n"
            "  2. Start it:        run `ollama serve` (the desktop app starts it too)\n"
            f"  3. Pull a model:    ollama pull {self.model}\n\n"
            f"Underlying error: {exc}"
        )

    def _model_missing_message(self, available: list[str] | None) -> str:
        have = f"\n  You currently have: {', '.join(available)}" if available else ""
        return (
            f"Ollama is running but the model '{self.model}' is not available.\n\n"
            f"  Pull it with:  ollama pull {self.model}\n"
            f"  Or choose another model with --model NAME.{have}"
        )


def _extract_model_names(resp: object) -> list[str]:
    """Pull model names out of an ollama ``list()`` response (object or dict)."""
    raw = getattr(resp, "models", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("models", [])
    names: list[str] = []
    for m in raw or []:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Anthropic — OPTIONAL. Off by default. Makes real, paid API calls.
# The SDK is imported lazily so the tool never depends on the `anthropic`
# package (or an API key) unless this client is actually constructed.
# See README -> "Optional: add Claude as a comparison target / judge".
# ---------------------------------------------------------------------------
class AnthropicClient(ModelClient):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        super().__init__(model)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # lazy: only imported when Anthropic is enabled
            except ImportError as exc:
                raise GenerationError(
                    "The 'anthropic' package is not installed. Enable the optional "
                    "Anthropic path with:  pip install anthropic"
                ) from exc
            if not self.api_key:
                raise GenerationError(
                    "ANTHROPIC_API_KEY is not set — cannot use the Anthropic target/judge."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict], *, system: str | None = None) -> Generation:
        client = self._ensure_client()
        t0 = time.perf_counter()
        try:
            # Note: current-generation Claude models (Sonnet 5, Opus 4.x, …) reject
            # `temperature` with a 400 ("temperature is deprecated for this model"),
            # so we don't send it. Behaviour is steered by the prompt instead; temp=0
            # never guaranteed determinism anyway. self.temperature is kept only for
            # interface parity with the Ollama client.
            kwargs: dict = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": list(messages),
            }
            if system:
                kwargs["system"] = system
            msg = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Anthropic call failed: {exc}") from exc
        latency = time.perf_counter() - t0
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
        return Generation(text=text, latency_s=latency, model=self.model)


# ---------------------------------------------------------------------------
# Mock — deterministic, offline. Powers integration tests and CI without Ollama.
# ---------------------------------------------------------------------------
class MockClient(ModelClient):
    """A scripted client that returns canned responses with no network.

    ``script`` is either a callable ``(system, prompt) -> str`` or a dict mapping a
    substring to the response returned when that substring appears in the prompt.
    Anything unmatched returns ``default``. Latency is reported as 0.
    """

    provider = "mock"

    def __init__(
        self,
        model: str = "mock",
        *,
        script=None,
        default: str = "I'm sorry, but I can't help with that.",
    ):
        super().__init__(model)
        self.script = script
        self.default = default

    def chat(self, messages: list[dict], *, system: str | None = None) -> Generation:
        # Match the script against the concatenated user turns (covers single- and
        # multi-turn), so an attack whose payload lands on the final turn still hits.
        prompt = "\n".join(m["content"] for m in messages if m.get("role") == "user")
        text: str | None = None
        if callable(self.script):
            text = self.script(system, prompt)
        elif isinstance(self.script, dict):
            for needle, response in self.script.items():
                if needle in prompt:
                    text = response
                    break
        return Generation(text=text if text is not None else self.default, latency_s=0.0, model=self.model)


def build_ollama_client(settings: Settings, *, model: str | None = None) -> OllamaClient:
    """Construct an Ollama client from settings (optionally overriding the model)."""
    return OllamaClient(
        model=model or settings.ollama_model,
        host=settings.ollama_host,
        temperature=settings.temperature,
        num_predict=settings.num_predict,
    )
