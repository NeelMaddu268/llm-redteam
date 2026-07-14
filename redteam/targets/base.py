"""The ``TargetSystem`` abstraction every attackable system implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from ..defenses import Defense, NoDefense
from ..providers import GenerationError, ModelClient
from ..schemas import Payload, RunResult


class TargetSystem(ABC):
    """A system-under-test.

    Subclasses decide only *how a payload becomes a prompt* (``build_prompt``);
    the base class owns applying the defense, the call, timing, metadata capture,
    and error handling so every target produces a uniform ``RunResult``.

    ``kind`` is the base identity ("ollama-direct") independent of the model or
    defense suffix in ``name`` — the CLI ``--targets`` filter matches on it.
    """

    def __init__(
        self,
        client: ModelClient,
        *,
        name: str,
        system_prompt: str | None = None,
        defense: Defense | None = None,
        kind: str | None = None,
    ):
        self.client = client
        self.name = name
        self.system_prompt = system_prompt
        self.defense = defense or NoDefense()
        self.kind = kind or name

    @property
    def provider(self) -> str:
        return self.client.provider

    @property
    def model(self) -> str:
        return self.client.model

    @abstractmethod
    def build_prompt(self, payload: Payload) -> str:
        """Turn a payload into the user-turn prompt actually sent to the model."""

    def health_check(self) -> None:
        """Verify the underlying provider is usable (delegates to the client)."""
        self.client.health_check()

    def run(self, payload: Payload, *, trial: int = 0) -> RunResult:
        """Run one payload and capture the full interaction.

        For a **multi-turn** payload (``setup_turns`` non-empty), the priming turns
        are each sent and answered by the model to build real context, then the main
        attack turn is sent; the *final* response is what gets classified. The
        defense transforms the (system, attack-turn) pair, and the recorded
        ``prompt_sent`` is the full transcript so the log shows what the model saw.
        Provider failures are caught into ``RunResult.error`` so one bad call never
        aborts a sweep.
        """
        built = self.build_prompt(payload)
        system, attack_turn = self.defense.defend(self.system_prompt, built)
        timestamp = datetime.now(UTC).isoformat()
        base = dict(
            timestamp=timestamp,
            target=self.name,
            provider=self.provider,
            model=self.model,
            payload_id=payload.id,
            category=payload.category,
            defense=self.defense.name,
            trial=trial,
            system_prompt=system,
        )

        messages: list[dict] = []
        latency = 0.0
        try:
            # Priming turns first (each answered, so the attack builds on real context).
            for turn in payload.setup_turns:
                messages.append({"role": "user", "content": turn})
                gen = self.client.chat(messages, system=system)
                latency += gen.latency_s
                messages.append({"role": "assistant", "content": gen.text})
            # The main attack turn — its response is the one we classify.
            messages.append({"role": "user", "content": attack_turn})
            gen = self.client.chat(messages, system=system)
            latency += gen.latency_s
        except GenerationError as exc:
            sent = _transcript(messages) if payload.setup_turns else attack_turn
            return RunResult(**base, prompt_sent=sent, response="", latency_s=0.0, error=str(exc))

        prompt_sent = _transcript(messages) if payload.setup_turns else attack_turn
        return RunResult(**base, prompt_sent=prompt_sent, response=gen.text, latency_s=latency)


def _transcript(messages: list[dict]) -> str:
    """Render a conversation as a readable transcript for the results log."""
    return "\n\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
