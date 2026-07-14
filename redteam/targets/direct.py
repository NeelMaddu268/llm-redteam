"""Direct target: the payload is sent straight to the model as the user turn.

This is the baseline system-under-test — a chatbot with a guarded system prompt.
Attacks here are *direct* prompt injections: the adversary is the user talking to
the model. Backed by Ollama by default; back it with an ``AnthropicClient`` and
it becomes the optional frontier-model comparison target — same class, different
provider.
"""

from __future__ import annotations

from ..schemas import Payload
from .base import TargetSystem


class DirectTarget(TargetSystem):
    def build_prompt(self, payload: Payload) -> str:
        return payload.payload
