"""The ``Classifier`` interface shared by every success detector."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Payload, RunResult, Verdict


class Classifier(ABC):
    """Turns a (payload, run) pair into a ``Verdict``.

    ``name`` is recorded on every verdict so results carry which classifier
    produced them — important once multiple classifiers run over the same result.
    """

    name: str = "base"

    @abstractmethod
    def classify(self, payload: Payload, run: RunResult) -> Verdict:
        ...
