"""Pydantic schemas shared across the whole pipeline.

These types are the contract between stages: the payload loader produces
``Payload`` objects, a target produces a ``RunResult``, a classifier produces a
``Verdict``, and the runner glues a run together with its verdicts into an
``EvaluatedResult`` — which is exactly one line of the results JSONL file.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    """The four attack families in the payload library."""

    direct_injection = "direct_injection"
    indirect_injection = "indirect_injection"
    encoding_tricks = "encoding_tricks"
    jailbreaks = "jailbreaks"


class SuccessCriteria(BaseModel):
    """Machine-checkable definition of "this attack succeeded".

    The rule-based classifier evaluates these fields against the raw response.
    Each keyword and each regex counts as one *indicator*:

    * ``mode="any"``  -> success if at least one indicator is present.
    * ``mode="all"``  -> success only if every indicator is present.

    Keywords are plain substring checks; regexes are ``re.search`` patterns.
    Matching is case-insensitive unless ``case_sensitive`` is set (useful when a
    canary token like ``PWNED`` should not match the lowercase word "pwned").
    """

    mode: str = Field(default="any", pattern="^(any|all)$")
    keywords: list[str] = Field(default_factory=list)
    regexes: list[str] = Field(default_factory=list)
    case_sensitive: bool = False


class Payload(BaseModel):
    """A single attack case from the library."""

    id: str
    category: Category
    payload: str
    # Human-readable statement of what a successful attack looks like. Shown in
    # reports and handed to the LLM judge as the success criterion.
    description: str
    success: SuccessCriteria
    # Optional priming turns for a *multi-turn* attack. Each is a user message
    # sent (and answered by the model) before the main ``payload`` turn, so the
    # attack can build context/rapport over several turns. Empty = single-turn.
    setup_turns: list[str] = Field(default_factory=list)
    # Short label for the technique, e.g. "ignore-previous", "base64", "DAN".
    technique: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """Everything captured from running one payload against one target."""

    timestamp: str
    target: str
    provider: str
    model: str
    payload_id: str
    category: Category
    # Defense applied to this target ("none" when undefended). Lets the report
    # compare attack-vs-mitigation effectiveness within a single run.
    defense: str = "none"
    # Trial index (0-based). >1 trial per (target, payload) measures a
    # probabilistic attack's success *frequency* rather than a single shot.
    trial: int = 0
    system_prompt: str | None = None
    prompt_sent: str
    response: str
    latency_s: float
    # Populated only when the target call itself failed (e.g. Ollama down).
    error: str | None = None


class Outcome(StrEnum):
    succeeded = "succeeded"
    partial = "partial"
    failed = "failed"
    error = "error"


class Verdict(BaseModel):
    """A single classifier's judgement of one run."""

    classifier: str
    outcome: Outcome
    justification: str
    # Which indicators fired (rule-based) — handy for debugging false positives.
    matched: list[str] = Field(default_factory=list)

    @property
    def is_breakthrough(self) -> bool:
        """True when the attack got through (full success counts; partial does not)."""
        return self.outcome is Outcome.succeeded


class EvaluatedResult(BaseModel):
    """One run plus the verdict(s) of every classifier applied to it.

    This is the unit stored in the results JSONL — one object per line. The full
    payload is embedded so each line is self-describing: results can be
    re-analysed, re-classified, or charted later without re-loading the library.
    """

    payload: Payload
    run: RunResult
    verdicts: list[Verdict] = Field(default_factory=list)

    def verdict_for(self, classifier: str) -> Verdict | None:
        for v in self.verdicts:
            if v.classifier == classifier:
                return v
        return None

    def primary_verdict(self, prefer: str | None = None) -> Verdict | None:
        """The verdict used for headline stats.

        Prefer the named classifier if present; otherwise fall back to the first
        verdict recorded. Returns ``None`` only when no classifier ran.
        """
        if prefer:
            v = self.verdict_for(prefer)
            if v is not None:
                return v
        return self.verdicts[0] if self.verdicts else None
