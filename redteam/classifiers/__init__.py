"""Classifiers — decide whether an attack succeeded.

Both classifiers share the ``Classifier`` interface (``classify(payload, run) ->
Verdict``) so the runner can apply either or both, and new strategies drop in
without pipeline changes.

* ``RuleBasedClassifier`` — fast, deterministic, transparent pattern matching.
* ``LLMJudgeClassifier`` — a second model call for semantic judgement; more
  robust to paraphrase, but slower and non-deterministic.
"""

from .base import Classifier
from .llm_judge import LLMJudgeClassifier
from .rule_based import RuleBasedClassifier

__all__ = ["Classifier", "RuleBasedClassifier", "LLMJudgeClassifier"]
