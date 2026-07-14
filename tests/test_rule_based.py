"""Tests for the rule-based classifier's matching logic."""

from __future__ import annotations

from redteam.classifiers import RuleBasedClassifier
from redteam.schemas import Category, Outcome, Payload, RunResult, SuccessCriteria

CLF = RuleBasedClassifier()


def make_payload(**success_kwargs) -> Payload:
    return Payload(
        id="t-1",
        category=Category.direct_injection,
        payload="attack text",
        description="what success looks like",
        success=SuccessCriteria(**success_kwargs),
    )


def make_run(response: str, *, error: str | None = None) -> RunResult:
    return RunResult(
        timestamp="2026-07-12T00:00:00Z",
        target="ollama-direct",
        provider="ollama",
        model="llama3.1:8b",
        payload_id="t-1",
        category=Category.direct_injection,
        prompt_sent="...",
        response=response,
        latency_s=0.01,
        error=error,
    )


def test_keyword_match_succeeds():
    v = CLF.classify(make_payload(keywords=["BLUE-MOON-42"]), make_run("The pass is BLUE-MOON-42."))
    assert v.outcome is Outcome.succeeded
    assert "BLUE-MOON-42" in v.matched


def test_no_match_fails():
    v = CLF.classify(make_payload(keywords=["BLUE-MOON-42"]), make_run("I can't help with that."))
    assert v.outcome is Outcome.failed
    assert v.matched == []


def test_keyword_case_insensitive_by_default():
    v = CLF.classify(make_payload(keywords=["pwned"]), make_run("Fine: PWNED"))
    assert v.outcome is Outcome.succeeded


def test_case_sensitive_requires_exact_case():
    payload = make_payload(keywords=["PWNED"], case_sensitive=True)
    assert CLF.classify(payload, make_run("here is pwned")).outcome is Outcome.failed
    assert CLF.classify(payload, make_run("here is PWNED")).outcome is Outcome.succeeded


def test_mode_all_requires_every_indicator():
    payload = make_payload(mode="all", keywords=["alpha", "beta"])
    assert CLF.classify(payload, make_run("only alpha here")).outcome is Outcome.failed
    assert CLF.classify(payload, make_run("alpha and beta")).outcome is Outcome.succeeded


def test_mode_any_needs_one_indicator():
    payload = make_payload(mode="any", keywords=["alpha", "beta"])
    assert CLF.classify(payload, make_run("just beta")).outcome is Outcome.succeeded


def test_regex_match():
    payload = make_payload(regexes=[r"pass(word|phrase)\s*:\s*\w+"])
    v = CLF.classify(payload, make_run("passphrase: hunter2"))
    assert v.outcome is Outcome.succeeded
    assert v.matched  # records which indicator fired


def test_invalid_regex_is_skipped_not_crashed():
    # An unbalanced group is an invalid pattern; it should be ignored, and the
    # valid keyword should still drive the verdict.
    payload = make_payload(keywords=["ok"], regexes=["(unclosed"])
    v = CLF.classify(payload, make_run("this is ok"))
    assert v.outcome is Outcome.succeeded


def test_errored_run_yields_error_outcome():
    payload = make_payload(keywords=["anything"])
    v = CLF.classify(payload, make_run("", error="Ollama not reachable"))
    assert v.outcome is Outcome.error


def test_empty_criteria_never_false_positives():
    v = CLF.classify(make_payload(), make_run("some arbitrary response"))
    assert v.outcome is Outcome.failed
