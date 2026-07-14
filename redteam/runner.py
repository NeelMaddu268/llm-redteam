"""The runner: load payloads -> run against targets -> classify -> log results.

This module wires configuration into concrete targets and classifiers, executes
the (target x payload) matrix, applies every classifier to each response, and
streams the results to a JSONL file as it goes (so a long run is never lost).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .classifiers import Classifier, LLMJudgeClassifier, RuleBasedClassifier
from .config import DIRECT_SYSTEM_PROMPT, DOC_SYSTEM_PROMPT, Settings
from .defenses import build_defense
from .providers import AnthropicClient, ModelClient, build_ollama_client
from .schemas import EvaluatedResult, Payload
from .targets import DirectTarget, DocumentSummarizationTarget, TargetSystem

ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Assembling targets and classifiers from settings
# ---------------------------------------------------------------------------
def build_targets(
    settings: Settings,
    *,
    models: list[str] | None = None,
    defenses: list[str] | None = None,
) -> list[TargetSystem]:
    """Construct the target list.

    The matrix is (model x base-framing x defense). Ollama targets are always
    included; Anthropic is added only when enabled. Targets sharing a model reuse
    one client so a single health check covers them.

    Args:
        models: Ollama models to attack (default: the configured one). More than
            one produces a cross-model comparison; each must be pulled locally.
        defenses: defense names to apply (default: ``["none"]``). Each base target
            gets one variant per defense so the report can compare mitigations.
    """
    models = models or [settings.ollama_model]
    defense_names = defenses or ["none"]
    multi_model = len(models) > 1
    targets: list[TargetSystem] = []

    for model in models:
        client = build_ollama_client(settings, model=model)
        model_suffix = f"[{model}]" if multi_model else ""
        for dname in defense_names:
            defense = build_defense(dname)
            dsuffix = "" if dname == "none" else f"+{dname}"
            targets.append(
                DirectTarget(
                    client,
                    name=f"ollama-direct{dsuffix}{model_suffix}",
                    system_prompt=DIRECT_SYSTEM_PROMPT,
                    defense=defense,
                    kind="ollama-direct",
                )
            )
            targets.append(
                DocumentSummarizationTarget(
                    client,
                    name=f"ollama-document{dsuffix}{model_suffix}",
                    system_prompt=DOC_SYSTEM_PROMPT,
                    defense=defense,
                    kind="ollama-document",
                )
            )

    if settings.anthropic_enabled:
        # Only reached when REDTEAM_ENABLE_ANTHROPIC=1 and ANTHROPIC_API_KEY is set.
        # Baseline (no defense) only, to avoid multiplying paid API calls.
        anthropic = AnthropicClient(
            settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.num_predict,
        )
        targets.append(
            DirectTarget(
                anthropic,
                name="anthropic-direct",
                system_prompt=DIRECT_SYSTEM_PROMPT,
                kind="anthropic-direct",
            )
        )
    return targets


def build_judge_client(settings: Settings) -> ModelClient:
    """Pick the model backing the LLM judge (defaults to the local Ollama model)."""
    if settings.judge_provider == "anthropic":
        # Default the Anthropic judge to Haiku: it's the cheapest model and plenty
        # reliable for the binary "did the attack succeed?" call — and far more
        # accurate than the local 8B judge, which is the weak link once a frontier
        # model is a target. Override with REDTEAM_JUDGE_MODEL.
        return AnthropicClient(
            settings.judge_model or "claude-haiku-4-5",
            api_key=settings.anthropic_api_key,
            temperature=0.0,
            max_tokens=256,
        )
    return build_ollama_client(settings, model=settings.judge_model or settings.ollama_model)


def build_classifiers(settings: Settings) -> list[Classifier]:
    """Rule-based is always on; the LLM judge is layered in when enabled."""
    classifiers: list[Classifier] = [RuleBasedClassifier()]
    if settings.judge_enabled:
        classifiers.append(LLMJudgeClassifier(build_judge_client(settings)))
    return classifiers


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
@dataclass
class EvaluationRun:
    """The output of a full sweep: results plus where they were written."""

    results: list[EvaluatedResult]
    output_path: Path
    target_names: list[str]
    classifier_names: list[str]


def preflight_targets(targets: list[TargetSystem]) -> None:
    """Health-check each distinct provider once, failing fast with guidance.

    De-duplicates by client identity so two targets sharing an Ollama client
    trigger a single check. Raises ``OllamaNotAvailableError`` (or another
    ``GenerationError``) if a backend is unusable.
    """
    checked: set[int] = set()
    for target in targets:
        if id(target.client) in checked:
            continue
        target.health_check()
        checked.add(id(target.client))


def _results_path(results_dir: Path | str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"run-{stamp}.jsonl"


def run_evaluation(
    *,
    payloads: list[Payload],
    targets: list[TargetSystem],
    classifiers: list[Classifier],
    results_dir: Path | str = "results",
    trials: int = 1,
    on_progress: ProgressCallback | None = None,
    preflight: bool = True,
) -> EvaluationRun:
    """Run every payload against every target, classify, and log to JSONL.

    Args:
        payloads: attack cases to run.
        targets: systems-under-test.
        classifiers: applied to every response, in order.
        results_dir: directory for the ``run-<timestamp>.jsonl`` output file.
        trials: how many times to run each (target, payload) pair. >1 measures a
            probabilistic attack's success frequency (pair with temperature>0).
        on_progress: optional ``(done, total, label)`` callback for a progress UI.
        preflight: if True, health-check each distinct provider up front so a
            down/misconfigured backend fails immediately instead of once per run.

    Returns:
        An ``EvaluationRun`` with the in-memory results and the output path.
    """
    if preflight:
        preflight_targets(targets)

    output_path = _results_path(results_dir)
    total = len(targets) * len(payloads) * trials
    done = 0
    results: list[EvaluatedResult] = []

    with output_path.open("w", encoding="utf-8") as fh:
        for target in targets:
            for payload in payloads:
                for trial in range(trials):
                    run = target.run(payload, trial=trial)
                    verdicts = [c.classify(payload, run) for c in classifiers]
                    result = EvaluatedResult(payload=payload, run=run, verdicts=verdicts)
                    results.append(result)
                    fh.write(result.model_dump_json() + "\n")
                    fh.flush()
                    done += 1
                    if on_progress is not None:
                        suffix = f" #{trial + 1}" if trials > 1 else ""
                        on_progress(done, total, f"{target.name} / {payload.id}{suffix}")

    return EvaluationRun(
        results=results,
        output_path=output_path,
        target_names=[t.name for t in targets],
        classifier_names=[c.name for c in classifiers],
    )


# ---------------------------------------------------------------------------
# Loading results back (for `report` and the dashboard)
# ---------------------------------------------------------------------------
def load_results(path: Path | str) -> list[EvaluatedResult]:
    """Read a results JSONL file back into ``EvaluatedResult`` objects."""
    path = Path(path)
    results: list[EvaluatedResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(EvaluatedResult.model_validate_json(line))
    return results


def latest_results_file(results_dir: Path | str = "results") -> Path | None:
    """Return the most recently written ``run-*.jsonl`` file, or None if there are none."""
    directory = Path(results_dir)
    if not directory.is_dir():
        return None
    runs = sorted(directory.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None
