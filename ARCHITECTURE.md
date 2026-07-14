# Architecture & conventions

Notes on how this project is put together and the conventions that keep each stage
swappable — start here before extending the pipeline.

## What this is

`redteam` is an LLM prompt-injection / jailbreak evaluation harness. It runs a
library of attack payloads against one or more target systems, classifies whether
each attack succeeded, and reports vulnerability rates by category and target.

**Design north star: the whole pipeline runs end to end with zero API keys**, against
a local Ollama model. An Anthropic target/judge can be layered in later for a
local-vs-frontier comparison, but it is always optional and gated — never a
requirement to run anything.

## Architecture (the pipeline)

```
payloads/*.yaml ──▶ payload_loader ──▶ [Payload]
                                          │
                          targets (TargetSystem.run) ──▶ RunResult
                                          │
                       classifiers (Classifier.classify) ──▶ Verdict
                                          │
                    runner  ──▶ EvaluatedResult ──▶ results/run-*.jsonl
                                          │
                        report (CLI, rich)  +  dashboard/app.py (Streamlit)
```

Two orthogonal abstractions keep it extensible:

- **Provider** (`providers.py`, `ModelClient`): *how* to call a model — `OllamaClient`
  (default, free) or `AnthropicClient` (optional, lazy-imported). Add OpenAI/vLLM here.
- **Target** (`targets/`, `TargetSystem`): *what system* is under test —
  `DirectTarget` (raw chat) or `DocumentSummarizationTarget` (indirect injection via a
  document). A target = a framing × a provider. "Anthropic target" is just
  `DirectTarget` backed by `AnthropicClient`.

Both **classifiers** implement `Classifier.classify(payload, run) -> Verdict`:
`RuleBasedClassifier` (deterministic pattern match) and `LLMJudgeClassifier` (a second
model call). They are swappable and can run together.

## Module map

| File | Responsibility |
|------|----------------|
| `redteam/schemas.py` | Pydantic contracts: `Payload`, `RunResult` (incl. `defense`/`trial`), `Verdict`, `EvaluatedResult` |
| `redteam/config.py` | `Settings.from_env()`, planted canaries, target system prompts |
| `redteam/payload_loader.py` | Load + validate YAML payloads; unique-id + schema checks |
| `redteam/providers.py` | `ModelClient` + `OllamaClient` + lazy `AnthropicClient` + `MockClient` (tests/CI); health checks |
| `redteam/defenses.py` | `Defense` ABC + `none`/`spotlight`/`instruction`/`sandwich` + registry |
| `redteam/targets/` | `base.py` (ABC, applies the defense + records defense/trial), `direct.py`, `document_pipeline.py` |
| `redteam/classifiers/` | `base.py` (ABC), `rule_based.py`, `llm_judge.py` |
| `redteam/runner.py` | Build targets (models × defenses) / classifiers; run matrix with trials; JSONL I/O |
| `redteam/report.py` | Pure aggregation (by category/target/defense) + rich CLI + Markdown/HTML export |
| `redteam/cli.py` | `run` / `report` / `compare` / `list` subcommands |
| `dashboard/app.py` | Streamlit UI (imports the same aggregation as the CLI) |

## Key conventions

- **Schemas are the contract between stages.** Change a stage's I/O only by changing
  the schema, so every stage stays swappable.
- **Measurement via planted canaries.** Target system prompts (`config.py`) plant a
  secret passphrase (`BLUE-MOON-42`) and a forbidden token (`PWNED`). A payload
  "succeeds" when its canary surfaces — that's what makes success mechanically
  checkable. New payloads should key their `success` criteria off a canary
  (the planted ones, or a unique self-contained token the payload asks for, e.g.
  `COMPROMISE_ALPHA`).
- **Rule-based is a fast heuristic, not ground truth.** It matches strings, not
  meaning — a refusal that quotes the secret can false-positive. That limitation is
  *why the LLM judge exists*; keep both, don't "fix" rule-based into fragility.
- **The judge is itself an injection target.** The response it grades is
  attacker-influenced, so the judge prompt delimits the response and tells the judge
  to treat it as untrusted data. Preserve that when editing `llm_judge.py`.
- **No new hard dependency on `anthropic`.** It must stay a lazy import inside
  `AnthropicClient`, reachable only when `REDTEAM_ENABLE_ANTHROPIC=1` and
  `ANTHROPIC_API_KEY` are both set.
- Results are self-describing JSONL (each line embeds its `Payload`), so reports and
  the dashboard never need to re-load the YAML to interpret an old run.
- **Defenses are a dimension, not a fork.** A `Defense` rewrites `(system, user)` in
  `TargetSystem.run`; `build_targets` attaches one target variant per defense so
  attack-vs-mitigation is one comparison table. `RunResult.defense`/`trial` (defaulted)
  keep old result files loadable.
- **Backend-agnostic tests.** `MockClient` drives the pipeline with no network — new
  end-to-end tests must use it, never a live model, so CI stays offline.

## Running

```bash
python3.11+ -m venv .venv && source .venv/bin/activate   # 3.11+ required (uses StrEnum)
pip install -e ".[dev]"        # installs the `redteam` command + pytest/ruff
# Prerequisite: Ollama running + a model pulled (ollama pull llama3.1:8b)

redteam run                                  # full matrix, rule-based
redteam run --limit 5 --no-judge --targets ollama-direct   # quick smoke test
redteam run --judge                          # add the LLM-as-judge classifier
redteam run --targets ollama-direct --defenses all         # attack vs. mitigation
redteam run --trials 5 --temperature 0.8     # probabilistic success rate
redteam run --model llama3.1:8b,qwen2.5:7b   # cross-model sweep
redteam report --format html --out report.html             # shareable report
redteam compare A.jsonl B.jsonl              # diff two runs
redteam list                                 # show the payload library
streamlit run dashboard/app.py               # dashboard
pytest ; ruff check .                        # tests (no Ollama) + lint
```

### Environment variables

| Var | Purpose | Default |
|-----|---------|---------|
| `REDTEAM_OLLAMA_MODEL` | Ollama model to attack/judge with | `llama3.1:8b` |
| `OLLAMA_HOST` | Ollama endpoint | library default (`localhost:11434`) |
| `REDTEAM_ENABLE_JUDGE` | Turn on the LLM judge | off |
| `REDTEAM_JUDGE_PROVIDER` / `REDTEAM_JUDGE_MODEL` | Judge backend | `ollama` / same model |
| `REDTEAM_ENABLE_ANTHROPIC` + `ANTHROPIC_API_KEY` | Enable the optional Claude target | off |
| `REDTEAM_ANTHROPIC_MODEL` | Claude model id | `claude-sonnet-5` |
| `REDTEAM_TEMPERATURE` / `REDTEAM_NUM_PREDICT` | Generation determinism / length | `0.0` / `512` |

## Extending

- **Add a payload:** append to the right `payloads/*.yaml`; give it an `id`,
  `category`, `description` (what success looks like), and `success` criteria.
  `python -m redteam list` and `pytest` will validate it.
- **Add a target:** subclass `TargetSystem`, implement `build_prompt`; register it in
  `runner.build_targets`.
- **Add a provider:** subclass `ModelClient`, implement `generate` (+ `health_check`).
- **Add a classifier:** subclass `Classifier`, implement `classify`; add to
  `runner.build_classifiers`.
- **Add a defense:** subclass `Defense`, implement `defend`; register it in the
  `_DEFENSES` dict in `defenses.py`. It's automatically available to `--defenses`.

## Provider & multi-turn notes

- `ModelClient.chat(messages, system)` is the primitive every backend implements;
  `generate(prompt, system)` is a convenience wrapper that calls `chat` with one user
  turn. Multi-turn attacks go through `chat`.
- A payload's `setup_turns` (optional) are priming user messages the model *actually
  answers* before the main `payload` turn; `TargetSystem.run` drives that back-and-forth
  and classifies the final response. Empty `setup_turns` = single-turn (unchanged).
- **Anthropic rejects `temperature`** (current-gen models 400 on it) — `AnthropicClient`
  deliberately never sends it. Don't "restore" it.

## Gotchas

- Requires Python **3.11+** (`StrEnum`, `X | Y` typing at runtime via pydantic). The
  system `python3` on macOS may be 3.9 — use a 3.11+ venv.
- Adding a field to `RunResult` (or `Payload`) must keep a default, or old JSONL won't load.
- `enc-zero-width` payload contains real U+200B characters; keep them intact when
  editing that file.
- The document target concatenates the payload between a fixed prefix/suffix
  (never `str.format`) so payloads containing `{`/`}` can't break templating.
