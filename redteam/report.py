"""Aggregate results into a report.

The aggregation functions are pure (they return plain data) so both the rich CLI
report and the Streamlit dashboard compute stats the same way. Rendering with
``rich`` lives in ``render_report``.

"Breakthrough rate" counts full successes over the runs that actually completed
(errored runs are excluded from the denominator and reported separately, so a
flaky backend doesn't quietly deflate the numbers).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .schemas import EvaluatedResult, Outcome

DEFAULT_CLASSIFIER = "rule_based"


@dataclass
class Bucket:
    """Aggregated outcomes for one group (a category, or a target)."""

    name: str
    total: int = 0
    succeeded: int = 0
    partial: int = 0
    errors: int = 0

    @property
    def evaluated(self) -> int:
        """Runs that produced a judgeable response (total minus errors)."""
        return self.total - self.errors

    @property
    def rate(self) -> float:
        """Breakthrough rate: full successes over evaluated runs."""
        return self.succeeded / self.evaluated if self.evaluated else 0.0


def _outcome(result: EvaluatedResult, classifier: str) -> Outcome | None:
    verdict = result.primary_verdict(prefer=classifier)
    return verdict.outcome if verdict else None


def aggregate(
    results: list[EvaluatedResult],
    *,
    by: str,
    classifier: str = DEFAULT_CLASSIFIER,
) -> list[Bucket]:
    """Group results into buckets by ``"category"`` or ``"target"``."""
    if by == "category":
        key = lambda r: r.run.category.value  # noqa: E731
    elif by == "target":
        key = lambda r: r.run.target  # noqa: E731
    elif by == "defense":
        key = lambda r: r.run.defense  # noqa: E731
    else:
        raise ValueError(f"Unknown grouping: {by!r} (expected 'category', 'target', or 'defense')")

    buckets: dict[str, Bucket] = {}
    for r in results:
        name = key(r)
        bucket = buckets.setdefault(name, Bucket(name=name))
        bucket.total += 1
        outcome = _outcome(r, classifier)
        if outcome is Outcome.succeeded:
            bucket.succeeded += 1
        elif outcome is Outcome.partial:
            bucket.partial += 1
        elif outcome is Outcome.error:
            bucket.errors += 1

    return sorted(buckets.values(), key=lambda b: b.rate, reverse=True)


@dataclass
class PayloadStat:
    payload_id: str
    category: str
    technique: str | None
    description: str
    total: int = 0
    succeeded: int = 0
    targets_hit: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


def worst_payloads(
    results: list[EvaluatedResult],
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    limit: int = 5,
) -> list[PayloadStat]:
    """Payloads that broke through the most, aggregated across every target."""
    stats: dict[str, PayloadStat] = {}
    for r in results:
        pid = r.payload.id
        stat = stats.get(pid)
        if stat is None:
            stat = PayloadStat(
                payload_id=pid,
                category=r.payload.category.value,
                technique=r.payload.technique,
                description=r.payload.description,
            )
            stats[pid] = stat
        stat.total += 1
        if _outcome(r, classifier) is Outcome.succeeded:
            stat.succeeded += 1
            stat.targets_hit.append(r.run.target)

    ranked = [s for s in stats.values() if s.succeeded > 0]
    ranked.sort(key=lambda s: (s.rate, s.succeeded), reverse=True)
    return ranked[:limit]


def interesting_successes(
    results: list[EvaluatedResult],
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    limit: int = 4,
) -> list[EvaluatedResult]:
    """A diverse handful of successful attacks — spread across categories."""
    successes = [r for r in results if _outcome(r, classifier) is Outcome.succeeded]

    picked: list[EvaluatedResult] = []
    seen_categories: set[str] = set()
    # First pass: one per category for variety.
    for r in successes:
        cat = r.run.category.value
        if cat not in seen_categories:
            picked.append(r)
            seen_categories.add(cat)
        if len(picked) >= limit:
            return picked
    # Second pass: fill remaining slots with any other successes.
    for r in successes:
        if r not in picked:
            picked.append(r)
        if len(picked) >= limit:
            break
    return picked


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _rate_text(rate: float) -> Text:
    """Colour a breakthrough rate: green (safe) -> red (leaky)."""
    if rate >= 0.5:
        colour = "bold red"
    elif rate >= 0.2:
        colour = "yellow"
    elif rate > 0:
        colour = "green"
    else:
        colour = "bold green"
    return Text(f"{rate:5.0%}", style=colour)


def _bucket_table(title: str, buckets: list[Bucket], first_col: str) -> Table:
    table = Table(title=title, title_style="bold", header_style="bold cyan", expand=False)
    table.add_column(first_col)
    table.add_column("Breakthrough", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Partial", justify="right")
    table.add_column("Evaluated", justify="right")
    table.add_column("Errors", justify="right")
    for b in buckets:
        table.add_row(
            b.name,
            _rate_text(b.rate),
            str(b.succeeded),
            str(b.partial),
            str(b.evaluated),
            str(b.errors) if b.errors else "-",
        )
    return table


def _truncate(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " […]"


def render_report(
    results: list[EvaluatedResult],
    *,
    console: Console | None = None,
    classifier: str = DEFAULT_CLASSIFIER,
    transcripts: int = 3,
) -> None:
    """Print the full CLI report: summary tables, worst payloads, transcripts."""
    console = console or Console()

    if not results:
        console.print("[yellow]No results to report.[/yellow]")
        return

    classifiers_used = sorted({v.classifier for r in results for v in r.verdicts})
    total = len(results)
    breakthroughs = sum(1 for r in results if _outcome(r, classifier) is Outcome.succeeded)

    console.rule("[bold]LLM Red-Team Report[/bold]")
    console.print(
        f"Runs: [bold]{total}[/bold]   "
        f"Breakthroughs: [bold red]{breakthroughs}[/bold red]   "
        f"Overall rate: ",
        end="",
    )
    console.print(_rate_text(breakthroughs / total if total else 0.0))
    console.print(
        f"Scored by: [cyan]{classifier}[/cyan]   "
        f"Classifiers available: {', '.join(classifiers_used)}\n"
    )

    console.print(_bucket_table("Breakthrough rate by category", aggregate(results, by="category", classifier=classifier), "Category"))
    console.print()
    console.print(_bucket_table("Breakthrough rate by target", aggregate(results, by="target", classifier=classifier), "Target"))
    console.print()

    # Only meaningful when more than one defense was exercised.
    defenses = {r.run.defense for r in results}
    if len(defenses) > 1:
        console.print(_bucket_table("Breakthrough rate by defense", aggregate(results, by="defense", classifier=classifier), "Defense"))
        console.print()

    # Worst payloads
    worst = worst_payloads(results, classifier=classifier, limit=5)
    if worst:
        table = Table(title="Most effective payloads", title_style="bold", header_style="bold cyan")
        table.add_column("Payload")
        table.add_column("Category")
        table.add_column("Rate", justify="right")
        table.add_column("Targets hit")
        for s in worst:
            table.add_row(
                s.payload_id,
                s.category,
                _rate_text(s.rate),
                ", ".join(sorted(set(s.targets_hit))) or "-",
            )
        console.print(table)
        console.print()

    # Example transcripts
    examples = interesting_successes(results, classifier=classifier, limit=transcripts)
    if examples:
        console.rule("[bold]Example successful attacks[/bold]")
        for r in examples:
            rule = r.verdict_for("rule_based")
            judge = r.verdict_for("llm_judge")
            verdict_line = ""
            if rule:
                verdict_line += f"rule_based=[bold]{rule.outcome.value}[/bold]"
            if judge:
                verdict_line += f"   llm_judge=[bold]{judge.outcome.value}[/bold] — {judge.justification}"
            body = Text()
            body.append("Attack goal: ", style="bold")
            body.append(r.payload.description + "\n\n")
            body.append("Prompt sent:\n", style="bold")
            body.append(_truncate(r.run.prompt_sent, 500) + "\n\n", style="dim")
            body.append("Model response:\n", style="bold")
            body.append(_truncate(r.run.response, 500) + "\n", style="red")
            console.print(
                Panel(
                    body,
                    title=f"[bold]{r.payload.id}[/bold]  ·  {r.run.target}  ·  {r.run.category.value}",
                    subtitle=verdict_line or None,
                    border_style="red",
                )
            )


# ---------------------------------------------------------------------------
# Exports: Markdown and self-contained HTML (shareable report artifacts)
# ---------------------------------------------------------------------------
def _headline(results: list[EvaluatedResult], classifier: str) -> tuple[int, int, float]:
    total = len(results)
    breakthroughs = sum(1 for r in results if _outcome(r, classifier) is Outcome.succeeded)
    return total, breakthroughs, (breakthroughs / total if total else 0.0)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _bucket_rows(results: list[EvaluatedResult], by: str, classifier: str) -> list[list[str]]:
    return [
        [b.name, f"{b.rate:.0%}", str(b.succeeded), str(b.partial), str(b.evaluated), str(b.errors)]
        for b in aggregate(results, by=by, classifier=classifier)
    ]


def render_markdown(
    results: list[EvaluatedResult],
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    transcripts: int = 3,
) -> str:
    """Render the report as a Markdown string (for sharing or embedding)."""
    if not results:
        return "# LLM Red-Team Report\n\n_No results._\n"

    total, breakthroughs, rate = _headline(results, classifier)
    classifiers_used = sorted({v.classifier for r in results for v in r.verdicts})
    cols = ["Group", "Rate", "Success", "Partial", "Evaluated", "Errors"]

    parts = [
        "# 🛡️ LLM Red-Team Report",
        "",
        f"**{total} runs · {breakthroughs} breakthroughs · {rate:.0%} overall** — "
        f"scored by `{classifier}` (available: {', '.join(classifiers_used)})",
        "",
        "## Breakthrough rate by category",
        _md_table(cols, _bucket_rows(results, "category", classifier)),
        "",
        "## Breakthrough rate by target",
        _md_table(cols, _bucket_rows(results, "target", classifier)),
    ]

    if len({r.run.defense for r in results}) > 1:
        parts += [
            "",
            "## Breakthrough rate by defense",
            _md_table(cols, _bucket_rows(results, "defense", classifier)),
        ]

    worst = worst_payloads(results, classifier=classifier, limit=10)
    if worst:
        rows = [
            [s.payload_id, s.category, f"{s.rate:.0%}", ", ".join(sorted(set(s.targets_hit))) or "-", s.description]
            for s in worst
        ]
        parts += ["", "## Most effective payloads", _md_table(["Payload", "Category", "Rate", "Targets hit", "Goal"], rows)]

    examples = interesting_successes(results, classifier=classifier, limit=transcripts)
    if examples:
        parts += ["", "## Example successful attacks"]
        for r in examples:
            verdicts = "; ".join(f"`{v.classifier}` → {v.outcome.value}" for v in r.verdicts)
            parts += [
                "",
                f"### {r.payload.id} · {r.run.target} · {r.run.category.value}",
                f"**Goal:** {r.payload.description}",
                "",
                "**Prompt sent:**",
                "```text",
                _truncate(r.run.prompt_sent, 800),
                "```",
                "**Model response:**",
                "```text",
                _truncate(r.run.response, 800),
                "```",
                f"_Verdicts: {verdicts}_",
            ]

    return "\n".join(parts) + "\n"


_HTML_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       max-width: 1000px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.5;
       color: #1a1a1a; background: #fff; }
h1 { font-size: 1.7rem; } h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #e5e5e5; padding-bottom: .3rem; }
.summary { font-size: 1.05rem; background: #f6f8fa; border: 1px solid #e5e5e5; border-radius: 8px; padding: .8rem 1rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .92rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; }
th { background: #f6f8fa; }
td.rate { font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }
.r-hi { color: #b30000; } .r-mid { color: #b26a00; } .r-lo { color: #1a7f37; }
details { border: 1px solid #e5e5e5; border-radius: 8px; padding: .5rem .9rem; margin: .6rem 0; }
summary { cursor: pointer; font-weight: 600; }
pre { background: #f6f8fa; border-radius: 6px; padding: .6rem .8rem; overflow-x: auto; font-size: .85rem; white-space: pre-wrap; word-break: break-word; }
.goal { color: #444; margin: .4rem 0; }
.verdicts { font-size: .85rem; color: #555; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #0d1117; }
  h2 { border-color: #30363d; } .summary, th, pre { background: #161b22; }
  .summary, details { border-color: #30363d; } th, td { border-color: #30363d; }
  .r-hi { color: #ff6a6a; } .r-mid { color: #e3a008; } .r-lo { color: #3fb950; } .goal { color: #aaa; } .verdicts { color: #9aa; }
}
"""


def _rate_class(rate: float) -> str:
    return "r-hi" if rate >= 0.5 else "r-mid" if rate >= 0.2 else "r-lo"


def _html_bucket_table(results: list[EvaluatedResult], by: str, classifier: str, first_col: str) -> str:
    head = f"<tr><th>{first_col}</th><th>Rate</th><th>Success</th><th>Partial</th><th>Evaluated</th><th>Errors</th></tr>"
    rows = []
    for b in aggregate(results, by=by, classifier=classifier):
        rows.append(
            f"<tr><td>{html.escape(b.name)}</td>"
            f"<td class='rate {_rate_class(b.rate)}'>{b.rate:.0%}</td>"
            f"<td>{b.succeeded}</td><td>{b.partial}</td><td>{b.evaluated}</td><td>{b.errors}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def render_html(
    results: list[EvaluatedResult],
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    transcripts: int = 5,
) -> str:
    """Render the report as a single self-contained HTML page."""
    if not results:
        return "<!doctype html><meta charset='utf-8'><h1>LLM Red-Team Report</h1><p>No results.</p>"

    total, breakthroughs, rate = _headline(results, classifier)
    classifiers_used = sorted({v.classifier for r in results for v in r.verdicts})

    body = [
        "<h1>🛡️ LLM Red-Team Report</h1>",
        f"<p class='summary'><b>{total}</b> runs · <b>{breakthroughs}</b> breakthroughs · "
        f"<b class='{_rate_class(rate)}'>{rate:.0%}</b> overall — scored by <code>{html.escape(classifier)}</code> "
        f"(available: {html.escape(', '.join(classifiers_used))})</p>",
        "<h2>Breakthrough rate by category</h2>",
        _html_bucket_table(results, "category", classifier, "Category"),
        "<h2>Breakthrough rate by target</h2>",
        _html_bucket_table(results, "target", classifier, "Target"),
    ]

    if len({r.run.defense for r in results}) > 1:
        body += ["<h2>Breakthrough rate by defense</h2>", _html_bucket_table(results, "defense", classifier, "Defense")]

    worst = worst_payloads(results, classifier=classifier, limit=10)
    if worst:
        rows = "".join(
            f"<tr><td>{html.escape(s.payload_id)}</td><td>{html.escape(s.category)}</td>"
            f"<td class='rate {_rate_class(s.rate)}'>{s.rate:.0%}</td>"
            f"<td>{html.escape(', '.join(sorted(set(s.targets_hit))) or '-')}</td>"
            f"<td>{html.escape(s.description)}</td></tr>"
            for s in worst
        )
        body += [
            "<h2>Most effective payloads</h2>",
            f"<table><tr><th>Payload</th><th>Category</th><th>Rate</th><th>Targets hit</th><th>Goal</th></tr>{rows}</table>",
        ]

    examples = interesting_successes(results, classifier=classifier, limit=transcripts)
    if examples:
        body.append("<h2>Example successful attacks</h2>")
        for r in examples:
            verdicts = "; ".join(f"<code>{v.classifier}</code> → {v.outcome.value}" for v in r.verdicts)
            body.append(
                f"<details><summary>{html.escape(r.payload.id)} · {html.escape(r.run.target)} · "
                f"{r.run.category.value}</summary>"
                f"<p class='goal'><b>Goal:</b> {html.escape(r.payload.description)}</p>"
                f"<b>Prompt sent</b><pre>{html.escape(_truncate(r.run.prompt_sent, 1200))}</pre>"
                f"<b>Model response</b><pre>{html.escape(_truncate(r.run.response, 1200))}</pre>"
                f"<p class='verdicts'>{verdicts}</p></details>"
            )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>LLM Red-Team Report</title>"
        f"<style>{_HTML_STYLE}</style></head><body>{''.join(body)}</body></html>"
    )
