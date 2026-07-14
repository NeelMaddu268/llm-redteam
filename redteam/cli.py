"""Command-line interface: ``python -m redteam <command>`` (or ``redteam ...``).

Commands:
    run      Run payloads against targets, classify, log, and print a report.
    report   Re-render a report (CLI / Markdown / HTML) from a saved file.
    compare  Diff two runs — regressions and fixes between A and B.
    list     Show the payload library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import Settings
from .defenses import available_defenses
from .payload_loader import DEFAULT_PAYLOAD_DIR, PayloadError, load_payloads
from .providers import GenerationError
from .report import DEFAULT_CLASSIFIER, render_html, render_markdown, render_report
from .runner import (
    build_classifiers,
    build_targets,
    latest_results_file,
    load_results,
    preflight_targets,
    run_evaluation,
)
from .schemas import Category, EvaluatedResult, Outcome

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redteam",
        description="LLM prompt-injection / jailbreak evaluation harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run attacks against targets and report.")
    run_p.add_argument("--payload-dir", default=str(DEFAULT_PAYLOAD_DIR), help="Directory of YAML payloads.")
    run_p.add_argument(
        "--category",
        action="append",
        choices=[c.value for c in Category],
        help="Only run this category (repeatable).",
    )
    run_p.add_argument("--tag", action="append", help="Only run payloads carrying this tag, e.g. 'advanced' (repeatable).")
    run_p.add_argument("--limit", type=int, default=None, help="Cap the number of payloads (useful for a smoke test).")
    run_p.add_argument("--targets", default=None, help="Comma-separated target names/kinds to include (default: all).")
    run_p.add_argument("--model", default=None, help="Ollama model(s), comma-separated for a cross-model sweep (e.g. llama3.1:8b,qwen2.5:7b).")
    run_p.add_argument(
        "--defenses",
        default=None,
        help=f"Defenses to apply, comma-separated, or 'all'. Choices: {', '.join(available_defenses())}. "
        "A 'none' baseline is always included for comparison.",
    )
    run_p.add_argument("--trials", type=int, default=1, help="Runs per (target, payload). >1 measures success frequency (use with --temperature>0).")
    run_p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default 0.0 for reproducibility).")
    run_p.add_argument("--results-dir", default=None, help="Where to write the results JSONL.")
    judge = run_p.add_mutually_exclusive_group()
    judge.add_argument("--judge", dest="judge", action="store_true", default=None, help="Enable the LLM-judge classifier.")
    judge.add_argument("--no-judge", dest="judge", action="store_false", help="Disable the LLM-judge classifier.")
    run_p.add_argument("--classifier", default=DEFAULT_CLASSIFIER, help="Classifier used for headline stats in the report.")

    # --- report ---
    rep_p = sub.add_parser("report", help="Render a report from a saved results file.")
    rep_p.add_argument("--file", default=None, help="Results JSONL (default: newest in results dir).")
    rep_p.add_argument("--results-dir", default=None, help="Results directory to search for the newest run.")
    rep_p.add_argument("--classifier", default=DEFAULT_CLASSIFIER, help="Classifier used for headline stats.")
    rep_p.add_argument("--format", choices=["cli", "md", "html"], default="cli", help="Output format.")
    rep_p.add_argument("--out", default=None, help="Write to this file (md prints to stdout by default; html defaults to <results-dir>/report.html).")

    # --- compare ---
    cmp_p = sub.add_parser("compare", help="Diff two runs (regressions and fixes).")
    cmp_p.add_argument("file_a", help="Baseline results JSONL ('A').")
    cmp_p.add_argument("file_b", help="New results JSONL ('B').")
    cmp_p.add_argument("--classifier", default=DEFAULT_CLASSIFIER, help="Classifier used to decide breakthroughs.")

    # --- list ---
    list_p = sub.add_parser("list", help="List the payload library.")
    list_p.add_argument("--payload-dir", default=str(DEFAULT_PAYLOAD_DIR))
    list_p.add_argument(
        "--category",
        action="append",
        choices=[c.value for c in Category],
        help="Filter by category (repeatable).",
    )

    return parser


def _parse_defenses(raw: str | None) -> list[str] | None:
    """Turn a --defenses value into a validated name list, always incl. baseline."""
    if not raw:
        return None
    if raw.strip().lower() == "all":
        return available_defenses()
    names = [n.strip() for n in raw.split(",") if n.strip()]
    valid = set(available_defenses())
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValueError(f"Unknown defense(s): {', '.join(unknown)}. Choices: {', '.join(available_defenses())}")
    if "none" not in names:  # always keep a baseline to compare against
        names = ["none", *names]
    return names


def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.results_dir:
        settings.results_dir = args.results_dir
    if args.judge is not None:
        settings.judge_enabled = args.judge
    if args.temperature is not None:
        settings.temperature = args.temperature

    models = [m.strip() for m in args.model.split(",")] if args.model else None
    try:
        defenses = _parse_defenses(args.defenses)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    if args.trials < 1:
        console.print("[bold red]--trials must be >= 1[/bold red]")
        return 1
    if args.trials > 1 and settings.temperature == 0.0:
        console.print(
            "[yellow]Note:[/yellow] --trials > 1 at temperature 0 gives identical runs. "
            "Pass e.g. --temperature 0.8 to measure a real success frequency."
        )

    categories = [Category(c) for c in args.category] if args.category else None
    try:
        payloads = load_payloads(args.payload_dir, categories=categories, tags=args.tag)
    except PayloadError as exc:
        console.print(f"[bold red]Payload error:[/bold red] {exc}")
        return 1

    if args.limit is not None:
        payloads = payloads[: args.limit]
    if not payloads:
        console.print("[yellow]No payloads matched the given filters.[/yellow]")
        return 1

    targets = build_targets(settings, models=models, defenses=defenses)
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(",")}
        targets = [t for t in targets if t.name in wanted or t.kind in wanted]
        if not targets:
            console.print(f"[bold red]No targets matched:[/bold red] {args.targets}")
            return 1

    classifiers = build_classifiers(settings)

    console.print(
        f"[bold]Targets:[/bold] {', '.join(t.name for t in targets)}\n"
        f"[bold]Payloads:[/bold] {len(payloads)}   "
        f"[bold]Trials:[/bold] {args.trials}   "
        f"[bold]Classifiers:[/bold] {', '.join(c.name for c in classifiers)}"
    )

    # Fail fast (before the progress bar) if a backend is unavailable.
    try:
        preflight_targets(targets)
    except GenerationError as exc:
        console.print(f"\n[bold red]Cannot start:[/bold red]\n{exc}")
        return 1

    total = len(targets) * len(payloads) * args.trials
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running attacks", total=total)

        def on_progress(done: int, _total: int, label: str) -> None:
            progress.update(task, completed=done, description=f"[cyan]{label}[/cyan]")

        run = run_evaluation(
            payloads=payloads,
            targets=targets,
            classifiers=classifiers,
            results_dir=settings.results_dir,
            trials=args.trials,
            on_progress=on_progress,
            preflight=False,  # already checked above
        )

    console.print(f"\nResults written to [green]{run.output_path}[/green]\n")
    render_report(run.results, console=console, classifier=args.classifier)
    console.print("\n[dim]Dashboard:[/dim] streamlit run dashboard/app.py    [dim]HTML report:[/dim] redteam report --format html")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results_dir = args.results_dir or Settings.from_env().results_dir
    path = Path(args.file) if args.file else latest_results_file(results_dir)
    if path is None or not Path(path).exists():
        console.print("[yellow]No results found.[/yellow] Run [bold]redteam run[/bold] first.")
        return 1
    results = load_results(path)

    if args.format == "cli":
        console.print(f"[dim]Loaded {len(results)} results from {path}[/dim]\n")
        render_report(results, console=console, classifier=args.classifier)
        return 0

    if args.format == "md":
        text = render_markdown(results, classifier=args.classifier)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            console.print(f"Markdown report written to [green]{args.out}[/green]")
        else:
            print(text)  # raw to stdout so it can be piped/redirected
        return 0

    # html
    out = Path(args.out) if args.out else Path(results_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(results, classifier=args.classifier), encoding="utf-8")
    console.print(f"HTML report written to [green]{out}[/green]")
    return 0


def _broke_through(result: EvaluatedResult, classifier: str) -> bool:
    v = result.primary_verdict(prefer=classifier)
    return bool(v and v.outcome is Outcome.succeeded)


def _cmd_compare(args: argparse.Namespace) -> int:
    for f in (args.file_a, args.file_b):
        if not Path(f).exists():
            console.print(f"[bold red]File not found:[/bold red] {f}")
            return 1
    a = load_results(args.file_a)
    b = load_results(args.file_b)
    clf = args.classifier

    def per_payload(results: list[EvaluatedResult]) -> dict[str, bool]:
        # A payload "breaks through" a run if it succeeded on any of its runs.
        broke: dict[str, bool] = {}
        for r in results:
            broke[r.payload.id] = broke.get(r.payload.id, False) or _broke_through(r, clf)
        return broke

    pa, pb = per_payload(a), per_payload(b)
    ra = sum(pa.values())
    rb = sum(pb.values())

    console.rule("[bold]Run comparison[/bold]")
    console.print(f"[bold]A[/bold] {args.file_a}  —  {ra}/{len(pa)} payloads broke through")
    console.print(f"[bold]B[/bold] {args.file_b}  —  {rb}/{len(pb)} payloads broke through")
    delta = rb - ra
    colour = "red" if delta > 0 else "green" if delta < 0 else "dim"
    console.print(f"[{colour}]Change: {delta:+d} payloads[/{colour}] (scored by {clf})\n")

    pids = sorted(set(pa) | set(pb))
    regressions = [p for p in pids if pb.get(p) and not pa.get(p)]  # newly breaking in B
    fixes = [p for p in pids if pa.get(p) and not pb.get(p)]  # no longer breaking in B

    if not regressions and not fixes:
        console.print("[green]No per-payload differences between the two runs.[/green]")
        return 0

    table = Table(header_style="bold cyan")
    table.add_column("Payload")
    table.add_column("A")
    table.add_column("B")
    table.add_column("Change")
    for p in regressions:
        table.add_row(p, "safe", "[red]broke[/red]", "[red]▲ regression[/red]")
    for p in fixes:
        table.add_row(p, "[red]broke[/red]", "safe", "[green]▼ fixed[/green]")
    console.print(table)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    categories = [Category(c) for c in args.category] if args.category else None
    try:
        payloads = load_payloads(args.payload_dir, categories=categories)
    except PayloadError as exc:
        console.print(f"[bold red]Payload error:[/bold red] {exc}")
        return 1

    table = Table(title=f"Payload library ({len(payloads)} payloads)", header_style="bold cyan")
    table.add_column("ID", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Technique", no_wrap=True)
    table.add_column("Success looks like", max_width=70, overflow="ellipsis", no_wrap=True)
    for p in payloads:
        goal = " ".join(p.description.split())  # collapse folded-scalar whitespace
        table.add_row(p.id, p.category.value, p.technique or "-", goal)
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch = {
        "run": _cmd_run,
        "report": _cmd_report,
        "compare": _cmd_compare,
        "list": _cmd_list,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
