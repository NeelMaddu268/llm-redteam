"""Streamlit dashboard for red-team results.

Run from the project root:

    streamlit run dashboard/app.py

Reads the JSONL files written by ``redteam run`` and reuses the same aggregation
functions as the CLI report, so the numbers always match.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the `redteam` package importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from redteam.report import (  # noqa: E402
    DEFAULT_CLASSIFIER,
    aggregate,
    interesting_successes,
    worst_payloads,
)
from redteam.runner import load_results  # noqa: E402
from redteam.schemas import Outcome  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SAMPLE = "three-way-model-comparison.jsonl"  # what the hosted demo opens on

st.set_page_config(page_title="RedTeam · LLM injection eval", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner=False)
def load_results_cached(path_str: str, _mtime: float):
    """Parse a results file once and cache it (keyed by path + mtime).

    Without this, every click re-parses every run into pydantic objects and
    re-aggregates — churn that, under memory pressure, can crash the process.
    """
    return load_results(path_str)


# Control chars (except tab/newline/return), zero-width & bidi marks, line/para
# separators, and BOM — built from code points so this source stays pure ASCII.
_UNSAFE_CODEPOINTS = (
    list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20))
    + list(range(0x200B, 0x2010)) + [0x2028, 0x2029, 0xFEFF]
)
_UNSAFE = re.compile("[" + "".join(re.escape(chr(c)) for c in _UNSAFE_CODEPOINTS) + "]")


def clean(value, limit: int = 240) -> str:
    """Make a value safe + tidy for a table cell (strip control/zero-width chars)."""
    text = _UNSAFE.sub("", str(value)).replace("\n", " ").strip()
    return text[:limit]


def esc(value) -> str:
    """HTML-escape for custom markup."""
    return (
        str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# --- Severity encoding (reserved status palette; higher rate = worse) ----------
BAND_ORDER = ["good", "warning", "serious", "critical"]
BAND_COLORS = ["#3fb950", "#e3b341", "#ec8e4b", "#f0564a"]
BAND_HEX = dict(zip(BAND_ORDER, BAND_COLORS, strict=True))


def band(rate: float) -> str:
    if rate >= 0.5:
        return "critical"
    if rate >= 0.25:
        return "serious"
    if rate > 0:
        return "warning"
    return "good"


OUTCOME_DISPLAY = {
    "succeeded": "● breakthrough",
    "partial": "◐ partial",
    "failed": "○ blocked",
    "error": "· error",
}

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

      :root{
        --bg:#08080a; --surface:#111114; --surface-2:#17171b;
        --border:rgba(255,255,255,.07); --border-2:rgba(255,255,255,.12);
        --ink:#eef0f3; --dim:#9b9ba6; --mute:#63636e; --accent:#5b8cff;
        --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
        --display:'Space Grotesk', system-ui, sans-serif;
        --sans:system-ui, -apple-system, 'Segoe UI', sans-serif;
      }

      /* strip Streamlit chrome */
      [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; }
      [data-testid="stDecoration"] { display:none !important; }
      .stApp { background:
          radial-gradient(1200px 500px at 15% -10%, rgba(91,140,255,.08), transparent 60%),
          var(--bg); }
      .block-container { padding:1.4rem 2.6rem 4rem; max-width:1280px; }
      html, body, [class*="css"] { font-family:var(--sans); color:var(--ink); }

      /* top hairline accent */
      .rt-topbar { height:2px; background:linear-gradient(90deg,var(--accent),#8a5bff 55%,transparent);
                   margin:-1.4rem -2.6rem 1.6rem; }

      /* header */
      .rt-head { display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
                 flex-wrap:wrap; margin-bottom:.2rem; }
      .rt-brand { display:flex; align-items:center; gap:.6rem; }
      .rt-brand svg { width:26px; height:26px; }
      .rt-title { font-family:var(--display); font-weight:700; font-size:1.5rem; letter-spacing:-.02em;
                  color:#fff; margin:0; }
      .rt-title span { color:var(--accent); }
      .rt-meta { font-family:var(--mono); font-size:.74rem; color:var(--mute); text-align:right; line-height:1.5; }
      .rt-tag { color:var(--dim); }
      .rt-lead { color:var(--dim); font-size:.95rem; margin:.15rem 0 1.5rem; max-width:60ch; }

      /* KPI row */
      .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:.6rem; }
      .kpi { background:linear-gradient(180deg, var(--surface-2), var(--surface));
             border:1px solid var(--border); border-radius:12px; padding:.9rem 1rem 1rem;
             position:relative; overflow:hidden; }
      .kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:var(--accent); opacity:.8; }
      .kpi.hero::before { width:3px; }
      .kpi-label { font-family:var(--mono); font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
                   color:var(--mute); margin-bottom:.5rem; }
      .kpi-value { font-family:var(--mono); font-weight:600; font-size:1.9rem; line-height:1;
                   color:var(--ink); font-variant-numeric:tabular-nums; }
      .kpi.hero .kpi-value { font-size:2.6rem; }
      .kpi-sub { font-size:.74rem; color:var(--mute); margin-top:.45rem; }

      /* section label */
      .rt-sec { display:flex; align-items:center; gap:.8rem; margin:2rem 0 .9rem; }
      .rt-sec h2 { font-family:var(--mono); font-size:.76rem; letter-spacing:.14em; text-transform:uppercase;
                   color:var(--dim); font-weight:600; margin:0; white-space:nowrap; }
      .rt-sec .rule { height:1px; background:var(--border); flex:1; }
      .rt-chip { font-family:var(--mono); font-size:.72rem; color:var(--dim);
                 border:1px solid var(--border-2); border-radius:999px; padding:.12rem .55rem; }
      .legend { display:flex; gap:1rem; font-size:.76rem; color:var(--mute); margin:-.4rem 0 .8rem; flex-wrap:wrap; }
      .legend b { font-weight:600; }
      .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.3rem; vertical-align:middle; }
      .cap { font-family:var(--mono); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--mute); margin:.2rem 0 .3rem; }

      /* custom table */
      .rt-table { width:100%; border-collapse:collapse; font-size:.86rem; }
      .rt-table th { text-align:left; font-family:var(--mono); font-size:.68rem; letter-spacing:.07em;
                     text-transform:uppercase; color:var(--mute); font-weight:500; padding:.3rem .7rem; border-bottom:1px solid var(--border-2); }
      .rt-table td { padding:.55rem .7rem; border-bottom:1px solid var(--border); color:var(--dim); vertical-align:middle; }
      .rt-table tr:hover td { background:rgba(255,255,255,.02); }
      .rt-table .mono { font-family:var(--mono); color:var(--ink); font-size:.82rem; }
      .rt-bar { display:flex; align-items:center; gap:.5rem; min-width:130px; }
      .rt-bar .track { flex:1; height:7px; border-radius:4px; background:rgba(255,255,255,.06); overflow:hidden; }
      .rt-bar .fill { height:100%; border-radius:4px; }
      .rt-bar .pct { font-family:var(--mono); font-size:.76rem; color:var(--ink); min-width:34px; text-align:right; font-variant-numeric:tabular-nums; }
      .goal { color:var(--mute); font-size:.8rem; }

      /* sidebar */
      [data-testid="stSidebar"] { background:#0c0c0f; border-right:1px solid var(--border); }
      [data-testid="stSidebar"] .block-container { padding-top:1.4rem; }
      .side-brand { font-family:var(--display); font-weight:700; color:#fff; font-size:1.05rem; display:flex; align-items:center; gap:.45rem; margin-bottom:.2rem; }
      .side-brand svg { width:20px; height:20px; }
      .side-tag { font-family:var(--mono); font-size:.68rem; color:var(--mute); letter-spacing:.05em; margin-bottom:1.2rem; }
      .side-foot { font-family:var(--mono); font-size:.7rem; color:var(--mute); margin-top:1.4rem; padding-top:1rem; border-top:1px solid var(--border); line-height:1.7; }
      .side-foot a { color:var(--dim); text-decoration:none; } .side-foot a:hover { color:var(--accent); }

      /* dataframe + expander polish */
      [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; }
      [data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:10px !important; background:var(--surface); }
      [data-testid="stExpander"] summary { font-family:var(--mono); font-size:.82rem; }
      .stCode, pre { border-radius:8px !important; }
      label[data-testid="stWidgetLabel"] p { font-family:var(--mono); font-size:.7rem !important; letter-spacing:.05em; text-transform:uppercase; color:var(--mute); }
    </style>
    """,
    unsafe_allow_html=True,
)

SHIELD = (
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 2l8 3v6c0 5-3.5 8.5-8 11-4.5-2.5-8-6-8-11V5l8-3z" '
    'fill="rgba(91,140,255,.14)" stroke="#5b8cff" stroke-width="1.4"/>'
    '<path d="M8.5 12l2.3 2.3 4.7-4.9" stroke="#5b8cff" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def rate_bar_chart(buckets, axis_label: str, height_step: int = 40) -> alt.LayerChart:
    df = pd.DataFrame(
        [
            {"group": b.name, "rate": b.rate, "band": band(b.rate),
             "succeeded": b.succeeded, "evaluated": b.evaluated}
            for b in buckets
        ]
    )
    base = alt.Chart(df).encode(
        y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12))
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=16).encode(
        x=alt.X("rate:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, grid=True, domain=False)),
        color=alt.Color("band:N", scale=alt.Scale(domain=BAND_ORDER, range=BAND_COLORS), legend=None),
        tooltip=[alt.Tooltip("group:N", title=axis_label),
                 alt.Tooltip("rate:Q", format=".0%", title="Rate"),
                 alt.Tooltip("succeeded:Q", title="Breakthroughs"),
                 alt.Tooltip("evaluated:Q", title="Evaluated")],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight="bold", color="#e7e7e3").encode(
        x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("rate:Q", format=".0%"),
    )
    return (bars + labels).properties(height=alt.Step(height_step))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f'<div class="side-brand">{SHIELD}RedTeam</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-tag">LLM INJECTION EVAL</div>', unsafe_allow_html=True)

run_files = sorted(RESULTS_DIR.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
if not run_files:
    # Deploy fallback: results/ is gitignored, so a hosted demo shows bundled samples.
    run_files = sorted((PROJECT_ROOT / "sample_results").glob("*.jsonl"))
if not run_files:
    st.markdown('<div class="rt-topbar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rt-head"><div class="rt-brand">{SHIELD}<h1 class="rt-title">Red<span>Team</span></h1></div></div>', unsafe_allow_html=True)
    st.warning("No results found yet. Generate some with:\n\n```\nredteam run\n```")
    st.stop()

file_names = [p.name for p in run_files]
file_idx = file_names.index(DEFAULT_SAMPLE) if DEFAULT_SAMPLE in file_names else 0
choice = st.sidebar.selectbox("Results file", run_files, index=file_idx, format_func=lambda p: p.name)
results = load_results_cached(str(choice), choice.stat().st_mtime)

classifiers = sorted({v.classifier for r in results for v in r.verdicts})
default_idx = classifiers.index(DEFAULT_CLASSIFIER) if DEFAULT_CLASSIFIER in classifiers else 0
classifier = st.sidebar.selectbox("Score by classifier", classifiers, index=default_idx)

n_payloads = len({r.payload.id for r in results})
n_targets = len({r.run.target for r in results})
n_defenses = len({r.run.defense for r in results})
st.sidebar.markdown(
    f'<div class="side-foot">{len(results)} runs · {n_payloads} payloads · {n_targets} targets'
    + (f" · {n_defenses} defenses" if n_defenses > 1 else "")
    + '<br><br>A breakthrough = the attack worked:<br>a secret leaked or a rule broke.'
    '<br><br><a href="https://github.com/NeelMaddu268/llm-redteam">★ view on github</a></div>',
    unsafe_allow_html=True,
)


def primary_outcome(r) -> str:
    v = r.primary_verdict(prefer=classifier)
    return v.outcome.value if v else "error"


# ---------------------------------------------------------------------------
# Header + KPI tiles
# ---------------------------------------------------------------------------
st.markdown('<div class="rt-topbar"></div>', unsafe_allow_html=True)

total = len(results)
succeeded = sum(1 for r in results if primary_outcome(r) == Outcome.succeeded.value)
errors = sum(1 for r in results if primary_outcome(r) == Outcome.error.value)
evaluated = total - errors
rate = succeeded / evaluated if evaluated else 0.0
hero_accent = BAND_HEX[band(rate)]

st.markdown(
    f'<div class="rt-head">'
    f'<div class="rt-brand">{SHIELD}<h1 class="rt-title">Red<span>Team</span> — LLM injection eval</h1></div>'
    f'<div class="rt-meta"><span class="rt-tag">scored by</span> {esc(classifier)}<br>'
    f'{esc(choice.name)}</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="rt-lead">Testing AI systems against a library of prompt-injection &amp; jailbreak '
    'attacks — capturing responses, classifying whether each attack broke through, and comparing '
    'vulnerability across models.</div>',
    unsafe_allow_html=True,
)


def kpi(label, value, sub, accent="var(--accent)", hero=False, value_color=None):
    vstyle = f' style="color:{value_color}"' if value_color else ""
    return (
        f'<div class="kpi{" hero" if hero else ""}" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"{vstyle}>{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


st.markdown(
    '<div class="kpi-row">'
    + kpi("Total runs", f"{total:,}", f"{n_payloads} payloads · {n_targets} targets")
    + kpi("Breakthroughs", f"{succeeded:,}", "attacks that got through", accent="#f0564a")
    + kpi("Breakthrough rate", f"{rate:.0%}", "of evaluated runs", accent=hero_accent, hero=True, value_color=hero_accent)
    + kpi("Errored", f"{errors:,}", "excluded from the rate", accent="#63636e")
    + "</div>",
    unsafe_allow_html=True,
)


def section(title, chip=None):
    chip_html = f'<span class="rt-chip">{esc(chip)}</span>' if chip else ""
    st.markdown(f'<div class="rt-sec"><h2>{title}</h2>{chip_html}<div class="rule"></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts — target comparison is the hero
# ---------------------------------------------------------------------------
section("Breakthrough rate by target", chip="the comparison")
st.markdown(
    '<div class="legend">'
    f'<span><i class="dot" style="background:{BAND_HEX["good"]}"></i>none</span>'
    f'<span><i class="dot" style="background:{BAND_HEX["warning"]}"></i>&lt;25%</span>'
    f'<span><i class="dot" style="background:{BAND_HEX["serious"]}"></i>25–50%</span>'
    f'<span><i class="dot" style="background:{BAND_HEX["critical"]}"></i>≥50%</span>'
    "</div>",
    unsafe_allow_html=True,
)
st.altair_chart(rate_bar_chart(aggregate(results, by="target", classifier=classifier), "Target", 44),
                width="stretch", theme="streamlit")

section("By attack category" + ("  ·  by defense" if n_defenses > 1 else ""))
if n_defenses > 1:
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="cap">Attack category</div>', unsafe_allow_html=True)
        st.altair_chart(rate_bar_chart(aggregate(results, by="category", classifier=classifier), "Category"),
                        width="stretch", theme="streamlit")
    with right:
        st.markdown('<div class="cap">Defense (attack vs. mitigation)</div>', unsafe_allow_html=True)
        st.altair_chart(rate_bar_chart(aggregate(results, by="defense", classifier=classifier), "Defense"),
                        width="stretch", theme="streamlit")
else:
    st.altair_chart(rate_bar_chart(aggregate(results, by="category", classifier=classifier), "Category"),
                    width="stretch", theme="streamlit")

# ---------------------------------------------------------------------------
# Most effective payloads — bespoke table with inline rate bars
# ---------------------------------------------------------------------------
section("Most effective payloads")
worst = worst_payloads(results, classifier=classifier, limit=10)
if worst:
    rows_html = ""
    for s in worst:
        pct = round(s.rate * 100)
        col = BAND_HEX[band(s.rate)]
        hits = clean(", ".join(sorted(set(s.targets_hit))), 60)
        rows_html += (
            f"<tr><td class='mono'>{esc(clean(s.payload_id, 60))}</td>"
            f"<td>{esc(clean(s.category, 30))}</td>"
            f"<td><div class='rt-bar'><div class='track'><div class='fill' style='width:{pct}%;background:{col}'></div></div>"
            f"<span class='pct'>{pct}%</span></div></td>"
            f"<td class='mono' style='font-size:.76rem'>{esc(hits)}</td>"
            f"<td class='goal'>{esc(clean(s.description, 150))}</td></tr>"
        )
    st.markdown(
        "<table class='rt-table'><thead><tr><th>Payload</th><th>Category</th>"
        "<th>Rate across targets</th><th>Targets hit</th><th>Goal</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
else:
    st.info("No successful breakthroughs under this classifier — the target held.")

# ---------------------------------------------------------------------------
# Results explorer
# ---------------------------------------------------------------------------
section("All results", chip="filterable")
targets = sorted({r.run.target for r in results})
categories = sorted({r.run.category.value for r in results})
outcomes = sorted({primary_outcome(r) for r in results})

f1, f2, f3 = st.columns(3)
sel_targets = f1.multiselect("Target", targets, default=targets)
sel_categories = f2.multiselect("Category", categories, default=categories)
sel_outcomes = f3.multiselect("Outcome", outcomes, default=outcomes, format_func=lambda o: OUTCOME_DISPLAY.get(o, o))

rows = []
for r in results:
    outcome = primary_outcome(r)
    if r.run.target in sel_targets and r.run.category.value in sel_categories and outcome in sel_outcomes:
        rows.append(
            {
                "payload": clean(r.payload.id, 60),
                "category": clean(r.run.category.value, 30),
                "target": clean(r.run.target, 50),
                "technique": clean(r.payload.technique or "", 40),
                "outcome": OUTCOME_DISPLAY.get(outcome, outcome),
                "latency (s)": round(r.run.latency_s, 2),
                "response preview": clean(r.run.response or r.run.error or "", 160),
            }
        )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=330)

# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------
section("Successful-attack transcripts")
examples = interesting_successes(results, classifier=classifier, limit=8)
if not examples:
    st.info("No successful attacks to show for this classifier.")
for r in examples:
    with st.expander(f"● {r.payload.id}   ·   {r.run.target}   ·   {r.run.category.value}"):
        st.markdown(f"**Attack goal** — {r.payload.description}")
        st.markdown("**Prompt sent**")
        st.code(r.run.prompt_sent, language="text")
        st.markdown("**Model response**")
        st.code(r.run.response or "(empty)", language="text")
        for v in r.verdicts:
            st.markdown(f"- `{v.classifier}` → **{v.outcome.value}** — {v.justification}")
