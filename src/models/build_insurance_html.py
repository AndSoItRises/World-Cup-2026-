"""
Build outputs/insurance_tracker.html — the cognitive, accessible view of the
underdog +0.5 insurance strategy (DL-18).

Reads data/processed/insurance_summary.json and renders a single self-contained,
mobile-friendly dark page: a plain-English explainer, KPI cards for the three
bankroll streams, the three equity curves, and per-pick recommendation cards with
tier badges, edges, stakes and a one-line rationale. Data is embedded inline so the
file opens with no server; Chart.js is loaded from CDN (same as the main dashboard).

Run:  python -m src.models.build_insurance_html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SUMMARY = BASE / "data" / "processed" / "insurance_summary.json"
OUT = BASE / "outputs" / "insurance_tracker.html"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC2026 — Underdog +0.5 Insurance Tracker</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{--bg:#0c0f14;--card:#151a22;--ln:#222b38;--tx:#e6edf3;--mut:#8b98a8;
        --grn:#3fb950;--red:#f85149;--yel:#d29922;--blu:#58a6ff;--vio:#a371f7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);
       font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:18px}
  h1{font-size:21px;margin:0 0 2px} h2{font-size:16px;margin:26px 0 10px;color:var(--blu)}
  .sub{color:var(--mut);font-size:13px;margin-bottom:14px}
  .card{background:var(--card);border:1px solid var(--ln);border-radius:10px;padding:14px 16px;margin-bottom:12px}
  .explain{font-size:14px}.explain b{color:var(--tx)} .explain code{background:#0a0d12;padding:1px 5px;border-radius:4px;color:var(--vio)}
  .kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  @media(max-width:620px){.kpis{grid-template-columns:1fr}}
  .kpi{background:var(--card);border:1px solid var(--ln);border-radius:10px;padding:12px}
  .kpi .lab{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  .kpi .big{font-size:26px;font-weight:700;margin:4px 0}
  .kpi .row{font-size:12px;color:var(--mut)}
  .pos{color:var(--grn)} .neg{color:var(--red)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:7px 8px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap}
  th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}
  .badge{padding:1px 7px;border-radius:20px;font-size:11px;font-weight:700}
  .b-dog{background:rgba(163,113,247,.15);color:var(--vio)} .b-toss{background:rgba(210,153,34,.15);color:var(--yel)}
  .won{color:var(--grn);font-weight:700}.lost{color:var(--red);font-weight:700}.pend{color:var(--mut)}
  .rat{color:var(--mut);font-size:12px;white-space:normal}
  .note{color:var(--mut);font-size:12px;border-left:3px solid var(--yel);padding:6px 10px;margin-top:8px}
  .wrap{overflow-x:auto}
</style>
</head>
<body>
<h1>Underdog <span style="color:var(--vio)">+0.5</span> Insurance Tracker</h1>
<div class="sub" id="sub"></div>

<div class="card explain">
  <b>What this is.</b> When the model backs an underdog (say <b>Senegal over Argentina</b>),
  we place two bets on the same game:
  <ul style="margin:8px 0">
    <li><b>ML</b> — Senegal to <b>win</b> (pays only if Senegal wins).</li>
    <li><b>+0.5</b> — Senegal to <b>win or draw</b> — it cashes whenever Argentina
        does <b>not</b> win. This is the "insurance": it catches the draw the moneyline throws away.</li>
  </ul>
  Both bets share the "Senegal wins" outcome, so they're <b>correlated</b>. We size them
  <b>together</b> with <b>joint Kelly</b> (not separately — that would over-bet the shared win),
  then apply the desk's usual <code>½-Kelly</code> + <code>5% cap</code>.
  <div style="margin-top:8px">We track <b>three</b> bankrolls so we can see, honestly, whether the
  insurance helps: <b>ML-only</b>, <b>+0.5-only</b>, and <b>Combined</b> (what we'd actually run).</div>
</div>

<h2>Bankroll — 3 strategies compared</h2>
<div class="kpis" id="kpis"></div>
<div class="card"><canvas id="chart" height="150"></canvas></div>

<h2>Recommendations</h2>
<div class="wrap"><table id="tbl">
  <thead><tr>
    <th>Date</th><th>Pick</th><th>vs</th><th>Tier</th><th>Mkt&nbsp;win</th>
    <th>Edge&nbsp;ML</th><th>Edge&nbsp;+0.5</th><th>Odds&nbsp;ML</th><th>Odds&nbsp;+0.5</th>
    <th>Stake&nbsp;ML</th><th>Stake&nbsp;+0.5</th><th>ML</th><th>+0.5</th><th>Why</th>
  </tr></thead><tbody></tbody>
</table></div>

<div class="note" id="caveat"></div>

<script>
const DATA = __DATA__;
const f = (x,d=1)=>Number(x).toFixed(d);
const pct = x => (x*100).toFixed(1)+'%';
const cls = v => v>=0?'pos':'neg';

// subtitle
document.getElementById('sub').textContent =
  `${DATA.n_recommendations} recommendations · ${DATA.n_settled} settled · ${DATA.n_open} open`
  + ` · starting bankroll ${DATA.config.bank0} units · ½-Kelly, ${(DATA.config.cap*100)}% cap per leg`;

// KPI cards
const order=['ml_only','plus_half','combined'];
document.getElementById('kpis').innerHTML = order.map(k=>{
  const s=DATA.streams[k];
  return `<div class="kpi">
     <div class="lab">${s.label}</div>
     <div class="big ${cls(s.roi_pct)}">${s.roi_pct>=0?'+':''}${f(s.roi_pct)}%</div>
     <div class="row">bank ${f(s.final_bankroll)} u · ${s.n_bets} bets · ${f(s.win_rate)}% win</div>
     <div class="row">max drawdown ${f(s.max_drawdown_pct)}%</div>
   </div>`;
}).join('');

// equity chart (align all curves on the combined stream's settlement order)
const base = DATA.streams.combined.curve.map(p=>p.date);
const mk = (k,color)=>({label:DATA.streams[k].label,
   data:DATA.streams[k].curve.map(p=>p.bank),
   borderColor:color,backgroundColor:color,tension:.2,pointRadius:2,borderWidth:2});
new Chart(document.getElementById('chart'),{type:'line',
  data:{labels:base,datasets:[mk('ml_only','#58a6ff'),mk('plus_half','#a371f7'),mk('combined','#3fb950')]},
  options:{responsive:true,plugins:{legend:{labels:{color:'#e6edf3'}}},
    scales:{x:{ticks:{color:'#8b98a8'},grid:{color:'#1d2530'}},
            y:{ticks:{color:'#8b98a8'},grid:{color:'#1d2530'},title:{display:true,text:'bankroll (units)',color:'#8b98a8'}}}}});

// recommendation table — settled first, then open; newest dates near top within group
const rows=[...DATA.ledger].sort((a,b)=> (b.settled-a.settled) || a.date.localeCompare(b.date) || a.match_id-b.match_id);
const st = (v)=> v==='WON'?'<span class="won">WON</span>': v==='LOST'?'<span class="lost">LOST</span>':'<span class="pend">—</span>';
const stake = fr => fr>0 ? (fr*100).toFixed(1)+'%' : '—';
document.querySelector('#tbl tbody').innerHTML = rows.map(r=>{
  const badge = r.tier==='BIG DOG'?'<span class="badge b-dog">BIG DOG</span>':'<span class="badge b-toss">TOSS-UP</span>';
  const settledMark = r.settled? '' : ' style="opacity:.72"';
  return `<tr${settledMark}>
    <td>${r.date}</td><td>${r.selection}</td><td>${r.opponent}</td><td>${badge}</td>
    <td>${pct(r.market_implied_win)}</td>
    <td class="${cls(r.edge_ml)}">${r.edge_ml>=0?'+':''}${pct(r.edge_ml)}</td>
    <td class="${cls(r.edge_dc)}">${r.edge_dc>=0?'+':''}${pct(r.edge_dc)}</td>
    <td>${f(r.decimal_ml,2)}</td><td>${f(r.decimal_dc,2)}</td>
    <td>${stake(r.f_ml)}</td><td>${stake(r.f_dc)}</td>
    <td>${st(r.status_ml)}</td><td>${st(r.status_dc)}</td>
    <td class="rat">${r.rationale}</td>
  </tr>`;
}).join('');

document.getElementById('caveat').innerHTML =
  '<b>Read honestly:</b> odds are de-vigged (fair) market prices, so P&amp;L is research-grade and '
  + 'slightly optimistic vs a real book. Sample is tiny. The strategy only makes money if the model\'s '
  + 'underdog edge is real — still unproven (CONTEXT_V6 DL-10), being decided by this tournament. '
  + 'The +0.5 leg de-variances an edge; it does not create one.';
</script>
</body>
</html>
"""


def build() -> Path:
    summary = json.loads(SUMMARY.read_text())
    html = TEMPLATE.replace("__DATA__", json.dumps(summary))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> int:
    if not SUMMARY.exists():
        print(f"  {SUMMARY.name} missing — run insurance_tracker first.")
        return 1
    out = build()
    print(f"  wrote {out.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
