"""
V6 Phase 3 — Quant research dashboard generator.

Renders outputs/quant_dashboard.html: a single self-contained file (no server,
no CDN — works offline from file://) with all V6 data embedded as JSON at build
time. Re-run this after live_update.py / fetch_live_odds.py / bet_sim.py to
refresh the embedded data.

Panels: live ticker (in-browser ESPN refresh w/ offline fallback), value bet
scanner, match projections, groups, bracket/stage probs, model-vs-market
scatter, bankroll simulator (1k Kelly paths in JS), line movement + arb,
uncertainty (aleatoric proxy now, epistemic = phase 4), and the V6 context /
next-steps doc embedded as a panel (so the research state travels with the file).

Run: python -m src.models.build_dashboard
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"
OUT_PATH = BASE / "outputs" / "quant_dashboard.html"
HANDOFF_PATH = BASE / "HANDOFF.md"

# Fallback only — the NOTES tab embeds HANDOFF.md (single source of truth)
NEXT_STEPS = """\
V6 BUILD STATE (this file doubles as the context doc — see CONTEXT_V6.md in repo)

DONE
  1. market_ingestion.py — odds parsing, Shin de-vig, name audits
  2. bet_sim.py — edge / EV / Kelly (1/4, 5% cap), futures + matches
  +  fetch_live_odds.py — ESPN/DraftKings 3-way lines (open + current), 72/72 matched
  +  market_monitor.py — line movement vs model, cross-book arb scanner
  +  desk_call.py — BET/LEAN/PASS verdicts w/ evidence chains, bias haircuts,
     25% portfolio exposure cap (the DESK tab)
  3. quant_dashboard.html — this file

NEXT STEPS (in order)
  4. uncertainty.py — epistemic layer: per-match XGB vs LGBM vs DC disagreement
     (components must be saved per match; aleatoric entropy proxy already lives
     in the Uncertainty tab). Bet-sizing haircut for high-disagreement matches.
  5. clv_tracker.py — after matches settle: flagged bet vs closing line → CLV.
     Opening lines are already being archived per fetch; closing = last snapshot
     before kickoff.
  6. Signal tests (orthogonality gate): tournament pressure, manager tenure,
     travel/timezone, suspension exposure, confed ELO deflation.
  7. More books in wc2026_match_odds.csv → real arb scanning + best-line EV.
  8. Futures by stage (group winner / R16 / QF / SF) vs market when priced.
  9. Bracket sim re-run button + P10/P50/P90 bands (needs sim-distribution dump
     from live_update.py, ~trivial to add).

REFRESH RITUAL (PowerShell, venv active, PYTHONUTF8=1)
  python -m src.models.live_update        # after results land
  python -m src.models.fetch_live_odds    # any time (appends line history)
  python -m src.models.market_monitor
  python -m src.models.bet_sim
  python -m src.models.build_dashboard    # regenerates this file

STANDING CAVEATS (do not delete)
  - Model accuracy 62%, edge vs market UNPROVEN out-of-sample (V5 DL-05).
  - Draw probs are upweighted 1.75x by design -> draw/'dog "edges" in lopsided
    matches are mostly MODEL BIAS. Favorite-side edges are more trustworthy.
  - CONCACAF inflation: Mexico ~+6pp vs market is documented model error.
  - Futures with model_prob < 2% (<200 of 10k sims) = MC tail noise, flagged."""


def load(name, proc=True):
    p = (DATA_PROC if proc else DATA_RAW) / name
    if not p.exists():
        return pd.DataFrame()
    # An empty-but-present file (e.g. line_movement.csv on a no-movement cycle)
    # must NOT crash the whole build — degrade to an empty frame. Otherwise a
    # single 0-byte input freezes the dashboard while CSVs keep committing.
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_payload() -> dict:
    fixtures = load("wc2026_fixtures.csv", proc=False)
    results = load("wc2026_live_results.csv", proc=False)
    group_fix = fixtures[fixtures["stage"] == "Group Stage"]

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "model": "V4 ensemble (XGB .275 / LGBM .275 / DC .45) + V6 calibrator (log-pool · draw×0.871)",
            "accuracy": "62.0% | LL 0.8405 calibrated | model-market corr 0.84",
            "devig": "Shin",
        },
        "nextSteps": (HANDOFF_PATH.read_text(encoding="utf-8")
                      if HANDOFF_PATH.exists() else NEXT_STEPS),
        "desk": load("desk_calls.csv").fillna("").to_dict("records"),
        "futures": load("value_bets_futures.csv").to_dict("records"),
        "bets": load("value_bets.csv").to_dict("records"),
        "implied": load("market_implied_probs.csv").to_dict("records"),
        "tourney": load("tournament_probs_live.csv").to_dict("records"),
        "movement": load("line_movement.csv").to_dict("records"),
        "clv": load("clv_report.csv").fillna("").to_dict("records"),
        "scoreboard": load("prediction_scoreboard.csv").fillna("").to_dict("records"),
        "arb": load("arb_scan.csv").to_dict("records"),
        "results": results.to_dict("records"),
        "groups": {g: sorted(set(d["home_team"]) | set(d["away_team"]))
                   for g, d in group_fix.groupby("group")},
    }
    return payload


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>WC2026 Quant Desk — V6</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b0e14;--panel:#11151f;--line:#1e2433;--tx:#c8d0e0;--dim:#6b7488;
--grn:#3ddc84;--red:#ff5566;--amb:#ffb347;--cyn:#4dc3ff;--mono:'Consolas','Menlo',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:var(--mono);font-size:13px}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;color:#fff}header .sub{color:var(--dim);font-size:11px}
#ticker-wrap{background:#070a10;border-bottom:1px solid var(--line);overflow:hidden;
white-space:nowrap;padding:6px 0;position:relative}
#ticker{display:inline-block;padding-left:100%;animation:tick 240s linear infinite}
@keyframes tick{0%{transform:translateX(0)}100%{transform:translateX(-100%)}}
#ticker span{margin-right:34px}
.up{color:var(--grn)}.dn{color:var(--red)}.fl{color:var(--dim)}
nav{display:flex;gap:2px;padding:8px 12px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
nav button{background:var(--panel);border:1px solid var(--line);color:var(--dim);
padding:6px 12px;cursor:pointer;font-family:var(--mono);font-size:12px}
nav button.on{color:#fff;border-color:var(--cyn)}
main{padding:14px;max-width:1500px;margin:0 auto}
.tab{display:none}.tab.on{display:block}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--dim);cursor:pointer;user-select:none;position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child{text-align:left}
tr:hover td{background:#161b28}
.pos{color:var(--grn)}.neg{color:var(--red)}.warn{color:var(--amb)}
.flag{font-size:10px;padding:1px 5px;border:1px solid}
.flag.strong{color:var(--grn);border-color:var(--grn)}
.flag.value{color:var(--cyn);border-color:var(--cyn)}
.flag.tail{color:var(--amb);border-color:var(--amb)}
.card{background:var(--panel);border:1px solid var(--line);padding:12px;margin-bottom:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.bar{height:10px;background:#1c2330;position:relative;margin:2px 0}
.bar i{position:absolute;height:100%;background:var(--cyn);opacity:.85}
.bar i.mk{background:var(--amb);height:3px;top:7px;opacity:1}
.note{color:var(--dim);font-size:11px;margin:8px 0;line-height:1.5}
.cav{border-left:3px solid var(--amb);padding:8px 12px;background:#161208;
color:#d8c9a0;font-size:11px;margin:10px 0;line-height:1.6}
select,input[type=number]{background:var(--panel);color:var(--tx);
border:1px solid var(--line);padding:4px 6px;font-family:var(--mono)}
label{color:var(--dim);font-size:11px;margin-right:10px}
button.act{background:#0e2233;border:1px solid var(--cyn);color:var(--cyn);
padding:5px 14px;cursor:pointer;font-family:var(--mono)}
pre{white-space:pre-wrap;line-height:1.55;font-size:12px;color:#b8c2d8}
svg text{font-family:var(--mono)}
h2{font-size:14px;color:#fff;margin-bottom:8px}h3{font-size:12px;color:var(--cyn);margin:6px 0}
.kpi{display:inline-block;margin-right:22px}.kpi b{color:#fff;font-size:15px}
.kpi span{color:var(--dim);font-size:10px;display:block}
#live-status{font-size:11px;color:var(--dim);margin-left:10px}
.vb{font-weight:bold;font-size:11px;padding:2px 8px;border-radius:2px}
.vb.bet{background:#0c2e1c;color:var(--grn);border:1px solid var(--grn)}
.vb.lean{background:#0c2330;color:var(--cyn);border:1px solid var(--cyn)}
.deskcard{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--grn);
padding:12px 14px;margin-bottom:10px}
.deskcard.lean{border-left-color:var(--cyn)}
.deskcard .head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.deskcard .sel{color:#fff;font-size:14px;font-weight:bold}
.deskcard .stake{margin-left:auto;color:var(--grn);font-size:15px;font-weight:bold}
.deskcard ul{list-style:none;margin-top:4px}
.deskcard li{font-size:11.5px;line-height:1.55;color:#aab4c8}
.deskcard li.plus:before{content:"+ ";color:var(--grn)}
.deskcard li.minus:before{content:"! ";color:var(--amb)}
/* iOS / mobile: momentum scrolling, no auto text inflation, fluid tables */
html{-webkit-text-size-adjust:100%}
.scroll{-webkit-overflow-scrolling:touch;overflow:auto}
nav button{min-height:34px}
@media(max-width:760px){
  main{padding:8px}header{padding:8px 10px;gap:8px}
  .cards{grid-template-columns:1fr}
  table{font-size:11px}th,td{padding:4px 5px}
  .kpi{margin-right:14px}.kpi b{font-size:13px}
  .deskcard .stake{margin-left:0;width:100%}
}
</style></head><body>
<header>
  <h1>WC2026 QUANT DESK</h1>
  <span class="sub" id="meta-sub"></span>
  <button class="act" onclick="liveRefresh()">⟳ LIVE ODDS</button>
  <label style="margin:0"><input type="checkbox" id="auto-live"> auto 4m</label>
  <span id="live-status"></span>
</header>
<div id="ticker-wrap"><div id="ticker"></div></div>
<nav id="nav"></nav>
<main>

<div class="tab" id="tab-desk">
  <h2>Desk Calls — what we'd actually bet, and why</h2>
  <div class="card" style="border-left:3px solid var(--cyn)">
    <h3>How to use this desk</h3>
    <div class="note" style="margin:4px 0">
    <b style="color:var(--grn)">BET</b> = the model's edge survives every documented-bias haircut ·
    <b style="color:var(--cyn)">LEAN</b> = positive but thinner — half conviction ·
    PASS = explained below, not hidden.<br>
    Stakes are ¼-Kelly (5% per-bet cap, 25% whole-book cap) on a $1,000 bankroll — scale to yours.<br>
    Hunting on your own? <b>SCANNER</b> = every +EV price right now ·
    <b>GROUPS / BRACKET</b> = advancement &amp; winner odds vs fair price ·
    <b>MOVEMENT+ARB</b> = where the lines are going and where arbitrage would show ·
    <b>CLV</b> = the desk's scorecard vs closing lines (the honest test of edge).<br>
    Press <b>⟳ LIVE ODDS</b> (or tick auto) — odds, EV and these verdicts recompute on
    the live DraftKings number.</div>
  </div>
  <div class="card" style="border-left:3px solid var(--grn)">
    <h3>Next 5 games — what the model says</h3>
    <div id="next5"></div>
    <div class="note">desk score = conviction after every documented-bias haircut
    (BET ≥ 6 · LEAN ≥ 3 · under 3 = PASS) · cyan bar = model prob, amber tick = market
    fair · recomputes with ⟳ LIVE ODDS.</div>
  </div>
  <div class="note" id="desk-live-note"></div>
  <div id="desk-kpis" style="margin:10px 0"></div>
  <div id="desk-cards"></div>
  <div class="card"><h3 id="desk-pass-head"></h3><div class="note" id="desk-pass-note"></div></div>
  <div class="card">
    <h3>Model record — every prediction stored, scored when it settles</h3>
    <div id="record-kpis" style="margin:6px 0"></div>
    <div class="scroll" style="max-height:34vh"><table id="record-table"></table></div>
    <div class="note">Predictions are logged ONCE into append-only prediction_ledger.csv
    while the match is still unplayed, then scored on settlement — post-result re-sims
    can't rewrite history. Results already feed the model: live_update re-rates every
    team's ELO after each final, and the next 10k-sim run prices on it. Deeper
    recalibration is gated by prediction_tracker.py's n ≥ 40 reliability check and the
    validate-or-cut bar (DL-09) — refitting on a handful of results is noise-chasing.
    LL = log-loss (lower is better); market column scores the Shin-fair price logged at
    the same moment.</div>
  </div>
  <div class="cav">62% model · edge vs market UNPROVEN out-of-sample (V5 DL-05). These are
  research conclusions, not betting advice.</div>
</div>

<div class="tab" id="tab-scanner">
  <h2>Value Bet Scanner — match 1X2</h2>
  <label><input type="checkbox" id="sc-evpos" checked> EV &gt; 0 only</label>
  <label><input type="checkbox" id="sc-nodraw" checked> hide draw bets (model bias)</label>
  <label><input type="checkbox" id="sc-fav"> favorites only (fair ≥ 40%)</label>
  <div class="cav">Draw probs are upweighted 1.75× by design and ELO compresses
  lopsided ties — draw/underdog "edges" are mostly documented model bias, not
  market error. Edge is unproven out-of-sample (V5 DL-05). Stakes: ¼-Kelly, 5% cap.</div>
  <div class="scroll" style="max-height:60vh"><table id="sc-table"></table></div>
  <h2 style="margin-top:20px">Tournament Winner Futures</h2>
  <label><input type="checkbox" id="fu-tail" checked> hide tail-risk (model &lt; 2%)</label>
  <div class="scroll" style="max-height:40vh"><table id="fu-table"></table></div>
</div>

<div class="tab" id="tab-matches">
  <h2>Match Projections — model vs market</h2>
  <label>group <select id="mp-group"></select></label>
  <div class="cards" id="mp-cards"></div>
</div>

<div class="tab" id="tab-groups">
  <h2>Group Stage Tracker</h2>
  <div class="note">advance% = top-2 + best-3rd paths from 10k Monte Carlo (live ELO).
  Locked results grey out as they land in wc2026_live_results.csv.</div>
  <div class="cards" id="gr-cards"></div>
</div>

<div class="tab" id="tab-bracket">
  <h2>Bracket — stage probabilities (10k sims, live)</h2>
  <div class="note">P10/P50/P90 bands + in-browser re-sim are queued (next step 9 —
  needs the sim-distribution dump from live_update.py). The interactive bracket
  lives in <a style="color:var(--cyn)" href="bracket_simulator.html">bracket_simulator.html</a>.</div>
  <div class="scroll" style="max-height:70vh"><table id="br-table"></table></div>
</div>

<div class="tab" id="tab-scatter">
  <h2>Model vs Market — tournament winner</h2>
  <div class="note">above diagonal = model likes more than market (green). Dot size ∝ win
  prob. Sqrt scale. Mexico's gap is documented CONCACAF inflation — read with suspicion.</div>
  <div id="scatter"></div>
</div>

<div class="tab" id="tab-bankroll">
  <h2>Bankroll Simulator — 1,000 Kelly paths</h2>
  <label>bankroll $<input type="number" id="bk-start" value="1000" style="width:80px"></label>
  <label>kelly <select id="bk-frac"><option value="0.25">1/4</option>
    <option value="0.5">1/2</option><option value="1">full</option></select></label>
  <label>stake cap <select id="bk-cap"><option value="0.05">5%</option>
    <option value="0.02">2%</option><option value="0.1">10%</option></select></label>
  <label>min EV <input type="number" id="bk-minev" value="0.05" step="0.01" style="width:60px"></label>
  <label><input type="checkbox" id="bk-nodraw" checked> exclude draws</label>
  <label>truth <select id="bk-truth"><option value="model">model probs</option>
    <option value="blend">50/50 blend</option><option value="market">market fair</option></select></label>
  <button class="act" onclick="runBankroll()">RUN</button>
  <div class="cav">"truth = model" assumes the model is perfectly calibrated — the
  OPTIMISTIC bound. "market fair" assumes the market is right — then every bet is
  −EV by the vig (the pessimistic sanity check). Reality is somewhere between;
  the blend column is the honest middle. Bets resolve independently — same-group
  correlation ignored (parlay-correlation layer is a next step).</div>
  <div id="bk-out"></div>
</div>

<div class="tab" id="tab-monitor">
  <h2>Line Movement (open → current) + Arb Scan</h2>
  <div class="kpi"><b id="mv-sig">–</b><span>significant moves</span></div>
  <div class="kpi"><b id="mv-toward">–</b><span>toward model</span></div>
  <div class="kpi"><b id="mv-against">–</b><span>against model</span></div>
  <div class="kpi"><b id="arb-n">–</b><span>true arbs</span></div>
  <div class="kpi"><b id="arb-books">–</b><span>books feeding</span></div>
  <div class="note">moves toward model = the line travelled in the model's direction
  since open (sharps agreeing); against = market hardening the other way.</div>
  <div class="card">
    <h3>Where is the arbitrage?</h3>
    <div class="note" id="arb-answer"></div>
    <div class="scroll" style="max-height:30vh"><table id="arb-table"></table></div>
  </div>
  <div class="scroll" style="max-height:65vh"><table id="mv-table"></table></div>
</div>

<div class="tab" id="tab-clv">
  <h2>CLV — are the desk calls beating the closing line?</h2>
  <div class="note">Every BET/LEAN is logged once (append-only) with the line taken.
  CLV% = taken ÷ closing − 1. Consistently positive CLV = real edge, long before
  win/loss records mean anything. The dog rows decide DL-10: is the model's market
  disagreement on underdogs edge, or did the market know better? Until a match
  settles its "close" is just the latest snapshot (provisional) — only FINAL closes
  feed back into desk-call scoring (n ≥ 8 per category).</div>
  <div id="clv-kpis" style="margin:10px 0"></div>
  <div class="scroll" style="max-height:62vh"><table id="clv-table"></table></div>
</div>

<div class="tab" id="tab-uncertainty">
  <h2>Uncertainty</h2>
  <h3>Aleatoric — how unpredictable is the match itself (3-way entropy)</h3>
  <div class="note">entropy of the model's [home, draw, away] in bits; max = 1.585
  (perfect 3-way coin-flip). High entropy ⇒ genuinely volatile tie ⇒ size down.</div>
  <div class="scroll" style="max-height:45vh"><table id="un-table"></table></div>
  <div class="cav">Epistemic uncertainty (XGB vs LGBM vs DC disagreement per match)
  is phase 4 — needs per-component probabilities saved at predict time
  (src/models/uncertainty.py, next step 4). Until then: treat tail-risk futures
  and high-entropy matches as the size-down list.</div>
</div>

<div class="tab" id="tab-notes">
  <h2>Agent Handoff — full research state (mirror of HANDOFF.md in the repo)</h2>
  <div class="card"><pre id="notes-pre"></pre></div>
</div>

</main>
<script>
const D = __DATA__;
const fmtP = x => (100*x).toFixed(1)+'%';
const fmtPP = x => (x>=0?'+':'')+(100*x).toFixed(1)+'pp';
const cls = x => x>0.0001?'pos':(x<-0.0001?'neg':'fl');

/* ── nav ── */
const TABS = [['desk','DESK CALLS'],['scanner','SCANNER'],['matches','MATCHES'],['groups','GROUPS'],
 ['bracket','BRACKET'],['scatter','DIVERGENCE'],['bankroll','BANKROLL'],
 ['monitor','MOVEMENT+ARB'],['clv','CLV'],['uncertainty','UNCERTAINTY'],['notes','NOTES']];
const nav = document.getElementById('nav');
TABS.forEach(([id,label],i)=>{
  const b=document.createElement('button');b.textContent=label;b.dataset.t=id;
  b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');document.getElementById('tab-'+id).classList.add('on');};
  nav.appendChild(b);if(i===0)b.click();});
document.getElementById('meta-sub').textContent =
  `${D.meta.model} · ${D.meta.accuracy} · de-vig: ${D.meta.devig} · built ${D.meta.built_at}`;
document.getElementById('notes-pre').textContent = D.nextSteps;

/* ── next 5 games — model board (re-rendered by drawDesk so live odds
      recomputes flow through) ── */
function drawNext5(rows){
  const el=document.getElementById('next5');if(!el)return;
  const done=new Set(D.results.map(r=>+r.match_id));
  const up=D.implied.filter(m=>!done.has(+m.match_id))
    .sort((a,b)=>String(a.date).localeCompare(String(b.date))||(+a.match_id-+b.match_id))
    .slice(0,5);
  el.innerHTML=up.map(m=>{
    const sides=['home','draw','away'].map(s=>{
      const b=D.bets.find(x=>x.match_id===m.match_id&&x.outcome===s)||{};
      return {s,name:s==='draw'?'Draw':(s==='home'?m.home_team:m.away_team),
        p:+b.model_prob||0,fair:+m[s+'_fair_prob']||0,edge:+b.edge||0};});
    const pick=[...sides].sort((a,b)=>b.p-a.p)[0];
    const calls=rows.filter(r=>r.kind==='match'&&+r.match_id===+m.match_id);
    const act=calls.filter(r=>r.verdict!=='PASS').sort((a,b)=>b.score-a.score)[0];
    const top=act||[...calls].sort((a,b)=>b.score-a.score)[0];
    const verdictHtml=act
      ?`<span class="vb ${act.verdict.toLowerCase()}">${act.verdict}</span>
        <b style="color:#fff">${act.selection}</b> @ ${(+act.decimal_odds).toFixed(2)}
        · desk score <b>${(+act.score).toFixed(1)}</b>
        · stake <b class="pos">$${(+act.stake_usd).toFixed(0)}</b>`
      :`<span style="color:var(--dim)">PASS — ${top?String(top.cautions||top.why)
          .split(' | ')[0]:'no priced edge'}</span>`;
    const bars=sides.map(x=>
      `<div style="display:flex;gap:8px;align-items:center">
        <span style="width:130px;overflow:hidden;white-space:nowrap">${x.name}</span>
        <span style="flex:1">${bar(x.p,x.fair)}</span>
        <span style="width:50px">${fmtP(x.p)}</span>
        <span class="${cls(x.edge)}" style="width:58px">${fmtPP(x.edge)}</span></div>`).join('');
    return `<div class="card" style="margin-bottom:8px">
      <h3>#${m.match_id} ${m.home_team} vs ${m.away_team}
        <span style="color:var(--dim);font-weight:normal">· ${m.date}</span></h3>
      ${bars}
      <div style="margin-top:6px;font-size:12px">model pick: <b>${pick.name}</b>
        (${fmtP(pick.p)} vs fair ${fmtP(pick.fair)}) &nbsp;·&nbsp; ${verdictHtml}</div>
    </div>`;
  }).join('')||'<div class="note">no upcoming matches with prices</div>';
}

/* ── desk calls — renderer + a JS mirror of desk_call.py so verdicts
      recompute on live lines (model probs stay fixed; re-sim is the
      local Python pipeline) ── */
function drawDesk(rows,liveNote){
  drawNext5(rows);
  const picks=rows.filter(r=>r.verdict!=='PASS')
    .sort((a,b)=>(a.verdict.localeCompare(b.verdict))||(b.score-a.score));
  const passes=rows.filter(r=>r.verdict==='PASS');
  const nBet=picks.filter(r=>r.verdict==='BET').length;
  const total=picks.reduce((s,r)=>s+(+r.stake_usd||0),0);
  document.getElementById('desk-live-note').innerHTML=liveNote||'';
  document.getElementById('desk-kpis').innerHTML=
    `<span class="kpi"><b class="pos">${nBet}</b><span>BET</span></span>`+
    `<span class="kpi"><b style="color:var(--cyn)">${picks.length-nBet}</b><span>LEAN</span></span>`+
    `<span class="kpi"><b>${passes.length}</b><span>PASS</span></span>`+
    `<span class="kpi"><b>$${total.toFixed(0)}</b><span>total book (of $1,000)</span></span>`;
  const el=document.getElementById('desk-cards');el.innerHTML='';
  picks.forEach(r=>{
    const c=document.createElement('div');
    c.className='deskcard'+(r.verdict==='LEAN'?' lean':'');
    const why=String(r.why).split(' | ').filter(Boolean)
      .map(w=>`<li class="plus">${w}</li>`).join('');
    const cau=String(r.cautions).split(' | ').filter(Boolean)
      .map(w=>`<li class="minus">${w}</li>`).join('');
    c.innerHTML=`<div class="head"><span class="vb ${r.verdict.toLowerCase()}">${r.verdict}</span>
      <span class="sel">${r.selection}</span>
      <span style="color:var(--dim)">${r.label}${r.date?' · '+r.date:''}
      · ${(+r.decimal_odds).toFixed(2)}</span>
      <span class="stake">$${(+r.stake_usd).toFixed(0)}</span></div>
      <ul>${why}${cau}</ul>`;
    el.appendChild(c);});
  const reasons={};
  passes.forEach(r=>{const k=String(r.why||r.cautions).split(' | ')[0].split(' — ')[1]||
    String(r.why).split(' — ')[0];reasons[k]=(reasons[k]||0)+1;});
  document.getElementById('desk-pass-head').textContent=
    `${passes.length} PASSED — why nothing else made the cut`;
  document.getElementById('desk-pass-note').innerHTML=Object.entries(reasons)
    .sort((a,b)=>b[1]-a[1]).slice(0,6)
    .map(([k,v])=>`${v}× ${k}`).join('<br>')+
    '<br><br>Full detail per bet: SCANNER tab (uncheck the filters to see everything).';
}
drawDesk(D.desk);

/* constants + rules mirror src/models/desk_call.py — keep in sync */
const DK={BET:6,LEAN:3,ENT:1.5,
  CONCACAF:new Set(['Mexico','USA','Canada','Panama','Haiti','Curacao','Curaçao'])};
const liveMove={};   // `${mid}|${outcome}` → {significant,toward,open,now} from live fetch
function embMove(mid,outcome){
  const r=D.movement.find(x=>x.match_id===mid&&x.outcome===outcome);
  if(!r)return null;
  return {significant:r.significant===true||r.significant==='True',
    toward:r.direction_vs_model==='toward',open:+r.open_decimal,now:+r.now_decimal};
}
function entropyOf(mid){
  const p=['home','draw','away'].map(s=>{
    const b=D.bets.find(x=>x.match_id===mid&&x.outcome===s);
    return Math.max(b?+b.model_prob:1e-9,1e-9);});
  return -p.reduce((a,x)=>a+x*Math.log2(x),0);
}
function clvAdjJS(){
  const fin=D.clv.filter(r=>(r.close_is_final===true||r.close_is_final==='True')
    &&r.clv_pct!==''&&r.clv_pct!==null);
  const adj={};
  ['fav','dog','draw'].forEach(cat=>{
    const g=fin.filter(r=>r.category===cat);
    if(g.length<8)return;
    const avg=g.reduce((s,r)=>s+(+r.clv_pct),0)/g.length;
    if(avg>=2)adj[cat]=[1.5,`desk's ${cat} calls are beating final closes (${avg>=0?'+':''}${avg.toFixed(1)}% avg CLV, n=${g.length}) — market confirms this lane`];
    else if(avg<=-2)adj[cat]=[-1.5,`desk's ${cat} calls are losing to final closes (${avg.toFixed(1)}% avg CLV, n=${g.length}) — market keeps beating us here`];
  });
  return adj;
}
function deskCallJS(b,mv,ent,clvAdj){
  const why=[],cau=[];
  if(b.ev<=0)return{verdict:'PASS',score:0,sizeDown:false,
    why:['negative EV — the price is better than the model'],cau:[]};
  if(b.outcome==='draw')return{verdict:'PASS',score:0,sizeDown:false,
    why:["draw bet — model upweights draws 1.75× by design; this 'edge' is model bias"],cau:[]};
  let score=Math.min(b.edge*100,10)*0.6;
  why.push(`model ${(b.model_prob*100).toFixed(0)}% vs market fair ${(b.market_implied*100).toFixed(0)}% = +${(b.edge*100).toFixed(1)}pp edge, EV ${b.ev>=0?'+':''}${(+b.ev).toFixed(2)}/$1 at ${(+b.decimal_odds).toFixed(2)}`);
  if(b.market_implied>=0.40){score+=2;
    why.push('favorite side — the zone where the model is most trustworthy');}
  else if(b.market_implied<0.15){score-=3;
    cau.push('longshot — ELO compression inflates underdog probs (model bias)');}
  if(mv&&mv.significant){
    if(mv.toward){score+=2;
      why.push(`line moved TOWARD model since open (${mv.open.toFixed(2)}→${mv.now.toFixed(2)}) — sharp money on our side of the number`);}
    else{score-=2;
      cau.push(`line moved AGAINST model since open (${mv.open.toFixed(2)}→${mv.now.toFixed(2)}) — the market is hardening the other way`);}
  }
  const sizeDown=ent>DK.ENT;
  if(sizeDown)cau.push(`coin-flip match (entropy ${ent.toFixed(2)} bits) — stake halved`);
  if(DK.CONCACAF.has(b.selection)){score-=2;
    cau.push('CONCACAF selection — model inflation documented (Mexico ~+6pp); edge partly model error');}
  const cat=b.market_implied>=0.40?'fav':'dog';
  if(clvAdj[cat]){const[pts,line]=clvAdj[cat];score+=pts;(pts>0?why:cau).push(line);}
  let verdict=score>=DK.BET?'BET':score>=DK.LEAN?'LEAN':'PASS';
  if(verdict==='PASS')cau.push('signal too weak after bias haircuts');
  return{verdict,score:Math.round(score*100)/100,sizeDown,why,cau};
}
function recomputeDesk(){
  const clvAdj=clvAdjJS();
  const rows=[];
  D.bets.filter(b=>b.market_source==='real').forEach(b=>{
    const mv=liveMove[`${b.match_id}|${b.outcome}`]||embMove(b.match_id,b.outcome);
    const r=deskCallJS(b,mv,entropyOf(b.match_id),clvAdj);
    const raw=(Math.min(+b.kelly_quarter||0,0.05)*1000)*(r.sizeDown?0.5:1);
    rows.push({kind:'match',match_id:b.match_id,outcome:b.outcome,date:b.date,
      label:b.match,selection:`${b.selection} (${b.outcome})`,
      verdict:r.verdict,score:r.score,model_prob:b.model_prob,
      market_implied:b.market_implied,edge:b.edge,ev:b.ev,
      decimal_odds:b.decimal_odds,
      stake_raw_usd:r.verdict==='PASS'?0:raw,
      why:r.why.join(' | '),cautions:r.cau.join(' | ')});
  });
  D.desk.filter(r=>r.kind==='futures').forEach(r=>
    rows.push({...r,stake_raw_usd:+r.stake_raw_usd||+r.stake_usd||0}));
  const rawTotal=rows.filter(r=>r.verdict!=='PASS')
    .reduce((s,r)=>s+(+r.stake_raw_usd||0),0);
  const k=rawTotal>250?250/rawTotal:1;   // 25% exposure cap on $1,000
  rows.forEach(r=>r.stake_usd=r.verdict==='PASS'?0:(+r.stake_raw_usd||0)*k);
  return rows;
}

/* ── sortable tables ── */
function renderTable(el, cols, rows){
  let sortK=null,asc=false;
  function draw(){
    const rs=[...rows];
    if(sortK!==null)rs.sort((a,b)=>{const x=a[sortK],y=b[sortK];
      return (typeof x==='number'?(x-y):String(x).localeCompare(String(y)))*(asc?1:-1);});
    el.innerHTML='<tr>'+cols.map(c=>`<th data-k="${c.k}">${c.h}</th>`).join('')+'</tr>'+
      rs.map(r=>'<tr>'+cols.map(c=>`<td class="${c.c?c.c(r):''}">${c.f(r)}</td>`).join('')+'</tr>').join('');
    el.querySelectorAll('th').forEach(th=>th.onclick=()=>{
      asc=(sortK===th.dataset.k)?!asc:false;sortK=th.dataset.k;draw();});
  }
  draw();
}
const flagHtml=(ev,tail)=>tail?'<span class="flag tail">TAIL</span>':
  ev>0.05?'<span class="flag strong">STRONG</span>':ev>0?'<span class="flag value">value</span>':'';

/* ── scanner ── */
function drawScanner(){
  const evpos=document.getElementById('sc-evpos').checked;
  const nodraw=document.getElementById('sc-nodraw').checked;
  const fav=document.getElementById('sc-fav').checked;
  let rows=D.bets.filter(r=>r.market_source==='real');
  if(evpos)rows=rows.filter(r=>r.ev>0);
  if(nodraw)rows=rows.filter(r=>r.outcome!=='draw');
  if(fav)rows=rows.filter(r=>r.market_implied>=0.4);
  renderTable(document.getElementById('sc-table'),[
    {k:'date',h:'date',f:r=>r.date},
    {k:'match',h:'match',f:r=>r.match},
    {k:'selection',h:'bet',f:r=>`${r.selection} (${r.outcome})`},
    {k:'model_prob',h:'model',f:r=>fmtP(r.model_prob)},
    {k:'market_implied',h:'fair',f:r=>fmtP(r.market_implied)},
    {k:'edge',h:'edge',f:r=>fmtPP(r.edge),c:r=>cls(r.edge)},
    {k:'decimal_odds',h:'odds',f:r=>r.decimal_odds.toFixed(2)},
    {k:'ev',h:'EV/$1',f:r=>r.ev.toFixed(3),c:r=>cls(r.ev)},
    {k:'kelly_quarter',h:'K/4',f:r=>fmtP(r.kelly_quarter)},
    {k:'recommended_stake_usd',h:'stake$',f:r=>r.recommended_stake_usd.toFixed(0)},
    {k:'ev',h:'',f:r=>flagHtml(r.ev,false)},
  ],rows);
}
['sc-evpos','sc-nodraw','sc-fav'].forEach(id=>
  document.getElementById(id).onchange=drawScanner);
drawScanner();
function drawFutures(){
  const hideTail=document.getElementById('fu-tail').checked;
  let rows=D.futures.filter(r=>!hideTail||!r.tail_risk);
  renderTable(document.getElementById('fu-table'),[
    {k:'selection',h:'team',f:r=>r.selection},
    {k:'model_prob',h:'model',f:r=>fmtP(r.model_prob)},
    {k:'market_implied',h:'fair',f:r=>fmtP(r.market_implied)},
    {k:'edge',h:'edge',f:r=>fmtPP(r.edge),c:r=>cls(r.edge)},
    {k:'american_odds',h:'amer',f:r=>r.american_odds},
    {k:'ev',h:'EV/$1',f:r=>r.ev.toFixed(3),c:r=>cls(r.ev)},
    {k:'kelly_quarter',h:'K/4',f:r=>fmtP(r.kelly_quarter)},
    {k:'recommended_stake_usd',h:'stake$',f:r=>r.recommended_stake_usd.toFixed(2)},
    {k:'ev',h:'',f:r=>flagHtml(r.ev,r.tail_risk)},
  ],rows);
}
document.getElementById('fu-tail').onchange=drawFutures;drawFutures();

/* ── matches ── */
const mpSel=document.getElementById('mp-group');
mpSel.innerHTML='<option value="">all</option>'+
  Object.keys(D.groups).map(g=>`<option>${g}</option>`).join('');
mpSel.onchange=drawMatches;
function bar(p,mk){return `<div class="bar"><i style="width:${p*100}%"></i>`+
  `<i class="mk" style="width:${mk*100}%"></i></div>`;}
function drawMatches(){
  const g=mpSel.value;
  const el=document.getElementById('mp-cards');el.innerHTML='';
  D.implied.filter(m=>!g||m.group===g).forEach(m=>{
    const bet3=D.bets.filter(b=>b.match_id===m.match_id);
    const card=document.createElement('div');card.className='card';
    let html=`<h3>#${m.match_id} ${m.home_team} vs ${m.away_team}</h3>
      <div class="note">${m.date} · grp ${m.group} · ${m.market_source==='real'
        ?m.book+' ('+m.snapshot+')':'<span class="warn">model-estimated</span>'}</div>`;
    [['home',m.home_team],['draw','Draw'],['away',m.away_team]].forEach(([side,name])=>{
      const b=bet3.find(x=>x.outcome===side)||{};
      const mp=b.model_prob??0, fp=m[side+'_fair_prob'];
      html+=`<div style="display:flex;gap:8px;align-items:center">
        <span style="width:130px;overflow:hidden;white-space:nowrap">${name}</span>
        <span style="flex:1">${bar(mp,fp)}</span>
        <span style="width:54px">${fmtP(mp)}</span>
        <span class="${cls(b.edge??0)}" style="width:60px">${fmtPP(b.edge??0)}</span>
        <span style="width:46px;color:var(--dim)">${(m[side+'_decimal_odds']).toFixed(2)}</span></div>`;});
    html+=`<div class="note">cyan bar = model · amber tick = market fair · edge = model − fair</div>`;
    card.innerHTML=html;el.appendChild(card);});
}
drawMatches();

/* ── groups ── */
(function(){
  const el=document.getElementById('gr-cards');
  const tp=Object.fromEntries(D.tourney.map(t=>[t.team,t]));
  const alias={'South Korea':'Korea Republic','Bosnia and Herzegovina':'Bosnia-Herzegovina',
    'Ivory Coast':"Côte d'Ivoire",'Cape Verde':'Cape Verde Islands','Curacao':'Curaçao',
    'Turkey':'Türkiye'};
  const done=new Set(D.results.map(r=>r.match_id));
  Object.entries(D.groups).forEach(([g,teams])=>{
    const card=document.createElement('div');card.className='card';
    let rows=teams.map(t=>{const r=tp[t]||tp[alias[t]]||{};return {t,adv:r.p_group_adv??0,win:r.p_winner??0};})
      .sort((a,b)=>b.adv-a.adv);
    card.innerHTML=`<h3>Group ${g}</h3>`+rows.map(r=>
      `<div style="display:flex;gap:8px;align-items:center">
       <span style="width:150px;overflow:hidden;white-space:nowrap">${r.t}</span>
       <span style="flex:1">${bar(r.adv,0)}</span>
       <span style="width:52px">${fmtP(r.adv)}</span>
       <span style="width:50px;color:var(--dim)">@${r.adv>0.001?(1/r.adv).toFixed(2):'–'}</span>
       <span style="width:60px;color:var(--dim)">W ${fmtP(r.win)}</span></div>`).join('')+
      `<div class="note">advance% · @fair decimal odds to advance (beat this price = value) · W = win tournament</div>`;
    el.appendChild(card);});
})();

/* ── bracket table ── */
renderTable(document.getElementById('br-table'),[
  {k:'team',h:'team',f:r=>r.team},{k:'fifa_rank',h:'rank',f:r=>r.fifa_rank},
  ...['p_group_adv','p_r16','p_quarterfinal','p_semifinal','p_final','p_winner']
    .map(k=>({k,h:k.replace('p_','').replace('quarterfinal','QF').replace('semifinal','SF')
      .replace('group_adv','adv'),f:r=>fmtP(r[k]||0)})),
  {k:'p_winner',h:'fair winner odds',f:r=>r.p_winner>0.001?(1/r.p_winner).toFixed(1):'–'},
  {k:'eliminated',h:'out?',f:r=>r.eliminated===true||r.eliminated==='True'?'✖':'',c:()=>'neg'},
],[...D.tourney].sort((a,b)=>b.p_winner-a.p_winner));

/* ── scatter ── */
(function(){
  const W=760,H=560,pad=50;
  const pts=D.futures.filter(f=>!f.tail_risk||f.model_prob>0.005);
  const S=x=>Math.sqrt(x), maxv=S(Math.max(...pts.map(p=>Math.max(p.model_prob,p.market_implied)))*1.15);
  const X=v=>pad+S(v)/maxv*(W-2*pad), Y=v=>H-pad-S(v)/maxv*(H-2*pad);
  let s=`<svg width="${W}" height="${H}" style="background:var(--panel);border:1px solid var(--line)">`;
  s+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(maxv*maxv)}" y2="${Y(maxv*maxv)}" stroke="#2a3040"/>`;
  [0.01,0.05,0.1,0.18].forEach(v=>{s+=`<text x="${X(v)}" y="${H-pad+16}" fill="#6b7488"
    font-size="10" text-anchor="middle">${(v*100)}%</text>
    <text x="${pad-8}" y="${Y(v)+3}" fill="#6b7488" font-size="10" text-anchor="end">${(v*100)}%</text>`;});
  s+=`<text x="${W/2}" y="${H-12}" fill="#6b7488" font-size="11" text-anchor="middle">market fair win% (sqrt)</text>`;
  s+=`<text x="14" y="${H/2}" fill="#6b7488" font-size="11" transform="rotate(-90 14 ${H/2})"
    text-anchor="middle">model win% (sqrt)</text>`;
  pts.forEach(p=>{const r=3+Math.sqrt(p.model_prob)*22, up=p.model_prob>p.market_implied;
    s+=`<circle cx="${X(p.market_implied)}" cy="${Y(p.model_prob)}" r="${r}"
      fill="${up?'#3ddc84':'#ff5566'}" fill-opacity="0.45" stroke="${up?'#3ddc84':'#ff5566'}">
      <title>${p.selection}: model ${fmtP(p.model_prob)} vs market ${fmtP(p.market_implied)} (${fmtPP(p.edge)})</title></circle>`;
    if(p.model_prob>0.035||p.market_implied>0.035)
      s+=`<text x="${X(p.market_implied)+r+2}" y="${Y(p.model_prob)+3}" fill="#c8d0e0" font-size="10">${p.selection}</text>`;});
  document.getElementById('scatter').innerHTML=s+'</svg>';
})();

/* ── bankroll sim ── */
function runBankroll(){
  const start=+document.getElementById('bk-start').value;
  const frac=+document.getElementById('bk-frac').value;
  const cap=+document.getElementById('bk-cap').value;
  const minev=+document.getElementById('bk-minev').value;
  const nodraw=document.getElementById('bk-nodraw').checked;
  const truth=document.getElementById('bk-truth').value;
  let bets=D.bets.filter(b=>b.market_source==='real'&&b.ev>=minev);
  if(nodraw)bets=bets.filter(b=>b.outcome!=='draw');
  bets.sort((a,b)=>a.date.localeCompare(b.date));
  if(!bets.length){document.getElementById('bk-out').innerHTML=
    '<div class="note">no bets pass the filter</div>';return;}
  const N=1000,ends=[];
  for(let i=0;i<N;i++){let bk=start;
    for(const b of bets){
      const pTrue=truth==='model'?b.model_prob:truth==='market'?b.market_implied
        :(b.model_prob+b.market_implied)/2;
      const stake=Math.min(b.kelly_full*frac,cap)*bk;
      if(stake<=0||bk<=1)continue;
      bk+=(Math.random()<pTrue)?stake*(b.decimal_odds-1):-stake;}
    ends.push(bk);}
  ends.sort((a,b)=>a-b);
  const q=p=>ends[Math.floor(p*(N-1))], med=q(0.5);
  const lo=ends[0],hi=ends[N-1],bins=new Array(30).fill(0);
  ends.forEach(e=>bins[Math.min(29,Math.floor((e-lo)/((hi-lo)||1)*30))]++);
  const bw=24,Hh=160,maxb=Math.max(...bins);
  let s=`<div style="margin:10px 0">
    <span class="kpi"><b>${bets.length}</b><span>bets placed</span></span>
    <span class="kpi"><b class="${med>=start?'pos':'neg'}">$${med.toFixed(0)}</b><span>median end</span></span>
    <span class="kpi"><b>${((med/start-1)*100).toFixed(1)}%</b><span>median ROI</span></span>
    <span class="kpi"><b>$${q(0.05).toFixed(0)}</b><span>P5</span></span>
    <span class="kpi"><b>$${q(0.95).toFixed(0)}</b><span>P95</span></span>
    <span class="kpi"><b class="${q(0.05)<start*0.5?'warn':''}">${(100*ends.filter(e=>e<start).length/N).toFixed(0)}%</b><span>paths losing</span></span></div>`;
  s+=`<svg width="${30*bw+60}" height="${Hh+40}" style="background:var(--panel);border:1px solid var(--line)">`;
  bins.forEach((b,i)=>{const h=b/maxb*Hh;
    const x0=lo+i*(hi-lo)/30;
    s+=`<rect x="${30+i*bw}" y="${Hh-h+10}" width="${bw-2}" height="${h}"
      fill="${x0>=start?'#3ddc84':'#ff5566'}" fill-opacity="0.6"><title>$${x0.toFixed(0)}+: ${b}</title></rect>`;});
  s+=`<text x="30" y="${Hh+30}" fill="#6b7488" font-size="10">$${lo.toFixed(0)}</text>
      <text x="${30*bw+20}" y="${Hh+30}" fill="#6b7488" font-size="10" text-anchor="end">$${hi.toFixed(0)}</text></svg>`;
  document.getElementById('bk-out').innerHTML=s;
}

/* ── movement + arb ── */
(function(){
  const mv=D.movement,sig=mv.filter(r=>r.significant===true||r.significant==='True');
  document.getElementById('mv-sig').textContent=sig.length;
  document.getElementById('mv-toward').textContent=sig.filter(r=>r.direction_vs_model==='toward').length;
  document.getElementById('mv-against').textContent=sig.filter(r=>r.direction_vs_model==='against').length;
  const arbs=D.arb.filter(r=>r.arb===true||r.arb==='True');
  document.getElementById('arb-n').textContent=arbs.length;
  const nBooks=D.arb.length?Math.max(...D.arb.map(r=>r.n_books)):0;
  document.getElementById('arb-books').textContent=nBooks;
  // "where is the arbitrage?" — straight answer + the tightest tickets
  const tight=D.implied.filter(m=>m.market_source==='real').map(m=>{
    const sum=1/m.home_decimal_odds+1/m.draw_decimal_odds+1/m.away_decimal_odds;
    return {match:`${m.home_team} vs ${m.away_team}`,date:m.date,sum,
      gap:(sum-1)*100};}).sort((a,b)=>a.sum-b.sum);
  document.getElementById('arb-answer').innerHTML=arbs.length?
    `<b class="pos">${arbs.length} riskless arb${arbs.length>1?'s':''} live</b> — best-line
     Σ(1/odds) &lt; 100% across books. Stake each outcome ∝ 1/odds and the profit is locked
     whatever happens. Details in arb_scan.csv.`:
    `<b>Nowhere yet — and that's the expected answer with ${nBooks||1} book feeding.</b>
     An arb = best available odds on all 3 outcomes summing below 100% implied. One book never
     offers that (the vig guarantees Σ &gt; 100%). The tightest ticket right now is
     <b>${tight.length?tight[0].match:'–'}</b> at Σ ${tight.length?(tight[0].sum*100).toFixed(1):'–'}%
     — still ${tight.length?tight[0].gap.toFixed(1):'–'}pp short of riskless.
     Add a second book's lines to wc2026_match_odds.csv (same schema, any source) and the
     scanner compares best-line across books automatically — that's where arbs appear,
     typically when books disagree after team news. Until then the money is made on
     +EV desk calls, not arbs.`;
  renderTable(document.getElementById('arb-table'),[
    {k:'match',h:'match (tightest first)',f:r=>r.match},
    {k:'date',h:'date',f:r=>r.date},
    {k:'sum',h:'Σ implied',f:r=>(r.sum*100).toFixed(1)+'%'},
    {k:'gap',h:'gap to arb',f:r=>r.gap.toFixed(1)+'pp',c:r=>r.gap<2?'warn':''},
  ],tight.slice(0,10));
  renderTable(document.getElementById('mv-table'),[
    {k:'match',h:'match',f:r=>r.match},{k:'outcome',h:'out',f:r=>r.outcome},
    {k:'open_decimal',h:'open',f:r=>r.open_decimal.toFixed(2)},
    {k:'now_decimal',h:'now',f:r=>r.now_decimal.toFixed(2)},
    {k:'implied_shift',h:'shift',f:r=>fmtPP(r.implied_shift),c:r=>cls(r.implied_shift)},
    {k:'model_prob',h:'model',f:r=>fmtP(r.model_prob)},
    {k:'direction_vs_model',h:'vs model',f:r=>r.direction_vs_model,
     c:r=>r.direction_vs_model==='toward'?'pos':'neg'},
    {k:'significant',h:'sig',f:r=>(r.significant===true||r.significant==='True')?'●':''},
  ],[...mv].sort((a,b)=>Math.abs(b.implied_shift)-Math.abs(a.implied_shift)));
})();

/* ── CLV ── */
(function(){
  const rows=D.clv;
  const withClv=rows.filter(r=>r.clv_pct!==''&&r.clv_pct!==null);
  const settled=rows.filter(r=>r.status!=='pending');
  const avg=withClv.length?withClv.reduce((s,r)=>s+(+r.clv_pct),0)/withClv.length:0;
  const beat=withClv.length?withClv.filter(r=>+r.clv_pct>0).length/withClv.length:0;
  const pnl=settled.reduce((s,r)=>s+(+r.pnl_usd||0),0);
  const dogs=withClv.filter(r=>r.category==='dog');
  const dogAvg=dogs.length?dogs.reduce((s,r)=>s+(+r.clv_pct),0)/dogs.length:0;
  document.getElementById('clv-kpis').innerHTML=
    `<span class="kpi"><b>${rows.length}</b><span>bets tracked</span></span>`+
    `<span class="kpi"><b class="${avg>=0?'pos':'neg'}">${avg>=0?'+':''}${avg.toFixed(2)}%</b><span>avg CLV</span></span>`+
    `<span class="kpi"><b>${(beat*100).toFixed(0)}%</b><span>beating close</span></span>`+
    `<span class="kpi"><b class="${dogAvg>=0?'pos':'neg'}">${dogAvg>=0?'+':''}${dogAvg.toFixed(2)}%</b><span>dog CLV (DL-10)</span></span>`+
    `<span class="kpi"><b>${settled.filter(r=>r.status==='WON').length}-${settled.filter(r=>r.status==='LOST').length}</b><span>record</span></span>`+
    `<span class="kpi"><b class="${pnl>=0?'pos':'neg'}">$${pnl.toFixed(0)}</b><span>realized P&L</span></span>`;
  renderTable(document.getElementById('clv-table'),[
    {k:'date',h:'date',f:r=>r.date},
    {k:'label',h:'match',f:r=>r.label},
    {k:'selection',h:'bet',f:r=>r.selection},
    {k:'verdict',h:'call',f:r=>r.verdict},
    {k:'category',h:'cat',f:r=>r.category},
    {k:'taken_decimal',h:'taken',f:r=>(+r.taken_decimal).toFixed(2)},
    {k:'closing_decimal',h:'close',f:r=>r.closing_decimal?(+r.closing_decimal).toFixed(2):'–'},
    {k:'clv_pct',h:'CLV%',f:r=>r.clv_pct===''?'–':(+r.clv_pct>=0?'+':'')+(+r.clv_pct).toFixed(2)+'%',
     c:r=>r.clv_pct===''?'':cls(+r.clv_pct)},
    {k:'stake_usd',h:'stake$',f:r=>(+r.stake_usd).toFixed(0)},
    {k:'status',h:'status',f:r=>r.status,
     c:r=>r.status==='WON'?'pos':r.status==='LOST'?'neg':'fl'},
    {k:'pnl_usd',h:'P&L',f:r=>r.status==='pending'?'–':'$'+(+r.pnl_usd).toFixed(0),
     c:r=>cls(+r.pnl_usd)},
  ],rows);
})();

/* ── model record (prediction_tracker scoreboard) ── */
(function(){
  const sc=D.scoreboard||[];
  const isTrue=v=>v===true||v==='True';
  const scored=sc.filter(r=>r.status==='settled'&&r.prob_source==='pre_result');
  const pending=sc.filter(r=>r.status==='pending').length;
  const correct=scored.filter(r=>isTrue(r.correct)).length;
  const mLL=scored.length?scored.reduce((s,r)=>s+(+r.log_loss),0)/scored.length:0;
  const wm=scored.filter(r=>r.mkt_log_loss!==''&&r.mkt_log_loss!==null);
  const kLL=wm.length?wm.reduce((s,r)=>s+(+r.mkt_log_loss),0)/wm.length:0;
  const mLLm=wm.length?wm.reduce((s,r)=>s+(+r.log_loss),0)/wm.length:0;
  document.getElementById('record-kpis').innerHTML=
    `<span class="kpi"><b>${correct}-${scored.length-correct}</b><span>record (argmax)</span></span>`+
    `<span class="kpi"><b>${scored.length?(100*correct/scored.length).toFixed(0)+'%':'–'}</b><span>accuracy</span></span>`+
    `<span class="kpi"><b>${scored.length?mLL.toFixed(3):'–'}</b><span>model log-loss</span></span>`+
    (wm.length?`<span class="kpi"><b class="${mLLm<=kLL?'pos':'neg'}">${kLL.toFixed(3)}</b><span>market log-loss</span></span>`:'')+
    `<span class="kpi"><b>${pending}</b><span>pending</span></span>`+
    (scored.length&&scored.length<40?`<span class="kpi"><b class="warn">n=${scored.length}</b><span>too small to mean anything</span></span>`:'');
  renderTable(document.getElementById('record-table'),[
    {k:'date',h:'date',f:r=>r.date},
    {k:'home_team',h:'match',f:r=>`${r.home_team} vs ${r.away_team}`},
    {k:'model_pick',h:'pick',f:r=>r.model_pick==='home'?r.home_team
      :r.model_pick==='away'?r.away_team:'Draw'},
    {k:'realized',h:'result',f:r=>r.realized||'–'},
    {k:'correct',h:'✓',f:r=>r.status!=='settled'?'':(isTrue(r.correct)?'✓':'✗'),
     c:r=>r.status!=='settled'?'':(isTrue(r.correct)?'pos':'neg')},
    {k:'p_realized',h:'p(result)',f:r=>r.p_realized===''?'–':fmtP(+r.p_realized)},
    {k:'log_loss',h:'LL',f:r=>r.log_loss===''?'–':(+r.log_loss).toFixed(3)},
    {k:'mkt_log_loss',h:'mkt LL',f:r=>r.mkt_log_loss===''?'–':(+r.mkt_log_loss).toFixed(3),
     c:r=>(r.log_loss!==''&&r.mkt_log_loss!=='')?(+r.log_loss<=+r.mkt_log_loss?'pos':'neg'):''},
  ],[...sc].sort((a,b)=>((a.status==='settled'?0:1)-(b.status==='settled'?0:1))
    ||String(a.date).localeCompare(String(b.date))));
})();

/* ── uncertainty (aleatoric proxy) ── */
(function(){
  const rows=D.implied.map(m=>{
    const b=D.bets.filter(x=>x.match_id===m.match_id);
    const p=['home','draw','away'].map(s=>(b.find(x=>x.outcome===s)||{}).model_prob||1e-9);
    const H=-p.reduce((a,x)=>a+x*Math.log2(x),0);
    return {match:`${m.home_team} vs ${m.away_team}`,date:m.date,
      entropy:+H.toFixed(3),fav:Math.max(...p)};});
  renderTable(document.getElementById('un-table'),[
    {k:'match',h:'match',f:r=>r.match},{k:'date',h:'date',f:r=>r.date},
    {k:'entropy',h:'entropy (bits)',f:r=>r.entropy.toFixed(3),
     c:r=>r.entropy>1.5?'warn':''},
    {k:'fav',h:'top prob',f:r=>fmtP(r.fav)},
    {k:'entropy',h:'read',f:r=>r.entropy>1.5?'coin-flip — size down':r.entropy<1.2?'clear favorite':''},
  ],rows.sort((a,b)=>b.entropy-a.entropy));
})();

/* ── ticker + live refresh ── */
let liveScores=[];   // filled by liveRefresh from ESPN scoreboard
function buildTicker(){
  const moveByMid={};
  D.movement.forEach(r=>{(moveByMid[r.match_id]=moveByMid[r.match_id]||{})[r.outcome]=r;});
  const live=liveScores.map(s=>`<span>${s}</span>`).join('');
  const t=D.implied.map(m=>{
    const mv=moveByMid[m.match_id]||{};
    const arrow=s=>{const r=mv[s];if(!r)return '';
      return r.implied_shift>0.01?'<b class="up">▲</b>':r.implied_shift<-0.01?'<b class="dn">▼</b>':'';};
    return `<span><b>${m.home_team}</b> v <b>${m.away_team}</b> `+
      `${m.home_decimal_odds.toFixed(2)}${arrow('home')} / ${m.draw_decimal_odds.toFixed(2)}${arrow('draw')}`+
      ` / ${m.away_decimal_odds.toFixed(2)}${arrow('away')}`+
      `${m.market_source!=='real'?' <span class="fl">(est)</span>':''}</span>`;}).join('');
  document.getElementById('ticker').innerHTML=live+t+live+t;
}
buildTicker();
const A2D=a=>{a=+String(a).replace('+','');return 1+(a>0?a/100:100/-a);};
async function liveRefresh(){
  const st=document.getElementById('live-status');
  st.textContent='fetching ESPN…';
  try{
    const ds=D.implied.map(m=>m.date.replace(/-/g,'')).sort();
    const url=`https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=${ds[0]}-${ds[ds.length-1]}&limit=300`;
    const r=await fetch(url);const j=await r.json();
    const alias={'United States':'USA','Bosnia-Herzegovina':'Bosnia and Herzegovina',
      'Türkiye':'Turkey','Korea Republic':'South Korea',"Côte d'Ivoire":'Ivory Coast',
      'Cabo Verde':'Cape Verde','Curaçao':'Curacao','Congo DR':'DR Congo'};
    const norm=s=>String(s).normalize('NFD').replace(/[^a-zA-Z]/g,'').toLowerCase().replace('and','');
    const byPair={};D.implied.forEach(m=>byPair[norm(m.home_team)+'|'+norm(m.away_team)]=m);
    let hit=0;liveScores=[];
    (j.events||[]).forEach(ev=>{
      const c=ev.competitions[0];if(!c)return;
      const tm={},sc={};c.competitors.forEach(x=>{
        tm[x.homeAway]=alias[x.team.displayName]||x.team.displayName;
        sc[x.homeAway]=x.score;});
      const state=((ev.status||{}).type||{}).state;
      if(state==='in'||state==='post'){
        const clock=state==='in'?` ${(ev.status.displayClock||'')}'`:' FT';
        liveScores.push(`<b class="${state==='in'?'up':'fl'}">⚽ ${state==='in'?'LIVE':'FT'}</b> `+
          `<b>${tm.home}</b> ${sc.home??''}–${sc.away??''} <b>${tm.away}</b>${state==='in'?clock:''}`);
      }
      let m=byPair[norm(tm.home)+'|'+norm(tm.away)],sw=false;
      if(!m){m=byPair[norm(tm.away)+'|'+norm(tm.home)];sw=true;}
      const o=(c.odds||[]).find(x=>x.moneyline);if(!m||!o)return;
      const g=(s,k)=>{const n=o.moneyline[s]||{};return ((n[k]||{}).odds);};
      let h=g('home','close')??g('home','open'),d=g('draw','close')??g('draw','open'),
          a=g('away','close')??g('away','open');
      let ho=g('home','open'),dop=g('draw','open'),ao=g('away','open');
      if(h==null||a==null)return;
      if(sw){[h,a]=[a,h];[ho,ao]=[ao,ho];}
      m.home_decimal_odds=A2D(h);m.draw_decimal_odds=A2D(d);m.away_decimal_odds=A2D(a);
      m.market_source='real';m.snapshot='live';
      // recompute edge/EV/Kelly on the live number (proportional de-vig in-browser)
      const dec=[m.home_decimal_odds,m.draw_decimal_odds,m.away_decimal_odds];
      const imp=dec.map(x=>1/x),s2=imp.reduce((p,q)=>p+q,0);
      // open→now movement on the live numbers (same thresholds as market_monitor.py)
      let fairOpen=null;
      if(ho!=null&&ao!=null&&dop!=null){
        const dop3=[A2D(ho),A2D(dop),A2D(ao)],io=dop3.map(x=>1/x),so=io.reduce((p,q)=>p+q,0);
        fairOpen={home:io[0]/so,draw:io[1]/so,away:io[2]/so,dec:dop3};
      }
      ['home','draw','away'].forEach((side,i)=>{
        const b=D.bets.find(x=>x.match_id===m.match_id&&x.outcome===side);if(!b)return;
        b.decimal_odds=dec[i];b.market_implied=imp[i]/s2;b.market_source='real';
        b.edge=b.model_prob-b.market_implied;
        b.ev=b.model_prob*(dec[i]-1)-(1-b.model_prob);
        const kb=dec[i]-1;b.kelly_full=Math.max(0,(kb*b.model_prob-(1-b.model_prob))/kb);
        b.kelly_quarter=b.kelly_full/4;
        b.recommended_stake_usd=Math.min(b.kelly_quarter,0.05)*1000;
        m[side+'_fair_prob']=b.market_implied;
        if(fairOpen){
          const fNow=imp[i]/s2,fOpen=fairOpen[side];
          liveMove[`${m.match_id}|${side}`]={
            significant:Math.abs(dec[i]-fairOpen.dec[i])>=0.10||Math.abs(fNow-fOpen)>=0.05,
            toward:Math.abs(b.model_prob-fNow)<Math.abs(b.model_prob-fOpen),
            open:fairOpen.dec[i],now:dec[i]};
        }});
      hit++;});
    const ts=new Date().toISOString().slice(11,16);
    buildTicker();drawScanner();drawMatches();
    drawDesk(recomputeDesk(),
      `⟳ verdicts + stakes recomputed on LIVE lines at ${ts}Z (proportional de-vig in-browser). `+
      `Model probabilities are the local 10k-sim output — run the Python pipeline to re-simulate `+
      `after results land.`);
    st.textContent=`live: ${hit} matches + desk recomputed ${ts}Z`+
      (liveScores.length?` · ${liveScores.length} in-play/finished`:'');
  }catch(e){st.textContent='live fetch failed ('+e.message+') — showing embedded data';}
}
let autoTimer=null;
document.getElementById('auto-live').onchange=e=>{
  if(e.target.checked){liveRefresh();autoTimer=setInterval(liveRefresh,240000);}
  else clearInterval(autoTimer);};
</script></body></html>
"""


def main():
    print("═" * 60)
    print("  V6 — Dashboard Build")
    print("═" * 60)
    payload = build_payload()
    for k in ["futures", "bets", "implied", "tourney", "movement", "arb"]:
        print(f"  {k:<10} {len(payload[k]):>5} rows")
    html = HTML.replace("__DATA__", json.dumps(payload, default=str))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n✅ Saved: {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
