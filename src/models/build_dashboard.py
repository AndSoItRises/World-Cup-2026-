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

NEXT_STEPS = """\
V6 BUILD STATE (this file doubles as the context doc — see CONTEXT_V6.md in repo)

DONE
  1. market_ingestion.py — odds parsing, Shin de-vig, name audits
  2. bet_sim.py — edge / EV / Kelly (1/4, 5% cap), futures + matches
  +  fetch_live_odds.py — ESPN/DraftKings 3-way lines (open + current), 72/72 matched
  +  market_monitor.py — line movement vs model, cross-book arb scanner
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
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_payload() -> dict:
    fixtures = load("wc2026_fixtures.csv", proc=False)
    results = load("wc2026_live_results.csv", proc=False)
    group_fix = fixtures[fixtures["stage"] == "Group Stage"]

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "model": "V4 ensemble (XGB .275 / LGBM .275 / DC .45) — prod, ceiling-confirmed",
            "accuracy": "62.0% | LL 0.8461 | model-market corr 0.84",
            "devig": "Shin",
        },
        "nextSteps": NEXT_STEPS,
        "futures": load("value_bets_futures.csv").to_dict("records"),
        "bets": load("value_bets.csv").to_dict("records"),
        "implied": load("market_implied_probs.csv").to_dict("records"),
        "tourney": load("tournament_probs_live.csv").to_dict("records"),
        "movement": load("line_movement.csv").to_dict("records"),
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
#ticker{display:inline-block;padding-left:100%;animation:tick 90s linear infinite}
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
</style></head><body>
<header>
  <h1>WC2026 QUANT DESK</h1>
  <span class="sub" id="meta-sub"></span>
  <button class="act" onclick="liveRefresh()">⟳ LIVE ODDS</button>
  <span id="live-status"></span>
</header>
<div id="ticker-wrap"><div id="ticker"></div></div>
<nav id="nav"></nav>
<main>

<div class="tab" id="tab-scanner">
  <h2>Value Bet Scanner — match 1X2</h2>
  <label><input type="checkbox" id="sc-evpos" checked> EV &gt; 0 only</label>
  <label><input type="checkbox" id="sc-nodraw" checked> hide draw bets (model bias)</label>
  <label><input type="checkbox" id="sc-fav"> favorites only (fair ≥ 40%)</label>
  <div class="cav">Draw probs are upweighted 1.75× by design and ELO compresses
  lopsided ties — draw/underdog "edges" are mostly documented model bias, not
  market error. Edge is unproven out-of-sample (V5 DL-05). Stakes: ¼-Kelly, 5% cap.</div>
  <div style="max-height:60vh;overflow:auto"><table id="sc-table"></table></div>
  <h2 style="margin-top:20px">Tournament Winner Futures</h2>
  <label><input type="checkbox" id="fu-tail" checked> hide tail-risk (model &lt; 2%)</label>
  <div style="max-height:40vh;overflow:auto"><table id="fu-table"></table></div>
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
  <div style="max-height:70vh;overflow:auto"><table id="br-table"></table></div>
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
  since open (sharps agreeing); against = market hardening the other way. Arbs need
  ≥2 books — add rows to wc2026_match_odds.csv and rebuild.</div>
  <div style="max-height:65vh;overflow:auto"><table id="mv-table"></table></div>
</div>

<div class="tab" id="tab-uncertainty">
  <h2>Uncertainty</h2>
  <h3>Aleatoric — how unpredictable is the match itself (3-way entropy)</h3>
  <div class="note">entropy of the model's [home, draw, away] in bits; max = 1.585
  (perfect 3-way coin-flip). High entropy ⇒ genuinely volatile tie ⇒ size down.</div>
  <div style="max-height:45vh;overflow:auto"><table id="un-table"></table></div>
  <div class="cav">Epistemic uncertainty (XGB vs LGBM vs DC disagreement per match)
  is phase 4 — needs per-component probabilities saved at predict time
  (src/models/uncertainty.py, next step 4). Until then: treat tail-risk futures
  and high-entropy matches as the size-down list.</div>
</div>

<div class="tab" id="tab-notes">
  <h2>Context & Next Steps (embedded research state)</h2>
  <div class="card"><pre id="notes-pre"></pre></div>
</div>

</main>
<script>
const D = __DATA__;
const fmtP = x => (100*x).toFixed(1)+'%';
const fmtPP = x => (x>=0?'+':'')+(100*x).toFixed(1)+'pp';
const cls = x => x>0.0001?'pos':(x<-0.0001?'neg':'fl');

/* ── nav ── */
const TABS = [['scanner','SCANNER'],['matches','MATCHES'],['groups','GROUPS'],
 ['bracket','BRACKET'],['scatter','DIVERGENCE'],['bankroll','BANKROLL'],
 ['monitor','MOVEMENT+ARB'],['uncertainty','UNCERTAINTY'],['notes','NOTES']];
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
       <span style="width:60px;color:var(--dim)">W ${fmtP(r.win)}</span></div>`).join('')+
      `<div class="note">advance% · W = win tournament</div>`;
    el.appendChild(card);});
})();

/* ── bracket table ── */
renderTable(document.getElementById('br-table'),[
  {k:'team',h:'team',f:r=>r.team},{k:'fifa_rank',h:'rank',f:r=>r.fifa_rank},
  ...['p_group_adv','p_r16','p_quarterfinal','p_semifinal','p_final','p_winner']
    .map(k=>({k,h:k.replace('p_','').replace('quarterfinal','QF').replace('semifinal','SF')
      .replace('group_adv','adv'),f:r=>fmtP(r[k]||0)})),
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
  document.getElementById('arb-books').textContent=D.arb.length?Math.max(...D.arb.map(r=>r.n_books)):0;
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
function buildTicker(){
  const moveByMid={};
  D.movement.forEach(r=>{(moveByMid[r.match_id]=moveByMid[r.match_id]||{})[r.outcome]=r;});
  const t=D.implied.map(m=>{
    const mv=moveByMid[m.match_id]||{};
    const arrow=s=>{const r=mv[s];if(!r)return '';
      return r.implied_shift>0.01?'<b class="up">▲</b>':r.implied_shift<-0.01?'<b class="dn">▼</b>':'';};
    return `<span><b>${m.home_team}</b> v <b>${m.away_team}</b> `+
      `${m.home_decimal_odds.toFixed(2)}${arrow('home')} / ${m.draw_decimal_odds.toFixed(2)}${arrow('draw')}`+
      ` / ${m.away_decimal_odds.toFixed(2)}${arrow('away')}`+
      `${m.market_source!=='real'?' <span class="fl">(est)</span>':''}</span>`;}).join('');
  document.getElementById('ticker').innerHTML=t+t;
}
buildTicker();
const A2D=a=>{a=+String(a).replace('+','');return 1+(a>0?a/100:100/-a);};
async function liveRefresh(){
  const st=document.getElementById('live-status');
  st.textContent='fetching ESPN…';
  try{
    const ds=D.implied.map(m=>m.date.replace(/-/g,''));
    const url=`https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=${ds[0]}-${ds[ds.length-1]}&limit=300`;
    const r=await fetch(url);const j=await r.json();
    const alias={'United States':'USA','Bosnia-Herzegovina':'Bosnia and Herzegovina',
      'Türkiye':'Turkey','Korea Republic':'South Korea',"Côte d'Ivoire":'Ivory Coast',
      'Cabo Verde':'Cape Verde','Curaçao':'Curacao','Congo DR':'DR Congo'};
    const norm=s=>String(s).normalize('NFD').replace(/[^a-zA-Z]/g,'').toLowerCase().replace('and','');
    const byPair={};D.implied.forEach(m=>byPair[norm(m.home_team)+'|'+norm(m.away_team)]=m);
    let hit=0;
    (j.events||[]).forEach(ev=>{
      const c=ev.competitions[0];if(!c)return;
      const tm={};c.competitors.forEach(x=>tm[x.homeAway]=alias[x.team.displayName]||x.team.displayName);
      let m=byPair[norm(tm.home)+'|'+norm(tm.away)],sw=false;
      if(!m){m=byPair[norm(tm.away)+'|'+norm(tm.home)];sw=true;}
      const o=(c.odds||[]).find(x=>x.moneyline);if(!m||!o)return;
      const g=s=>{const n=o.moneyline[s]||{};return (n.close||{}).odds??(n.open||{}).odds;};
      let h=g('home'),d=g('draw'),a=g('away');if(h==null||a==null)return;
      if(sw)[h,a]=[a,h];
      m.home_decimal_odds=A2D(h);m.draw_decimal_odds=A2D(d);m.away_decimal_odds=A2D(a);
      m.market_source='real';m.snapshot='live';
      // recompute edge/EV/Kelly on the live number (proportional de-vig in-browser)
      const dec=[m.home_decimal_odds,m.draw_decimal_odds,m.away_decimal_odds];
      const imp=dec.map(x=>1/x),s2=imp.reduce((p,q)=>p+q,0);
      ['home','draw','away'].forEach((side,i)=>{
        const b=D.bets.find(x=>x.match_id===m.match_id&&x.outcome===side);if(!b)return;
        b.decimal_odds=dec[i];b.market_implied=imp[i]/s2;b.market_source='real';
        b.edge=b.model_prob-b.market_implied;
        b.ev=b.model_prob*(dec[i]-1)-(1-b.model_prob);
        const kb=dec[i]-1;b.kelly_full=Math.max(0,(kb*b.model_prob-(1-b.model_prob))/kb);
        b.kelly_quarter=b.kelly_full/4;
        b.recommended_stake_usd=Math.min(b.kelly_quarter,0.05)*1000;
        m[side+'_fair_prob']=b.market_implied;});
      hit++;});
    buildTicker();drawScanner();drawMatches();
    st.textContent=`live: ${hit} matches updated ${new Date().toISOString().slice(11,16)}Z (proportional de-vig in-browser)`;
  }catch(e){st.textContent='live fetch failed ('+e.message+') — showing embedded data';}
}
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
