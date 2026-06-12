r"""
Fable 5 research memo — "is the model's dog/draw disagreement with the market
EDGE or ERROR?" (the DL-10/DL-11 open question).

Runs claude-fable-5 (Anthropic's most capable model) over the repo's own context
docs + key source files, with web search enabled for academic-literature grounding,
and writes a research memo to outputs/research/.

Why Fable here (not Opus 4.8): this is a long-horizon, ambiguous, read-heavy
synthesis task handed over fully specified — Fable's wheelhouse. Routine coding on
this project should stay on Opus 4.8 (half the price).

PREREQUISITES
  1. An Anthropic API key (https://console.anthropic.com) — billed per token,
     separate from any Claude subscription. ~$3-8 per run (Fable $10/$50 per MTok,
     plus web search + high-effort thinking).
  2. The SDK in this venv:   .\venv\Scripts\python.exe -m pip install anthropic
  3. The key in the environment:
        PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
     (or run `ant auth login` once and leave the key unset).

RUN
  .\venv\Scripts\python.exe scripts\fable_research_memo.py

Model/API choices follow the current Anthropic guidance for Fable 5:
  - model "claude-fable-5"; thinking is always-on (we omit the `thinking` param).
  - effort "high" via output_config (depth control; not a token budget).
  - streaming (long output) + a manual loop to handle server-tool `pause_turn`.
  - web_search_20260209 server tool for literature grounding.
  - we check stop_reason == "refusal" before reading content.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "outputs" / "research"

# Repo files fed to the model as grounding context (path -> short label).
CONTEXT_FILES = [
    ("CONTEXT_V6.md",                         "V6 context + decision log (DL-01..12)"),
    ("HANDOFF.md",                            "agent handoff / system map"),
    ("src/models/calibrate_v6.py",            "stage-1 calibrator experiments (ADOPTED)"),
    ("src/models/calibrate_tilt.py",          "stage-2 dog-tilt experiments (CUT, DL-10)"),
    ("src/models/clv_tracker.py",             "CLV tracker (the live DL-10 decider)"),
    ("src/models/desk_call.py",               "rule-based BET/LEAN/PASS engine"),
    ("src/models/market_divergence.py",       "model-vs-market divergence"),
]

PROMPT = """\
I'm building a World Cup 2026 prediction + betting-research model as a quant-learning \
project (I'm transitioning into quant research, so show the reasoning, not just \
conclusions). Its central open question, documented in the attached CONTEXT_V6 \
decision logs DL-09 through DL-11:

The model systematically disagrees with the betting market on draws and underdogs. \
The held-out reliability check (DL-10) found the model's underdog probabilities are \
honest against realized international results (favorite-bucket reliability is flat, \
fitted dog-shrink ~= 1.0), yet the model still diverges from market prices on dogs and \
draws. So: is that disagreement EDGE (the market is mispricing) or ERROR (the market \
prices information the model can't see — lineups, injuries, motivation)? The CLV \
tracker (DL-11) is my live experiment intended to settle it during this tournament.

Using the attached context docs and source, and grounding yourself in the academic \
literature on football-betting market efficiency, the Dixon-Coles model, the \
favorite-longshot bias, and closing-line value (CLV) as a proxy for genuine edge, \
write me a rigorous research memo that covers:

1. What the evidence already in THIS model legitimately supports about edge-vs-error, \
   and exactly where my current reasoning has gaps or unjustified leaps.
2. What the literature says about whether disagreements of this shape (dog/draw, on a \
   well-calibrated probabilistic model) historically turn out to be edge or bias — \
   with citations I can follow up.
3. A concrete, falsifiable experimental protocol to resolve the question from THIS \
   tournament's results: what to measure, the sample-size/power reality of ~104 \
   matches, what statistic decides it, and — critically — what observation would \
   FALSIFY the "it's edge" hypothesis (not just confirm it).
4. Any methodological risks in my current CLV setup (provisional vs final closes, \
   per-category gating, the draw/longshot haircuts) that could bias the verdict.

Give me a recommendation, not an exhaustive survey of options. When you have enough \
to draw a conclusion, draw it.\
"""


def build_context() -> str:
    parts = []
    for rel, label in CONTEXT_FILES:
        p = BASE / rel
        if not p.exists():
            print(f"  warn: missing {rel} (skipping)", file=sys.stderr)
            continue
        parts.append(f"===== {rel} — {label} =====\n{p.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts)


def main():
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic SDK not installed. Run: "
                 ".\\venv\\Scripts\\python.exe -m pip install anthropic")

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No credentials. Set $env:ANTHROPIC_API_KEY (or run `ant auth login`).")

    client = anthropic.Anthropic()
    context = build_context()

    messages = [{
        "role": "user",
        "content": [
            {"type": "text",
             "text": "Repository context (treat as authoritative for what the model "
                     "actually does and has found):\n\n" + context,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": PROMPT},
        ],
    }]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]

    print("Running claude-fable-5 (effort=high, web search on). This can take several "
          "minutes — thinking + searching + writing.\n", flush=True)

    transcript, usage_in, usage_out = [], 0, 0
    for turn in range(12):  # bound the server-tool continuation loop
        with client.messages.stream(
            model="claude-fable-5",
            max_tokens=32000,
            output_config={"effort": "high"},   # depth control; thinking is always-on, so omit `thinking`
            tools=tools,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            msg = stream.get_final_message()

        usage_in += msg.usage.input_tokens
        usage_out += msg.usage.output_tokens
        transcript.extend(b.text for b in msg.content if b.type == "text")

        if msg.stop_reason == "refusal":
            sys.exit("\n\nModel refused (stop_reason=refusal). Sports-betting research "
                     "shouldn't trip the bio/cyber classifiers — inspect stop_details.")
        if msg.stop_reason == "pause_turn":
            # server tool (web search) hit the per-turn cap; resume.
            messages.append({"role": "assistant", "content": msg.content})
            continue
        break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"dog_vs_market_memo_{stamp}.md"
    header = (f"# Edge or Error? Dog/Draw vs the Market — Fable 5 research memo\n"
              f"_Generated {stamp}Z by claude-fable-5 (effort=high, web search). "
              f"Tokens: {usage_in:,} in / {usage_out:,} out._\n\n---\n\n")
    out.write_text(header + "\n".join(transcript), encoding="utf-8")
    print(f"\n\nSaved: {out}")
    # Rough cost at Fable list pricing ($10/$50 per MTok); thinking bills as output.
    cost = usage_in / 1e6 * 10 + usage_out / 1e6 * 50
    print(f"Tokens: {usage_in:,} in / {usage_out:,} out  (~${cost:.2f} at Fable list price)")


if __name__ == "__main__":
    main()
