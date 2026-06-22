# _archive — Context Summary

Cold storage for superseded material. **Nothing here is imported by `src/` or run by
the live pipeline** — it is reference only. Active context lives in the repo root
(`HANDOFF.md` + `CONTEXT_V6.md`). Archived 2026-06-22 (see `../CLEANUP_LOG_20260622.txt`).

---

## `context_versions/` — model-evolution decision logs (V2–V5)
The historical context docs, superseded by the active **`CONTEXT_V6.md`** in the repo root.
Read order if you ever need the full lineage: V2 → V3 → V4 → V5 → (active) V6.

| File | Era | What it covers |
|---|---|---|
| `CONTEXT_V2.md` | "Intelligent" | ELO, opponent-weighted form, Dixon-Coles → 62.1% acc, LL 0.8458, draw recall 9.7% |
| `CONTEXT_V3.md` | "Ceiling" | data fixes, residual analysis; model ceiling confirmed |
| `CONTEXT_V4.md` | "Final" | squad strength + depth (FIFA/EA ratings) → V4 frozen at 62.0% acc, LL 0.8461 |
| `CONTEXT_V5.md` | "Confirmed" | remaining feature/architecture levers tested and exhausted (do not re-litigate) |

The live model is **V4 (frozen)** + the **V6 calibrator** (log-pool + draw-shrink). See
`CONTEXT_V6.md` §1 / MODEL TRUTHS for the current state.

## `EVO_WC_2026_QNT/` — earlier standalone project iteration
A prior, self-contained version of the WC2026 model ("EVO WC 2026 QNT") merged in during the
2026-06-22 cleanup. 15 files: its own `README.md`, `CLAUDE.md`, `CLAUDE_CODE_PROMPT_V5.md`,
`CONTEXT_V5_FINAL.md`, `KEY_INSIGHTS.md` + `key_insights_charts/`, `LIVE_UPDATE_SPEC.md`, and
standalone dashboards (`bracket_simulator.html`, `index.html`). Documents the same V1→V5
ensemble evolution (XGBoost + LightGBM + Dixon-Coles, ELO, squad strength) and an early live
bracket simulator. Superseded by the current repo structure (`src/models/`, `outputs/quant_dashboard.html`).
Kept for provenance — historical KEY_INSIGHTS and the early LIVE_UPDATE_SPEC are the useful bits.

---

**If you are starting work cold:** ignore this directory. Go to `../HANDOFF.md`.
