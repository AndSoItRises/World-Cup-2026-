"""
Phase 8: Visualization
Generates three charts from Monte Carlo simulation results.

Outputs:
  outputs/viz/01_win_probability.png
  outputs/viz/02_stage_heatmap.png
  outputs/viz/03_advance_vs_win_scatter.png

Run with:
  python -m src.visualization.charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

BASE     = Path(__file__).resolve().parents[2]
DATA     = BASE / "data" / "processed"
OUT_DIR  = BASE / "outputs" / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#e6edf3",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})

ACCENT   = "#58a6ff"
GOLD     = "#f0c040"
SILVER   = "#8b949e"
BRONZE   = "#c87533"
PILL_CLR = "#1f6feb"


def load_data():
    probs = pd.read_csv(DATA / "tournament_probs.csv")
    return probs


# ── Chart 1: Tournament Win Probability ───────────────────────────────────────
def chart_win_probability(df):
    top20 = df.sort_values("p_winner", ascending=False).head(20).copy()
    top20 = top20.sort_values("p_winner", ascending=True)   # flip for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0d1117")

    # Color by podium position
    colors = []
    teams_desc = list(reversed(top20["team"].tolist()))
    for i, t in enumerate(top20["team"]):
        rank_in_top20 = teams_desc.index(t)
        if rank_in_top20 == 0:   colors.append(GOLD)
        elif rank_in_top20 == 1: colors.append(SILVER)
        elif rank_in_top20 == 2: colors.append(BRONZE)
        else:                    colors.append(ACCENT)

    bars = ax.barh(top20["team"], top20["p_winner"] * 100,
                   color=colors, height=0.65, zorder=3)

    # Value labels
    for bar, val in zip(bars, top20["p_winner"]):
        ax.text(val * 100 + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", ha="left",
                fontsize=8.5, color="#e6edf3")

    ax.set_xlabel("Tournament Win Probability (%)", fontsize=10, labelpad=10)
    ax.set_title("WC 2026 — Tournament Win Probability\n(Monte Carlo, 10,000 simulations)",
                 fontsize=13, fontweight="bold", pad=14, color="#e6edf3")
    ax.set_xlim(0, top20["p_winner"].max() * 100 * 1.18)
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    path = OUT_DIR / "01_win_probability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 2: Stage-by-Stage Heatmap ──────────────────────────────────────────
def chart_stage_heatmap(df):
    top16 = df.sort_values("p_winner", ascending=False).head(16).copy()

    stages   = ["p_group_adv", "p_r16", "p_quarterfinal",
                 "p_semifinal", "p_final", "p_winner"]
    labels   = ["Advance\nfrom Group", "Round\nof 16", "Quarter-\nfinal",
                 "Semi-\nfinal", "Final", "Winner"]

    matrix = top16[stages].values  # shape (16, 6)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0d1117")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wc", ["#0d1117", "#1f3d6e", "#1f6feb", "#58a6ff", "#f0c040"]
    )

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Cell text
    for i in range(len(top16)):
        for j in range(len(stages)):
            val = matrix[i, j]
            txt_color = "#0d1117" if val > 0.55 else "#e6edf3"
            ax.text(j, i, f"{val*100:.0f}%",
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold")

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(top16)))
    ax.set_yticklabels(top16["team"].tolist(), fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title("WC 2026 — Stage Progression Probabilities (Top 16 Teams)\n",
                 fontsize=12, fontweight="bold", pad=6, color="#e6edf3")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=8)
    cbar.set_label("Probability", color="#8b949e", fontsize=8)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / "02_stage_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 3: Group Advance vs Win Probability (scatter) ──────────────────────
def chart_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")

    # Color by FIFA rank bucket
    def rank_color(rank):
        if rank <= 5:    return GOLD
        elif rank <= 15: return ACCENT
        elif rank <= 30: return "#3fb950"
        else:            return SILVER

    for _, row in df.iterrows():
        c = rank_color(row["fifa_rank"])
        ax.scatter(row["p_group_adv"] * 100, row["p_winner"] * 100,
                   color=c, s=55, zorder=3, alpha=0.85)

    # Label top 12 and any interesting outliers
    label_teams = df.nlargest(12, "p_winner")["team"].tolist()
    # Add Argentina (likely an outlier worth showing)
    if "Argentina" not in label_teams:
        label_teams.append("Argentina")

    offsets = {}  # manual nudges for overlapping labels
    for _, row in df[df["team"].isin(label_teams)].iterrows():
        x = row["p_group_adv"] * 100
        y = row["p_winner"] * 100
        ox, oy = offsets.get(row["team"], (3, 1))
        ax.annotate(row["team"],
                    xy=(x, y), xytext=(x + ox, y + oy),
                    fontsize=7.5, color="#e6edf3",
                    arrowprops=dict(arrowstyle="-", color="#30363d", lw=0.7))

    # Legend for rank buckets
    legend_items = [
        plt.scatter([], [], color=GOLD,     s=55, label="FIFA Top 5"),
        plt.scatter([], [], color=ACCENT,   s=55, label="FIFA 6–15"),
        plt.scatter([], [], color="#3fb950",s=55, label="FIFA 16–30"),
        plt.scatter([], [], color=SILVER,   s=55, label="FIFA 31+"),
    ]
    ax.legend(handles=legend_items, fontsize=8, framealpha=0.2,
              facecolor="#0d1117", edgecolor="#30363d", loc="upper left")

    ax.set_xlabel("Group Stage Advance Probability (%)", fontsize=10, labelpad=8)
    ax.set_ylabel("Tournament Win Probability (%)", fontsize=10, labelpad=8)
    ax.set_title("WC 2026 — Group Advance vs Tournament Win Probability\n"
                 "Teams above the trend line are overperforming their bracket",
                 fontsize=11, fontweight="bold", pad=10, color="#e6edf3")
    ax.grid(zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#30363d")

    plt.tight_layout()
    path = OUT_DIR / "03_advance_vs_win_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✅ Saved: {path}")


# ── Chart 4: Model vs Market Divergence (V2) ─────────────────────────────────
def chart_market_divergence(df=None):
    """Diverging bar of edge (model − market) per team. Needs market_divergence.csv."""
    md_path = DATA / "market_divergence.csv"
    if not md_path.exists():
        print(f"  (skipping market chart — {md_path.name} not found)")
        return
    md = pd.read_csv(md_path)
    # Show the biggest divergences on each side
    md = md.sort_values("edge")
    top = pd.concat([md.head(8), md.tail(8)]).drop_duplicates("team")
    top = top.sort_values("edge")

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0d1117")

    colors = [GOLD if e > 0 else SILVER for e in top["edge"]]
    bars = ax.barh(top["team"], top["edge"] * 100, color=colors, height=0.65, zorder=3)

    for bar, e in zip(bars, top["edge"]):
        x = e * 100
        ax.text(x + (0.1 if x >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
                f"{x:+.1f}", va="center", ha="left" if x >= 0 else "right",
                fontsize=8, color="#e6edf3")

    ax.axvline(0, color="#30363d", lw=1, zorder=2)
    ax.set_xlabel("Model edge vs market (percentage points)", fontsize=10, labelpad=10)
    ax.set_title("WC 2026 — Model vs Market Divergence\n"
                 "Gold = model sees more value than market | Grey = model fades",
                 fontsize=12, fontweight="bold", pad=14, color="#e6edf3")
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    path = OUT_DIR / "04_market_divergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✅ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Phase 8: Visualization")
    print("═" * 60)

    df = load_data()
    print(f"  Loaded {len(df)} teams\n")

    chart_win_probability(df)
    chart_stage_heatmap(df)
    chart_scatter(df)
    chart_market_divergence(df)

    print(f"\nAll charts saved to outputs/viz/")
    print(f"Phase 8 complete ✅")


if __name__ == "__main__":
    main()
