"""
V3 Quantitative Audit
Runs: calibration, confederation breakdown, feature importance, ELO audit, WC2022 backtest
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE = Path("/sessions/zen-compassionate-thompson/mnt/World Cup 2026 Model")
MODELS = BASE / "models"
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FEATURE_COLS = [
    'home_fifa_rank','away_fifa_rank','fifa_rank_diff',
    'home_elo','away_elo','elo_diff',
    'home_win_rate_5','home_avg_goals_5','home_avg_gd_5',
    'home_win_rate_10','home_avg_goals_10','home_avg_gd_10',
    'home_weighted_win_rate_5','home_weighted_avg_goals_5','home_weighted_avg_gd_5',
    'home_weighted_win_rate_10','home_weighted_avg_goals_10','home_weighted_avg_gd_10',
    'away_win_rate_5','away_avg_goals_5','away_avg_gd_5',
    'away_win_rate_10','away_avg_goals_10','away_avg_gd_10',
    'away_weighted_win_rate_5','away_weighted_avg_goals_5','away_weighted_avg_gd_5',
    'away_weighted_win_rate_10','away_weighted_avg_goals_10','away_weighted_avg_gd_10',
    'h2h_home_wins','h2h_draws','h2h_away_wins','h2h_total','h2h_home_win_rate',
    'home_days_rest','away_days_rest','is_knockout','altitude_m'
]

CONF_MAP = {
    'France':'UEFA','England':'UEFA','Spain':'UEFA','Germany':'UEFA','Portugal':'UEFA',
    'Netherlands':'UEFA','Belgium':'UEFA','Italy':'UEFA','Croatia':'UEFA','Switzerland':'UEFA',
    'Denmark':'UEFA','Sweden':'UEFA','Norway':'UEFA','Austria':'UEFA','Czechia':'UEFA',
    'Serbia':'UEFA','Poland':'UEFA','Hungary':'UEFA','Scotland':'UEFA','Turkey':'UEFA',
    'Greece':'UEFA','Romania':'UEFA','Ukraine':'UEFA','Slovakia':'UEFA','Slovenia':'UEFA',
    'Albania':'UEFA','Iceland':'UEFA','Finland':'UEFA','Bosnia-Herzegovina':'UEFA',
    'Kosovo':'UEFA','North Macedonia':'UEFA','Montenegro':'UEFA','Georgia':'UEFA',
    'Bulgaria':'UEFA','Wales':'UEFA','Russia':'UEFA','Ireland':'UEFA',
    'Luxembourg':'UEFA','Estonia':'UEFA','Latvia':'UEFA','Lithuania':'UEFA',
    'Belarus':'UEFA','Armenia':'UEFA','Azerbaijan':'UEFA','Cyprus':'UEFA',
    'Malta':'UEFA','Faroe Islands':'UEFA','Liechtenstein':'UEFA','San Marino':'UEFA',
    'Andorra':'UEFA','Gibraltar':'UEFA',
    'Brazil':'CONMEBOL','Argentina':'CONMEBOL','Colombia':'CONMEBOL','Uruguay':'CONMEBOL',
    'Chile':'CONMEBOL','Ecuador':'CONMEBOL','Peru':'CONMEBOL','Paraguay':'CONMEBOL',
    'Venezuela':'CONMEBOL','Bolivia':'CONMEBOL',
    'Mexico':'CONCACAF','United States':'CONCACAF','Canada':'CONCACAF','Costa Rica':'CONCACAF',
    'Honduras':'CONCACAF','Jamaica':'CONCACAF','Panama':'CONCACAF','El Salvador':'CONCACAF',
    'Haiti':'CONCACAF','Trinidad and Tobago':'CONCACAF','Guatemala':'CONCACAF',
    'Cuba':'CONCACAF','Curacao':'CONCACAF','Curaçao':'CONCACAF',
    'Senegal':'CAF','Morocco':'CAF','Nigeria':'CAF','Ghana':'CAF','Egypt':'CAF',
    'Tunisia':'CAF','Algeria':'CAF','Cameroon':'CAF','Mali':'CAF','South Africa':'CAF',
    "Côte d'Ivoire":'CAF','Ivory Coast':'CAF','DR Congo':'CAF','Zambia':'CAF',
    'Tanzania':'CAF','Uganda':'CAF','Kenya':'CAF','Guinea':'CAF','Burkina Faso':'CAF',
    'Cape Verde Islands':'CAF','Mozambique':'CAF','Zimbabwe':'CAF','Gabon':'CAF',
    'Libya':'CAF','Sierra Leone':'CAF','Sudan':'CAF','Congo':'CAF','Benin':'CAF',
    'Angola':'CAF','Rwanda':'CAF',
    'Japan':'AFC','Korea Republic':'AFC','Iran':'AFC','Saudi Arabia':'AFC',
    'Australia':'AFC','Qatar':'AFC','China':'AFC','Iraq':'AFC',
    'Jordan':'AFC','Uzbekistan':'AFC','Vietnam':'AFC','India':'AFC','Syria':'AFC',
    'Lebanon':'AFC','Kyrgyzstan':'AFC','Thailand':'AFC','Indonesia':'AFC',
    'Philippines':'AFC','North Korea':'AFC','Tajikistan':'AFC',
    'New Zealand':'OFC','Solomon Islands':'OFC','Papua New Guinea':'OFC',
    'Fiji':'OFC','Tahiti':'OFC','New Caledonia':'OFC','Vanuatu':'OFC',
}

print("=" * 60)
print("  V3 QUANTITATIVE AUDIT")
print("=" * 60)

# ── Load models ──────────────────────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODELS / "xgb_v2.json"))
lgb_booster = lgb.Booster(model_file=str(MODELS / "lgbm_v2.txt"))

with open(MODELS / "dixon_coles_params.json") as f:
    dc_params = json.load(f)

# ── Load test set ─────────────────────────────────────────────
test = pd.read_csv(DATA / "processed/test_features.csv")
test = test.dropna(subset=['result'])
y_test = test['result'].astype(int).values
X_test = test[FEATURE_COLS].fillna(0)

# ── Ensemble probabilities ────────────────────────────────────
xgb_proba = xgb_model.predict_proba(X_test)
lgb_proba  = lgb_booster.predict(X_test.values)

ens_proba_path = MODELS / "ensemble_test_proba_v2.npy"
if ens_proba_path.exists():
    ens_proba = np.load(str(ens_proba_path))
    if ens_proba.shape[0] != len(y_test):
        ens_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        print("  [warn] ensemble shape mismatch — using XGB+LGB 50/50")
else:
    ens_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    print("  [warn] No ensemble_test_proba_v2.npy — using XGB+LGB 50/50")

# ─────────────────────────────────────────────────────────────
# 1. CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n\n── 1. CALIBRATION ──────────────────────────────────────")

overall_ll  = log_loss(y_test, ens_proba)
overall_acc = accuracy_score(y_test, ens_proba.argmax(axis=1))
print(f"  Test Log Loss : {overall_ll:.4f}")
print(f"  Test Accuracy : {overall_acc:.4f}")

classes = {0: "away_win", 1: "draw", 2: "home_win"}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Calibration Reliability Diagram — V2 Ensemble", fontsize=13)

cal_results = {}
for cls_idx, cls_name in classes.items():
    y_bin    = (y_test == cls_idx).astype(int)
    prob_col = ens_proba[:, cls_idx]
    try:
        frac_pos, mean_pred = calibration_curve(y_bin, prob_col, n_bins=10, strategy='quantile')
    except Exception:
        frac_pos, mean_pred = calibration_curve(y_bin, prob_col, n_bins=5, strategy='uniform')
    cal_err = float(np.mean(np.abs(frac_pos - mean_pred)))
    cal_results[cls_name] = {
        "mean_predicted_prob": round(float(np.mean(prob_col)),3),
        "actual_rate": round(float(np.mean(y_bin)),3),
        "mean_cal_error": round(cal_err,3),
        "bias": "OVER" if float(np.mean(prob_col)) > float(np.mean(y_bin)) else "UNDER"
    }
    print(f"  {cls_name:<12}: mean_pred={np.mean(prob_col):.3f}  actual={np.mean(y_bin):.3f}  "
          f"cal_err={cal_err:.3f}  {cal_results[cls_name]['bias']}")
    ax = axes[cls_idx]
    ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Perfect')
    ax.plot(mean_pred, frac_pos,'o-',color='steelblue',label='Model')
    ax.set_title(cls_name); ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction positives")
    ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.tight_layout()
calib_path = OUT / "plots/v3_calibration.png"
calib_path.parent.mkdir(exist_ok=True)
plt.savefig(str(calib_path), dpi=120, bbox_inches='tight')
plt.close()
print(f"  [saved] {calib_path}")

# ─────────────────────────────────────────────────────────────
# 2. CONFEDERATION-STRATIFIED METRICS
# ─────────────────────────────────────────────────────────────
print("\n\n── 2. CONFEDERATION BREAKDOWN ──────────────────────────")

test2 = test.copy().reset_index(drop=True)
test2['conf'] = test2['home_team'].map(CONF_MAP).fillna('OTHER')

conf_results = {}
print(f"  {'Conf':<10} {'N':>5}  {'LogLoss':>8}  {'Acc':>6}  {'DrawPred':>9}  {'DrawAct':>8}")
print(f"  {'-'*55}")
for conf in ['UEFA','CONMEBOL','CONCACAF','CAF','AFC','OFC','OTHER']:
    mask = (test2['conf'] == conf).values
    if mask.sum() < 10:
        continue
    y_c = y_test[mask]
    p_c = ens_proba[mask]
    ll  = log_loss(y_c, p_c)
    acc = accuracy_score(y_c, p_c.argmax(axis=1))
    draw_pred = p_c[:,1].mean()
    draw_act  = np.mean(y_c==1)
    conf_results[conf] = {"n": int(mask.sum()), "log_loss": round(ll,4),
                          "accuracy": round(acc,4),
                          "draw_pred_avg": round(draw_pred,3),
                          "draw_actual_rate": round(draw_act,3)}
    print(f"  {conf:<10} {mask.sum():>5}  {ll:>8.4f}  {acc:>6.4f}  {draw_pred:>9.3f}  {draw_act:>8.3f}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n\n── 3. FEATURE IMPORTANCE ───────────────────────────────")

xgb_scores = xgb_model.get_booster().get_score(importance_type='gain')
xgb_imp = pd.Series(xgb_scores).rename('xgb_gain')
lgb_gains = lgb_booster.feature_importance(importance_type='gain')
lgb_imp = pd.Series(dict(zip(FEATURE_COLS, lgb_gains))).rename('lgb_gain')

imp_df = pd.concat([xgb_imp, lgb_imp], axis=1).fillna(0)
xgb_total = imp_df['xgb_gain'].sum() or 1
lgb_total  = imp_df['lgb_gain'].sum() or 1
imp_df['avg_gain'] = (imp_df['xgb_gain']/xgb_total + imp_df['lgb_gain']/lgb_total) / 2
imp_df = imp_df.sort_values('avg_gain', ascending=False)

print(f"  {'Feature':<40} {'XGB%':>7} {'LGB%':>7} {'Avg%':>7}")
print(f"  {'-'*61}")
for feat, row in imp_df.head(20).iterrows():
    xpct = row['xgb_gain']/xgb_total*100
    lpct = row['lgb_gain']/lgb_total*100
    apct = row['avg_gain']*100
    print(f"  {feat:<40} {xpct:>6.1f}% {lpct:>6.1f}% {apct:>6.1f}%")

form_feats = [f for f in imp_df.index if any(k in f for k in ['win_rate','avg_goals','avg_gd'])]
elo_feats  = [f for f in imp_df.index if 'elo' in f.lower()]
rank_feats = [f for f in imp_df.index if 'fifa_rank' in f]
h2h_feats  = [f for f in imp_df.index if 'h2h' in f]

print(f"\n  Group contributions:")
print(f"  Rolling form : {imp_df.loc[form_feats,'avg_gain'].sum()*100:.1f}%")
print(f"  ELO          : {imp_df.loc[elo_feats,'avg_gain'].sum()*100:.1f}%")
print(f"  FIFA rank    : {imp_df.loc[rank_feats,'avg_gain'].sum()*100:.1f}%")
print(f"  H2H          : {imp_df.loc[h2h_feats,'avg_gain'].sum()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 4. ELO AUDIT
# ─────────────────────────────────────────────────────────────
print("\n\n── 4. ELO AUDIT ────────────────────────────────────────")

elo_df = pd.read_csv(DATA / "processed/elo_ratings.csv")
print(f"  Columns: {elo_df.columns.tolist()[:8]}  shape: {elo_df.shape}")

# Build latest ELO per team
try:
    if 'team' in elo_df.columns and 'elo' in elo_df.columns:
        if 'date' in elo_df.columns:
            latest_elo = elo_df.sort_values('date').groupby('team').last()['elo']
        else:
            latest_elo = elo_df.groupby('team').last()['elo']
    else:
        home = elo_df[['date','home_team','home_elo']].rename(columns={'home_team':'team','home_elo':'elo'})
        away = elo_df[['date','away_team','away_elo']].rename(columns={'away_team':'team','away_elo':'elo'})
        all_elo = pd.concat([home,away]).sort_values('date')
        latest_elo = all_elo.groupby('team').last()['elo']
except Exception as e:
    print(f"  ELO parse error: {e}")
    latest_elo = pd.Series(dtype=float)

mkt = pd.read_csv(DATA / "processed/market_divergence.csv").set_index('team')
rank_map = pd.concat([
    test[['home_team','home_fifa_rank']].rename(columns={'home_team':'team','home_fifa_rank':'rank'}),
    test[['away_team','away_fifa_rank']].rename(columns={'away_team':'team','away_fifa_rank':'rank'})
]).groupby('team')['rank'].last()

audit_teams = ['Mexico','France','England','Portugal','Iran','Japan',
               'Korea Republic','Brazil','Argentina','United States','Canada']

print(f"\n  {'Team':<22} {'ELO':>6} {'FIFA_Rk':>8} {'Mkt%':>7} {'Mdl%':>7} {'Edge':>7}")
print(f"  {'-'*60}")
elo_audit = {}
for t in audit_teams:
    elo_val  = latest_elo.get(t, float('nan'))
    rank_val = rank_map.get(t, float('nan'))
    if t in mkt.index:
        mkt_p = mkt.loc[t,'market_prob']*100
        mdl_p = mkt.loc[t,'model_prob']*100
        edge  = mkt.loc[t,'edge']*100
    else:
        mkt_p = mdl_p = edge = float('nan')
    elo_str  = f"{elo_val:.0f}" if not np.isnan(elo_val) else "N/A"
    rank_str = f"{rank_val:.0f}" if not np.isnan(rank_val) else "N/A"
    print(f"  {t:<22} {elo_str:>6} {rank_str:>8} {mkt_p:>6.1f}% {mdl_p:>6.1f}% {edge:>+6.1f}%")
    elo_audit[t] = {"elo": None if np.isnan(elo_val) else round(elo_val,1),
                    "fifa_rank": None if np.isnan(rank_val) else int(rank_val),
                    "market_pct": round(mkt_p,2), "model_pct": round(mdl_p,2), "edge_pp": round(edge,2)}

# ─────────────────────────────────────────────────────────────
# 5. WC2022 BACKTEST
# ─────────────────────────────────────────────────────────────
print("\n\n── 5. WC2022 BACKTEST ──────────────────────────────────")

wc22 = test[test['tournament'].str.contains('World Cup', na=False)].copy().reset_index(drop=True)
print(f"  WC2022 matches in test set: {len(wc22)}")

wc22_result = {}
if len(wc22) > 0:
    wc22_mask = test.reset_index(drop=True)['tournament'].str.contains('World Cup', na=False).values
    y_wc22 = y_test[wc22_mask]
    p_wc22 = ens_proba[wc22_mask]

    wc22_ll  = log_loss(y_wc22, p_wc22)
    wc22_acc = accuracy_score(y_wc22, p_wc22.argmax(axis=1))

    naive_ll = log_loss(y_wc22, np.full((len(y_wc22),3), 1/3))

    train_df = pd.read_csv(DATA / "processed/train.csv")
    vc = train_df['result'].value_counts(normalize=True).sort_index()
    br = np.array([[vc.get(0,1/3), vc.get(1,1/3), vc.get(2,1/3)]] * len(y_wc22))
    base_ll = log_loss(y_wc22, br)

    print(f"  Model  log loss : {wc22_ll:.4f}")
    print(f"  Naive  log loss : {naive_ll:.4f}  (uniform 1/3 each)")
    print(f"  Base   log loss : {base_ll:.4f}  (train base rates)")
    print(f"  Skill vs naive  : {(naive_ll - wc22_ll)/naive_ll*100:+.1f}%")
    print(f"  Skill vs base   : {(base_ll  - wc22_ll)/base_ll  *100:+.1f}%")
    print(f"  WC2022 accuracy : {wc22_acc:.4f}")
    dist = dict(zip(['away_win','draw','home_win'], np.bincount(y_wc22, minlength=3)))
    print(f"  Outcome dist    : {dist}")
    wc22_result = {
        "n_matches": len(y_wc22),
        "model_log_loss": round(wc22_ll,4),
        "naive_log_loss": round(naive_ll,4),
        "base_rate_log_loss": round(base_ll,4),
        "skill_vs_uniform_pct": round((naive_ll-wc22_ll)/naive_ll*100,1),
        "skill_vs_base_pct":    round((base_ll-wc22_ll)/base_ll*100,1),
        "accuracy": round(wc22_acc,4),
        "outcome_distribution": dist
    }

# ─────────────────────────────────────────────────────────────
# 6. SoS DISTORTION
# ─────────────────────────────────────────────────────────────
print("\n\n── 6. SoS DISTORTION (avg opponent ELO, last 20 matches) ──")

train_feats = pd.read_csv(DATA / "processed/train_features.csv")
audit_teams_sos = ['Mexico','France','England','Brazil','Argentina',
                   'Japan','Iran','Canada','United States','Korea Republic']

sos_results = {}
print(f"  {'Team':<22} {'Own ELO':>8}  {'Avg Opp ELO':>12}  {'ELO Gap':>8}")
print(f"  {'-'*55}")
for team in audit_teams_sos:
    as_home = train_feats[train_feats['home_team']==team][['date','away_elo']].rename(columns={'away_elo':'opp_elo'})
    as_away = train_feats[train_feats['away_team']==team][['date','home_elo']].rename(columns={'home_elo':'opp_elo'})
    all_m   = pd.concat([as_home,as_away]).dropna().sort_values('date')
    if len(all_m) < 5:
        continue
    last20      = all_m.tail(20)
    avg_opp     = last20['opp_elo'].mean()
    own_elo_val = latest_elo.get(team, float('nan'))
    gap = own_elo_val - avg_opp if not np.isnan(own_elo_val) else float('nan')
    own_str = f"{own_elo_val:.0f}" if not np.isnan(own_elo_val) else "N/A"
    gap_str = f"{gap:+.0f}" if not np.isnan(gap) else "N/A"
    print(f"  {team:<22} {own_str:>8}  {avg_opp:>12.0f}  {gap_str:>8}")
    sos_results[team] = {"own_elo": None if np.isnan(own_elo_val) else round(own_elo_val,1),
                         "avg_opp_elo_last20": round(avg_opp,1),
                         "elo_gap": None if np.isnan(gap) else round(gap,1)}

# ─────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────
report = {
    "overall": {"log_loss": round(overall_ll,4), "accuracy": round(overall_acc,4)},
    "calibration": cal_results,
    "confederation_breakdown": conf_results,
    "feature_importance_top10": {k: round(v,4) for k,v in imp_df.head(10)['avg_gain'].items()},
    "feature_group_pct": {
        "rolling_form": round(imp_df.loc[form_feats,'avg_gain'].sum()*100,1),
        "elo":          round(imp_df.loc[elo_feats,'avg_gain'].sum()*100,1),
        "fifa_rank":    round(imp_df.loc[rank_feats,'avg_gain'].sum()*100,1),
        "h2h":          round(imp_df.loc[h2h_feats,'avg_gain'].sum()*100,1),
    },
    "elo_audit": elo_audit,
    "sos_distortion": sos_results,
    "wc2022_backtest": wc22_result,
}

report_path = OUT / "v3_audit_report.json"
with open(str(report_path), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  AUDIT COMPLETE")
print(f"  Report: {report_path}")
print(f"{'='*60}")
