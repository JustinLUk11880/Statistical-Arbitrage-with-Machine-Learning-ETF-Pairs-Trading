"""
Generates notebooks 03-06 for the multi-pair pipeline.
Run once; after that execute the notebooks in order.
"""
import nbformat as nbf

C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell

def write(nb, path):
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    })
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ─── Notebook 03: Feature Engineering ────────────────────────────────────────
nb03 = nbf.v4.new_notebook()
nb03.cells = [
    M("# Feature Engineering\n"
      "Computes features for each selected pair over the full price history.\n\n"
      "**Hedge ratios (evec0/evec1) were estimated on in-sample data in notebook 02 and are frozen here.**\n"
      "Rolling statistics use backward-looking windows — no lookahead."),

    C("""\
import sys, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
sys.path.append('..')
from src.feature_engineering import hurst_exponent"""),

    C("""\
# ── IN-SAMPLE / OUT-OF-SAMPLE BOUNDARY ───────────────────────────────────────
IS_END         = '2023-12-31'
OOS_START      = '2024-01-01'
ROLLING_WINDOW = 252
# ─────────────────────────────────────────────────────────────────────────────"""),

    C("""\
selected_pairs = pd.read_csv('../data_processed/selected_pairs.csv', index_col=0)
prices_all     = pd.read_csv('../data_raw/prices.csv', index_col=0, parse_dates=True).dropna(how='all')
print(f'Price data: {prices_all.shape[1]} tickers, {prices_all.shape[0]} trading days')
print(f'Date range: {prices_all.index[0].date()} to {prices_all.index[-1].date()}')
print(f'Processing {len(selected_pairs)} pairs...')"""),

    C("""\
for _, row in selected_pairs.iterrows():
    t1, t2         = row['ticker1'], row['ticker2']
    evec0, evec1   = row['evec0'], row['evec1']

    aligned        = pd.DataFrame({t1: prices_all[t1], t2: prices_all[t2]}).dropna()
    log_p1         = np.log(aligned[t1])
    log_p2         = np.log(aligned[t2])

    # Spread using IS-estimated Johansen hedge ratios (frozen; no OOS data used)
    spread         = evec0 * log_p1 + evec1 * log_p2

    # Rolling z-score (252-day backward window)
    spr_mean       = spread.rolling(ROLLING_WINDOW).mean()
    spr_std        = spread.rolling(ROLLING_WINDOW).std()
    z_score        = (spread - spr_mean) / spr_std

    # Rolling ADF p-value (backward window) — ~2 min total for 15 pairs
    adf_vals       = [np.nan] * ROLLING_WINDOW
    for i in range(ROLLING_WINDOW, len(spread)):
        adf_vals.append(adfuller(spread.iloc[i - ROLLING_WINDOW : i])[1])
    adf_series     = pd.Series(adf_vals, index=spread.index, name='adf_p_252')

    # Rolling Hurst exponent (backward window)
    hurst_vals     = [np.nan] * ROLLING_WINDOW
    for i in range(ROLLING_WINDOW, len(spread)):
        hurst_vals.append(hurst_exponent(spread.iloc[i - ROLLING_WINDOW : i].values, max_lag=50))
    hurst_series   = pd.Series(hurst_vals, index=spread.index, name='hurst_252')

    # Spread return (log-return of linear combination; used in backtest, not ML)
    spread_ret     = (evec0 * log_p1.diff() + evec1 * log_p2.diff()).rename('spread_ret')

    features = pd.DataFrame({
        'z_score':       z_score,
        'spread_change': spread.diff(),
        'spread_vol_20': spread.rolling(20).std(),
        'spread_ret':    spread_ret,
        'adf_p_252':     adf_series,
        'hurst_252':     hurst_series,
    })
    features['adf_stationary'] = (features['adf_p_252'] < 0.05).astype(int)
    features['hurst_mr']       = (features['hurst_252'] < 0.5).astype(int)
    features['regime_score']   = features['adf_stationary'] + features['hurst_mr']

    features = features.dropna()
    features.to_csv(f'../data_processed/features_{t1}_{t2}.csv')
    oos_n = (features.index >= OOS_START).sum()
    print(f'  {t1}/{t2}: {len(features)} rows | IS: {len(features) - oos_n} | OOS: {oos_n}')

print('Feature engineering complete.')"""),
]
write(nb03, "notebooks/03_feature_engineering.ipynb")


# ─── Notebook 04: Label Creation ─────────────────────────────────────────────
nb04 = nbf.v4.new_notebook()
nb04.cells = [
    M("# Label Creation\n"
      "Label y = 1 if |z-score| decreases over the next 5 trading days (spread moves toward zero).\n\n"
      "Labels use future data by construction — this is intentional for supervised learning.\n"
      "The key discipline is that labels are only used *for training* on in-sample data."),

    C("""\
import sys, numpy as np, pandas as pd
sys.path.append('..')"""),

    C("""\
IS_END    = '2023-12-31'
OOS_START = '2024-01-01'
N_AHEAD   = 5   # prediction horizon: 5 trading days"""),

    C("""\
selected_pairs = pd.read_csv('../data_processed/selected_pairs.csv', index_col=0)

for _, row in selected_pairs.iterrows():
    t1, t2   = row['ticker1'], row['ticker2']
    features = pd.read_csv(f'../data_processed/features_{t1}_{t2}.csv',
                           index_col=0, parse_dates=True)

    z = features['z_score']
    # y=1 when |z| shrinks over the next N days => spread is mean-reverting
    future_abs_z = z.abs().shift(-N_AHEAD)
    y = (future_abs_z < z.abs()).astype(int).rename('y')

    dataset = features.join(y).dropna()
    dataset.to_csv(f'../data_processed/dataset_{t1}_{t2}.csv')

    is_n  = (dataset.index <= IS_END).sum()
    oos_n = (dataset.index >= OOS_START).sum()
    print(f'  {t1}/{t2}: total={len(dataset)} | IS={is_n} | OOS={oos_n} | y=1 rate={y.mean():.3f}')

print('Label creation complete.')"""),
]
write(nb04, "notebooks/04_label_creation.ipynb")


# ─── Notebook 05: Model Training ─────────────────────────────────────────────
nb05 = nbf.v4.new_notebook()
nb05.cells = [
    M("# Model Training\n"
      "Trains one LogisticRegression per pair on **in-sample data only** (≤ 2023-12-31).\n\n"
      "The train/test split is a hard calendar date, not a row-fraction split.\n"
      "Predictions are saved for the full period; the `split` column flags IS vs OOS."),

    C("""\
import sys, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.append('..')"""),

    C("""\
IS_END    = '2023-12-31'
OOS_START = '2024-01-01'

# Raw spread level excluded: non-stationary, different scale across pairs
ML_FEATURES = [
    'z_score', 'spread_change', 'spread_vol_20',
    'adf_p_252', 'hurst_252',
    'adf_stationary', 'hurst_mr', 'regime_score',
]"""),

    C("""\
selected_pairs = pd.read_csv('../data_processed/selected_pairs.csv', index_col=0)
summary = []

for _, row in selected_pairs.iterrows():
    t1, t2  = row['ticker1'], row['ticker2']
    dataset = pd.read_csv(f'../data_processed/dataset_{t1}_{t2}.csv',
                          index_col=0, parse_dates=True)

    X, y  = dataset[ML_FEATURES], dataset['y']

    # IN-SAMPLE split (hard calendar boundary)
    X_is  = X.loc[:IS_END];   y_is  = y.loc[:IS_END]
    X_oos = X.loc[OOS_START:]; y_oos = y.loc[OOS_START:]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(max_iter=2000, C=0.1)),
    ])
    model.fit(X_is, y_is)   # trained on IS only

    # Predictions for full period (IS in-sample fit + OOS genuine forward prediction)
    pred = pd.DataFrame({
        'y':     y,
        'proba': model.predict_proba(X)[:, 1],
        'split': 'IS',
    }, index=X.index)
    pred.loc[OOS_START:, 'split'] = 'OOS'
    pred.to_csv(f'../data_processed/predictions_{t1}_{t2}.csv')

    oos_auc = (roc_auc_score(y_oos, model.predict_proba(X_oos)[:, 1])
               if len(y_oos.unique()) == 2 else np.nan)
    summary.append({
        'pair':     f'{t1}/{t2}',
        'IS rows':  len(X_is),
        'OOS rows': len(X_oos),
        'OOS AUC':  round(oos_auc, 4),
    })

results_df = pd.DataFrame(summary)
print(results_df.to_string(index=False))
print(f'\\nMean OOS AUC: {results_df[\"OOS AUC\"].mean():.4f}')
print('Model training complete.')"""),
]
write(nb05, "notebooks/05_model_training.ipynb")


# ─── Notebook 06: Backtest & Evaluation ──────────────────────────────────────
nb06 = nbf.v4.new_notebook()
nb06.cells = [
    M("# Backtest & Evaluation\n"
      "Runs baseline Z-score and ML-filtered strategies on the **out-of-sample period only** (2024–2025).\n\n"
      "- Equal-weight portfolio across all 15 pairs\n"
      "- Transaction costs: 5 bps per leg = 10 bps round-trip per pair trade\n"
      "- Results reported with and without transaction costs"),

    C("""\
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.append('..')
from src.utils import sharpe_ratio, max_drawdown"""),

    C("""\
IS_END          = '2023-12-31'
OOS_START       = '2024-01-01'
ENTRY_Z         = 2.0
EXIT_Z          = 0.0
PROBA_THRESHOLD = 0.55
COST_PER_LEG    = 0.0005   # 5 bps per leg; 10 bps total round-trip"""),

    C("""\
def compute_positions(z, entry_z, exit_z):
    '''State-machine position computation. Avoids deprecated ffill hack.'''
    pos = np.zeros(len(z))
    cur = 0
    for i, zi in enumerate(z.values):
        if cur == 0:
            if zi < -entry_z:   cur =  1
            elif zi > entry_z:  cur = -1
        elif cur == 1:
            if zi >= exit_z:    cur =  0
        else:
            if zi <= -exit_z:   cur =  0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


def compute_ml_positions(z, proba, entry_z, exit_z, proba_threshold):
    '''ML gates entry only; exit still purely on z crossing 0.'''
    pos = np.zeros(len(z))
    cur = 0
    for i in range(len(z)):
        zi, pi = z.values[i], proba.values[i]
        if cur == 0:
            if zi < -entry_z and pi > proba_threshold:   cur =  1
            elif zi > entry_z and pi > proba_threshold:  cur = -1
        elif cur == 1:
            if zi >= exit_z:  cur = 0
        else:
            if zi <= -exit_z: cur = 0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


def get_trade_returns(pos, daily_ret):
    '''Per-trade cumulative returns (groups consecutive active days).'''
    active    = pos.shift(1).fillna(0) != 0
    if not active.any():
        return []
    new_trade = active & (~active.shift(1).fillna(False))
    tid       = new_trade.cumsum()
    tid[~active] = 0
    return [daily_ret[tid == t].sum() for t in sorted(tid[tid > 0].unique())]"""),

    C("""\
# ── Load data and compute OOS positions + returns per pair ────────────────────
selected_pairs = pd.read_csv('../data_processed/selected_pairs.csv', index_col=0)
pair_data = {}

for _, row in selected_pairs.iterrows():
    t1, t2 = row['ticker1'], row['ticker2']
    key    = f'{t1}/{t2}'

    feats = pd.read_csv(f'../data_processed/features_{t1}_{t2}.csv',
                        index_col=0, parse_dates=True)
    preds = pd.read_csv(f'../data_processed/predictions_{t1}_{t2}.csv',
                        index_col=0, parse_dates=True)

    idx         = feats.index.intersection(preds.index)
    feats, preds = feats.loc[idx], preds.loc[idx]

    z          = feats['z_score']
    spread_ret = feats['spread_ret']
    proba      = preds['proba']

    # Positions computed over full history so state carries over IS→OOS boundary
    b_pos_full = compute_positions(z, ENTRY_Z, EXIT_Z)
    m_pos_full = compute_ml_positions(z, proba, ENTRY_Z, EXIT_Z, PROBA_THRESHOLD)

    # Slice to OOS for reporting
    b_pos = b_pos_full.loc[OOS_START:]
    m_pos = m_pos_full.loc[OOS_START:]
    sr    = spread_ret.loc[OOS_START:]

    # Raw returns (no cost)
    b_ret = b_pos.shift(1).fillna(0) * sr
    m_ret = m_pos.shift(1).fillna(0) * sr

    # Returns minus transaction costs (5 bps charged each time position changes)
    b_ret_c = b_ret - b_pos.diff().fillna(0).abs() * COST_PER_LEG
    m_ret_c = m_ret - m_pos.diff().fillna(0).abs() * COST_PER_LEG

    pair_data[key] = dict(b_pos=b_pos, m_pos=m_pos,
                          b_ret=b_ret, m_ret=m_ret,
                          b_ret_c=b_ret_c, m_ret_c=m_ret_c)

print(f'Loaded {len(pair_data)} pairs, OOS period: {OOS_START} to present')"""),

    C("""\
# ── Compute portfolio-level + trade-level metrics ─────────────────────────────
def summarize(label, ret_key, pos_key, cost_tag):
    all_ret = pd.concat({k: v[ret_key] for k, v in pair_data.items()}, axis=1).fillna(0)
    port    = all_ret.mean(axis=1)   # equal-weight across 15 pairs
    eq      = (1 + port).cumprod()

    trades  = []
    for v in pair_data.values():
        trades.extend(get_trade_returns(v[pos_key], v[ret_key]))

    n_trades  = len(trades)
    win_rate  = float(np.mean([t > 0 for t in trades])) if trades else np.nan
    avg_trade = float(np.mean(trades)) if trades else np.nan

    return {
        'Strategy':        label,
        'Costs':           cost_tag,
        'Total Return':    round(float(eq.iloc[-1] - 1), 4),
        'Sharpe (ann.)':   round(sharpe_ratio(port), 3),
        'Max Drawdown':    round(max_drawdown(eq), 4),
        'Trade Count':     n_trades,
        'Win Rate':        round(win_rate, 3) if not np.isnan(win_rate) else np.nan,
        'Avg Trade Ret':   round(avg_trade, 5) if not np.isnan(avg_trade) else np.nan,
    }

results = pd.DataFrame([
    summarize('Baseline Z-score', 'b_ret',   'b_pos', 'None'),
    summarize('Baseline Z-score', 'b_ret_c', 'b_pos', '10 bps RT'),
    summarize('ML-filtered',      'm_ret',   'm_pos', 'None'),
    summarize('ML-filtered',      'm_ret_c', 'm_pos', '10 bps RT'),
])

print('\\n=== OOS Performance (2024-01-01 onwards) ===\\n')
print(results.to_string(index=False))"""),

    C("""\
# ── Per-pair trade count breakdown ────────────────────────────────────────────
tc_rows = []
for key, v in pair_data.items():
    b_t = get_trade_returns(v['b_pos'], v['b_ret'])
    m_t = get_trade_returns(v['m_pos'], v['m_ret'])
    tc_rows.append({'Pair': key, 'Baseline trades': len(b_t), 'ML trades': len(m_t)})

tc_df = pd.DataFrame(tc_rows)
print(tc_df.to_string(index=False))
print(f"\\nTotal baseline trades: {tc_df['Baseline trades'].sum()}")
print(f"Total ML trades:       {tc_df['ML trades'].sum()}")"""),

    C("""\
# ── Equity curves ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

for ret_key, pos_key, label, color, ls in [
    ('b_ret',   'b_pos', 'Baseline (no costs)',      'steelblue',   '-'),
    ('b_ret_c', 'b_pos', 'Baseline (10 bps RT)',     'steelblue',   '--'),
    ('m_ret',   'm_pos', 'ML-filtered (no costs)',   'darkorange',  '-'),
    ('m_ret_c', 'm_pos', 'ML-filtered (10 bps RT)',  'darkorange',  '--'),
]:
    port = pd.concat({k: v[ret_key] for k, v in pair_data.items()}, axis=1).fillna(0).mean(axis=1)
    eq   = (1 + port).cumprod()
    axes[0].plot(eq.index, eq.values, label=label, color=color, ls=ls, lw=1.4)

axes[0].axhline(1, color='black', lw=0.6, ls=':')
axes[0].set_title('Equal-Weight Portfolio Equity Curves — OOS 2024-2025', fontsize=11)
axes[0].set_ylabel('Equity (normalised to 1)')
axes[0].legend(fontsize=8)

# Trade counts per pair
tc_df.set_index('Pair')[['Baseline trades', 'ML trades']].plot(
    kind='bar', ax=axes[1], width=0.7, alpha=0.8)
axes[1].set_title('OOS Trade Count per Pair', fontsize=11)
axes[1].set_ylabel('# Trades')
axes[1].tick_params(axis='x', labelsize=7, rotation=45)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../data_processed/oos_results.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved oos_results.png')"""),
]
write(nb06, "notebooks/06_backtest_and_evaluation.ipynb")

print("All notebooks generated.")
