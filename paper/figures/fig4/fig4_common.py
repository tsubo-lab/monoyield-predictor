"""Shared helpers for the prediction panels of Fig. 4 / Extended Data Figs 10-12 (models, colours, hexbin panel, Wilson CI).

Tables: S15a (observed yield + test-fold predictions of all models, name level) and S15b (mutational effect = mutant - WT).
Model keys are the column names of S15a/S15b; the notebook keys are given in NB_KEY.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from scipy import stats
from sklearn.metrics import mean_squared_error
from common.tables import load, outdir
from common.style import set_style

BLUE_SHADES = ['#CDE3F1', '#8FC1E1', '#4F9ED0', '#0072BC', '#05568B', '#0B3754']
GREEN_SHADES = ['#DAF6EB', '#7DE0B9', '#32C189', '#2B9C70', '#257958', '#1F5741']
PURPLE_SHADES = ['#E7E5F7', '#C3BEED', '#A097E3', '#7E74C2', '#5E578E', '#403C5D']
RED_SHADES = ['#FCE2DE', '#F9B3A7', '#F67F6A', '#D25A45', '#9C4637', '#683229']
_cm = lambda shades: mcolors.LinearSegmentedColormap.from_list("c", shades)
bc_, gc_, pc_ = _cm(BLUE_SHADES), _cm(GREEN_SHADES), _cm(PURPLE_SHADES)
dark = lambda shades: mcolors.LinearSegmentedColormap.from_list("d", _cm(shades)(np.linspace(0, 1.0, 256)))
dark_blues, dark_greens, dark_purples, dark_reds = (dark(s) for s in (BLUE_SHADES, GREEN_SHADES, PURPLE_SHADES, RED_SHADES))

TOOLS = ['Camsol', 'DeepSoluE', 'MPEPE', 'NetSolP', 'RP3net', 'SKADE', 'SoluProt', 'AGGRESCAN3D', 'GATsol']
# notebook key -> S15 column (paper label)
XGB = {'feature': 'Features (XGBoost)', 'saprot_structure': 'SaProt 1.3B (XGBoost)', 'esm2': 'ESM2 3B (XGBoost)',
       'esm3': 'ESMC 6B(XGBoost)', 'saprot_structure_f': 'SaProt 1.3B + features (XGBoost)',
       'esm2_f': 'ESM2 3B + features (XGBoost)', 'esm3_f': 'ESMC 6B + features (XGBoost)',
       'feature_d': 'Features + ΔG (XGBoost)', 'saprot_structure_d': 'SaProt 1.3B + ΔG (XGBoost)',
       'esm2_d': 'ESM2 3B + ΔG (XGBoost)', 'esm3_d': 'ESMC 6B + ΔG (XGBoost)'}
LORA = {'ESM3': 'ESM3-open 1.4B (LoRA)', 'ESMC 600M': 'ESMC 600M (LoRA)', 'ESMC 6B': 'ESMC-MonoYield',
        'Saprot 1.3B': 'SaProt 1.3B (LoRA)'}
LINEAR = {'ESM3 linear': 'ESM3-open 1.4B (linear head)', 'Saprot 1.3B linear': 'SaProt 1.3B (linear head)',
          'ESMC 6B linear': 'ESMC 6B (linear head)'}
NB_KEY = {**{k: v for k, v in XGB.items()}, **LORA, **LINEAR, **{t: t for t in TOOLS}}   # key -> column
KEYS = list(XGB)

def classify_group(m):
    if m in TOOLS: return f'Tool: {m}'
    if m in LORA: return f'LoRA: {m}'
    if m not in XGB: return 'LoRA: linear head'
    if m.endswith('_f'): return 'XGBoost (fine-tuned, _f)'
    if m.endswith('_d'): return 'XGBoost (delta, _d)'
    return 'XGBoost (base)'

_lg = sorted(f'LoRA: {m}' for m in LORA)
_tg = sorted(f'Tool: {m}' for m in TOOLS)
_blues = [bc_(x) for x in np.linspace(0.2, 0.6, 3)]
GROUP_PALETTE = {**dict(zip(_lg, [pc_(x) for x in np.linspace(0.2, 0.7, len(_lg))])),
                 **dict(zip(_tg, [gc_(x) for x in np.linspace(0.2, 0.7, len(_tg))])),
                 'XGBoost (base)': _blues[0], 'XGBoost (fine-tuned, _f)': _blues[1], 'XGBoost (delta, _d)': _blues[2],
                 'LoRA: linear head': PURPLE_SHADES[2]}
GROUP_PALETTE['LoRA: ESMC 6B'] = RED_SHADES[3]

MODEL_LABEL = {**{k: v for k, v in XGB.items()}, 'ESM3': 'ESM3-open 1.4B (LoRA)', 'ESMC 600M': 'ESMC 600M (LoRA)',
               'ESMC 6B': 'ESMC-MonoYield', 'Saprot 1.3B': 'SaProt 1.3B (LoRA)', **LINEAR}

def set_style_pred():
    """Style of the prediction notebooks: user style + tight label / tick padding."""
    set_style()
    plt.rcParams.update({"axes.labelpad": 1, "axes.titlepad": 1, "xtick.major.pad": 1, "ytick.major.pad": 1,
                         "xtick.minor.pad": 1, "ytick.minor.pad": 1})

def load_pred():
    """S15a / S15b with the true-value column renamed to y_true / y_delta_true; returns (all rows, denovo rows, mutational effect)."""
    a = load("S15a").rename(columns={"true_monomer_log_mgL": "y_true", "Aggrescan3D": "AGGRESCAN3D"})
    b = load("S15b").rename(columns={"true_delta_monomer_log_mgL": "y_delta_true", "Aggrescan3D": "AGGRESCAN3D"})
    return a, a[a["data_type"] == "denovo"], b

def condition_data(model, dt, a, den, b):
    """(prediction, truth) for a model key and data type 'all_data' / 'denovo' / 'dms', dropping missing predictions."""
    col = NB_KEY[model]
    d = {'all_data': a, 'denovo': den, 'dms': b}[dt]
    tc = 'y_delta_true' if dt == 'dms' else 'y_true'
    d = d[[col, tc]].dropna()
    return d[col].to_numpy(), d[tc].to_numpy()

def hexbin_panel(ax, y_pred, y_true, dt, same_scale=True, cmap=None):
    ax.set_box_aspect(1)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred, y_true = y_pred[mask], y_true[mask]
    r = stats.pearsonr(y_pred, y_true)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.hexbin(y_pred, y_true, gridsize=60, mincnt=1, bins='log', cmap=cmap if cmap is not None else dark_blues, alpha=1.0)
    xlim = {'all_data': (-2.5, 2.0), 'denovo': (-2.5, 2.0), 'dms': (-2.0, 1.0)}.get(dt)
    if same_scale:
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
        d0 = min(ax.get_xlim()[0], y_true.min()); d1 = max(ax.get_xlim()[1], y_true.max())
        ax.plot([d0, d1], [d0, d1], color='grey', linestyle='--', linewidth=0.5, scaley=False)
    else:
        xr = y_pred.max() - y_pred.min()
        xp = xr * 0.05 if xr > 0 else 0.5
        ax.set_xlim(y_pred.min() - xp, y_pred.max() + xp)
        ax.locator_params(axis='x', nbins=4)
    ax.text(0.5, 1.01, f"r = {r:.2f}   RMSE = {rmse:.2f}", transform=ax.transAxes, ha='center', va='bottom', fontsize=6, clip_on=False)   # 2026-11-02: stats above the panel, n removed
    ax.set_title({'all_data': 'All data', 'denovo': 'De novo', 'dms': 'Mutational effect \nΔ(mutant-WT)'}.get(dt, dt), fontsize=6, pad=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
    ax.tick_params(axis='both', which='major', labelsize=6, pad=1)
    return r

def hexbin_cmap_for(model):
    if model == 'ESMC 6B': return dark_reds
    g = classify_group(model)
    if g.startswith('LoRA: '): return dark_purples
    if g.startswith('Tool: '): return dark_greens
    return dark_blues

def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)

def precision_at(y_true, y_pred, threshold, fracs):
    """Notebook evaluate_positive_prediction: positives (>= threshold) among the top ceil(n*frac) predictions."""
    d = pd.DataFrame({'t': y_true, 'p': y_pred}).dropna()
    pos = (d['t'] >= threshold).astype(int).to_numpy()[np.argsort(-d['p'].to_numpy(), kind='quicksort')]
    n = len(d)
    out = []
    for f in fracs:
        k = max(1, int(np.ceil(n * f)))
        out.append(dict(top_percent=f * 100, k=k, n_total=n, n_positive=int(pos.sum()), base=pos.sum() / n,
                        n_pos_top=int(pos[:k].sum()), precision=pos[:k].sum() / k))
    return pd.DataFrame(out)
