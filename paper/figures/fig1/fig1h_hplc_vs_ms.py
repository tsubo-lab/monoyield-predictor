"""Fig1h: pooled peptide-barcode MS monomeric yield vs individual HPLC-SEC yield (log10 mg/L, 48 proteins), dashed line x = y.
Tables: S14a (HPLC yield from the cytochrome C conversion, pooled MS yield of the same protein from the main library, S04)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from common.tables import load, outdir
from common.style import set_style

set_style(arial_math=True)
df = load("S14a")
x_col, y_col = "ms_monomer_log10_mgL", "hplc_log10_yield_mgL"
data = df.groupby("name", as_index=False)[[x_col, y_col]].mean().replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
x, y = data[x_col], data[y_col]
r, rho, n = pearsonr(x, y)[0], spearmanr(x, y)[0], len(x)
print(f"n={n} r={r:.3f} rho={rho:.3f}")

fig, ax = plt.subplots(figsize=(5 / 2.54, 4 / 2.54), dpi=300, tight_layout=True)
sns.scatterplot(data=data, x=x_col, y=y_col, color="#43B6D8", alpha=0.8, s=6, linewidth=0, ax=ax)
ax.set_xlabel("Peptide barcode based pooled assay\n" + r"$\log_{10}$(monomeric yield) (mg/L)")
ax.set_ylabel("HPLC based individual assay\n" + r"$\log_{10}$(monomeric yield) (mg/L)")
ax.set_xlim(-2.5, 1.6); ax.set_ylim(-1.2, 1.6)
ax.set_xticks([-2, -1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.plot([-1, 1.4], [-1, 1.4], lw=1, ls="--", color="grey", zorder=1)
ax.text(1.05, 1.45, "y = x", ha="left", va="bottom")
ax.text(0.05, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}\nn = {n}", transform=ax.transAxes, ha="left", va="top", linespacing=1.1)
ax.set_aspect("equal", adjustable="box"); ax.set_axisbelow(True)
ax.grid(color="#eeeeee", linewidth=0.7, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(outdir("fig1"), "Fig1h.pdf"), bbox_inches="tight")
