"""Fig1h: pooled peptide-barcode MS monomeric yield vs individual HPLC-SEC yield (log10 mg/L, 48 proteins), dashed line x = y.
Tables: S14a (HPLC yield from the cytochrome C conversion, pooled MS yield of the same protein from the main library, S04)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from common.tables import load, outdir
from common.style import set_style

set_style()
df = load("S14a")
x_col, y_col = "ms_monomer_log10_mgL", "hplc_log10_yield_mgL"
d = df[[x_col, y_col]].dropna()
r = stats.pearsonr(d[x_col], d[y_col])[0]
rho = stats.spearmanr(d[x_col], d[y_col])[0]
print(f"n={len(d)} r={r:.3f} rho={rho:.3f}")

fig, ax = plt.subplots(figsize=(3 / 2.54, 3 / 2.54), dpi=300)
sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, s=6, ax=ax, color="#0493C2")
ax.text(0.05, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}\nn = {len(d)}", transform=ax.transAxes, va="top")
ax.text(0.9, 1, "x = y", transform=ax.transAxes, va="top")
ax.plot([-1, 1.4], [-1, 1.4], linestyle="--", linewidth=1, label="x=y", color="grey")
ax.set_xlabel("Peptide barcode based pooled assay\n" + r"log$_{10}$(monomeric yield) (mg/L)")
ax.set_ylabel("HPLC based individual assay\n" + r"log$_{10}$(monomeric yield) (mg/L)")
ax.tick_params(axis="both", which="major", pad=1)
ax.set_xticks([-2, -1, 0, 1])
ax.set_xlim(-2.5, 1.6)
ax.set_ylim(-1.2, 1.6)
fig.savefig(os.path.join(outdir("fig1"), "Fig1h.pdf"), bbox_inches="tight")
