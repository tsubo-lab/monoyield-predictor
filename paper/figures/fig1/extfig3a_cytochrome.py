"""ExtFig3a: cytochrome C calibration of the HPLC-SEC assay.
chromatograms = LC traces of the dilution series (two replicates + blank), coloured by injected mass, integration window 33-40 min;
calibration   = log10(AUC) vs log10(injected mass) (mg).
Tables: hplc cytochromeC wide chromatograms (common.tables.hplc) for the traces, S14c for the calibration points
(6th dilution = 0.03125 mg/mL, i.e. 0.003125 mg injected). S14b gives the resulting conversion (printed as a check)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common.tables import load, hplc, outdir
from common.style import set_style

set_style(arial_math=True)
RT_WINDOW = (33, 40)
out = outdir("fig1")

chrom = hplc("cytochromeC")
samples = [c for c in chrom.columns if c != "R_Time"]
mass_mg = {c: 0.0 if c.startswith("blank") else float(c.split("_")[1].replace("mg", "")) for c in samples}  # mg injected (100 uL)
unique_masses = sorted(set(mass_mg.values()), reverse=True)
color_of = {m: plt.cm.viridis(i / (len(unique_masses) - 1)) for i, m in enumerate(unique_masses)}

fig = plt.figure(figsize=(7 / 2.54, 3.5 / 2.54), tight_layout=True, dpi=300)
for c in samples:
    plt.plot(chrom["R_Time"], chrom[c], color=color_of[mass_mg[c]], lw=0.5)
plt.xlabel("Retention Time (min)")
plt.ylabel("Intensity (mV)")
plt.title("LC Chromatograms")
plt.vlines(RT_WINDOW, 0, 2.5e6, color="red", linestyle="--", lw=0.5)
plt.xlim(0, 50)
plt.tight_layout()
fig.savefig(os.path.join(out, "ExtFig3a-chromatograms.pdf"))

cal = load("S14c")
res = stats.linregress(cal["log10_AUC"], cal["log10_input_mass_mg"])
conv = load("S14b").iloc[0]
print(f"calibration: n={len(cal)} slope={res.slope:.4f} intercept={res.intercept:.4f} r={res.rvalue:.4f}  (S14b: {conv['slope']:.4f} / {conv['intercept']:.4f})")

fig = plt.figure(figsize=(4 / 2.54, 3.5 / 2.54), tight_layout=True, dpi=300)
ax = fig.add_subplot(111)
colors = [color_of[min(unique_masses, key=lambda m: abs(np.log10(m) - v) if m > 0 else np.inf)] for v in cal["log10_input_mass_mg"]]
ax.scatter(cal["log10_AUC"], cal["log10_input_mass_mg"], c=colors, s=6, alpha=0.8, linewidths=0.2, edgecolors="white")
ax.set_xlabel(r"$\log_{10}$(AUC)")
ax.set_ylabel(r"$\log_{10}$(input mass) mg")
ax.set_xticks(np.arange(4.5, 6.5, 0.5))
ax.set_yticks(np.arange(-3, -0.8, 0.5))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.tick_params(axis="both", which="major", pad=1)
fig.savefig(os.path.join(out, "ExtFig3a.pdf"))
