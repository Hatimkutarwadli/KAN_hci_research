"""Example-participant figure for a simulated dataset dir. SYNTHETIC."""
import sys, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d, out = sys.argv[1], sys.argv[2]
r = lambda n: pd.read_csv(os.path.join(d, n + ".csv"))
tr, tg, ob, tp = r("truth"), r("targets"), r("observations"), r("truth_participants")
pids = tp[tp.onset_day_dep.notna()].pid.values[:4]
fig, ax = plt.subplots(3, 4, figsize=(15, 7), sharex=True)
for j, p in enumerate(pids):
    a = tr[tr.pid == p]; b = tg[tg.pid == p]; o = ob[ob.pid == p]
    ax[0, j].plot(a.day, a.z_dep, label="true z_dep"); ax[0, j].plot(a.day, a.z_anx, label="true z_anx", alpha=.6)
    ax[0, j].axhline(2.45, ls=":", c="k"); od = tp.loc[p, "onset_day_dep"]; ax[0, j].axvline(od, c="r", ls="--")
    ax[0, j].set_title(f"participant {p} (onset day {int(od)})")
    ax[1, j].plot(b.day, b.severity_dep, c="C2"); ax[1, j].set_ylabel("severity_dep")
    ax[2, j].plot(o.day, o.steps, ".-", ms=3, lw=.5, label="steps (NaN=gap)"); ax[2, j].set_xlabel("day")
ax[0, 0].legend(fontsize=7)
fig.suptitle("SYNTHETIC pilot data: example participants (dashed red = first dep onset, dotted = latent threshold)")
fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=110)
