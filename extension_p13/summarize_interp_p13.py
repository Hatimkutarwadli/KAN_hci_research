"""P1-3: tables and figures from results/p1_kan_mlp_control/interpretability/*.csv (SYNTHETIC; EXTENSION). Reads only; writes
interpretability/{interpretability_tables.md, fig_pdp_shapes.png, fig_kan_edges.png, fig_sensitivity.png, fig_lag_profile.png}.
Usage: python extension_p13/summarize_interp_p13.py"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "extension_p13")]
os.chdir(ROOT)
D = "results/p1_kan_mlp_control/interpretability"
CH = ["steps", "calories", "sleep_duration", "sleep_efficiency", "sleep_variability", "resting_hr", "hrv", "circadian_regularity"]
PLANTED = {("dep", "sleep_variability"): +1, ("dep", "steps"): -1, ("anx", "steps"): -1, ("anx", "hrv"): -1}
G = {"sleep_variability": lambda d: np.tanh(d), "steps": lambda d: np.tanh(0.7 * d), "hrv": lambda d: np.tanh(0.8 * d)}
ORDER = ["gru", "kan", "kan_reg", "kan_nospline", "mlp_b2", "mlp_b2_reg", "m1", "mlp_m1"]
ALL = [m for m in ORDER] + [m + "_untrained" for m in ("gru", "kan", "kan_nospline", "mlp_b2", "m1", "mlp_m1")]
COL = {"gru": "#444444", "kan": "#d95f02", "kan_reg": "#e6ab02", "kan_nospline": "#8c564b", "mlp_b2": "#1f78b4", "mlp_b2_reg": "#6baed6", "m1": "#c51b7d", "mlp_m1": "#1b9e77"}
LS = {"kan_nospline": "--", "kan_reg": "-.", "mlp_b2_reg": "-."}
PAIRS = [("kan", "mlp_b2"), ("kan_reg", "mlp_b2_reg"), ("m1", "mlp_m1"), ("kan", "kan_nospline"), ("kan", "gru"), ("mlp_b2", "gru"), ("m1", "gru"), ("mlp_m1", "gru")]
HIGHER = {"sign_acc": True, "shape_r": True, "recall_top3": True, "mean_rank_planted": False}


def agg_seed(rec, heads):
    r = rec[rec["head"].isin(heads)].groupby(["model", "seed"]).agg(sign_acc=("sign_acc", "mean"), shape_r=("shape_r", "mean"), mean_rank_planted=("mean_rank_planted", "mean"),
                                                                     recall_top3=("recall_top3", "mean"), calories_above_steps=("calories_above_steps", "mean")).reset_index()
    return r


def table_recovery(r):
    L = ["| model | sign accuracy (chance 0.5) | shape r with planted g | mean rank of planted channels (chance 4.5; 1 = best) | planted in top 3 (chance 0.375) | calories ranked above steps |", "|---|---|---|---|---|---|"]
    for m in ALL:
        x = r[r.model == m]
        if x.empty:
            continue
        f = lambda c: f"{x[c].mean():.2f} ±{x[c].std():.2f}"      # noqa: E731
        L.append(f"| {m} | {f('sign_acc')} | {f('shape_r')} | {f('mean_rank_planted')} | {f('recall_top3')} | {f('calories_above_steps')} |")
    return "\n".join(L)


def table_paired(r):
    L = ["| comparison (first vs second) | metric | mean difference (first - second) | seeds where first is better |", "|---|---|---|---|"]
    for a, b in PAIRS:
        for k, hi in HIGHER.items():
            p = r.pivot(index="seed", columns="model", values=k)
            if a not in p or b not in p:
                continue
            d = (p[a] - p[b])
            better = int((d > 0).sum() if hi else (d < 0).sum())
            L.append(f"| {a} vs {b} | {k} | {d.mean():+.3f} (SD {d.std():.3f}) | {better}/{len(d)} |")
    return "\n".join(L)


def main():
    rec = pd.read_csv(f"{D}/recovery_per_seed.csv")
    sens = pd.read_csv(f"{D}/sensitivity.csv")
    pdp = pd.read_csv(f"{D}/pdp_curves.csv")
    lag = pd.read_csv(f"{D}/lag_profile.csv")
    edge = pd.read_csv(f"{D}/kan_edge_importance.csv")
    assoc = pd.read_csv(f"{D}/association_baseline.csv")
    L = ["# P1-3 interpretability tables (SYNTHETIC; EXTENSION). Calibration split, fixed 2,000 origins, window-shift analysis. Recovered sign/shape/rank describe the PREDICTOR, not a causal effect.\n"]

    # ---- 1. recovery summary
    rs = agg_seed(rec, ["sev_dep_t0", "sev_anx_t0"])
    L += ["\n## 1. Recovery of the planted structure, severity heads (dep: sleep variability +, steps -; anx: steps -, HRV -); mean ±SD over 5 seeds\n", table_recovery(rs)]
    rst = agg_seed(rec, ["status_dep_t0", "status_anx_t0"])
    L += ["\n## 1b. Same, status (probability) heads\n", table_recovery(rst)]
    L += ["\n## 2. Paired seed comparisons (severity heads). KAN vs matched MLP = interpretability difference at matched parameters; KAN vs no-spline = contribution of the splines\n", table_paired(rs)]

    # ---- 3. per planted pair
    L += ["\n## 3. Per planted pair, severity heads: seeds with correct slope sign (of 5) / mean shape r / mean rank among 8 channels\n", "| model | " + " | ".join(f"{k}: {c}" for (k, c) in PLANTED) + " |", "|---|---|---|---|---|"]
    for m in ALL:
        cells = []
        for (k, c), sg in PLANTED.items():
            h = f"sev_{k}_t0"
            s_ = sens[(sens.model == m) & (sens["head"] == h)]
            if s_.empty:
                cells.append("")
                continue
            rows = []
            for sd, x in s_.groupby("seed"):
                x = x.set_index("channel")
                rk = x.sens.rank(ascending=False)[c]
                p = pdp[(pdp.model == m) & (pdp.seed == sd) & (pdp["head"] == h) & (pdp.channel == c)].sort_values("delta").value.values
                vc = p - p[len(p) // 2]
                r = np.corrcoef(vc, sg * G[c](np.linspace(-2, 2, 9)))[0, 1] if vc.std() > 1e-12 else 0.0
                rows.append((float(np.sign(x.slope_central[c]) == sg and x.sens[c] > 1e-4), r, rk))
            a = np.array(rows)
            cells.append(f"{int(a[:, 0].sum())}/5, r={a[:, 1].mean():.2f}, rank {a[:, 2].mean():.1f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    # ---- 4. channel ranking by model-sensitivity vs association-only baseline
    L += ["\n## 4. Most sensitive channel (window-shift range / SD of predictions): counts over 5 seeds, severity heads\n", "| model | head | " + " | ".join(CH) + " |", "|---|---|" + "---|" * 8]
    for m in ALL:
        for h in ("sev_dep_t0", "sev_anx_t0"):
            s_ = sens[(sens.model == m) & (sens["head"] == h)]
            if s_.empty:
                continue
            cnt = s_.loc[s_.groupby("seed").sens.idxmax()].channel.value_counts()
            L.append(f"| {m} | {h} | " + " | ".join(str(int(cnt.get(c, 0))) for c in CH) + " |")
    L += ["\n### Association-only reference (no model): Pearson r between each channel's window mean and the severity target, calibration origins\n", "| head | " + " | ".join(CH) + " |", "|---|" + "---|" * 8]
    for h, x in assoc.groupby("head"):
        x = x.set_index("channel")
        L.append(f"| {h} | " + " | ".join(f"{x.pearson_r[c]:+.3f}" for c in CH) + " |")
    ab = []
    for (k, c), sg in PLANTED.items():
        x = assoc[assoc["head"] == f"sev_{k}_t0"].set_index("channel").pearson_r
        ab.append((k, c, float(np.sign(x[c]) == sg), int(x.abs().rank(ascending=False)[c])))
    L += ["\nAssociation-only baseline vs planted pairs: " + "; ".join(f"{k}/{c}: sign {'ok' if s else 'WRONG'}, |r| rank {r}" for k, c, s, r in ab) + "\n"]

    # ---- 5. KAN first-layer edge importance
    L += ["\n## 5. KAN first-layer edge importance (mean |phi| over x in [-3, 3], summed over outputs, lags and the 3 scales), share of total, mean over 5 seeds\n", "| model | " + " | ".join(CH + ["missing_day", "hrv_missing"]) + " |", "|---|" + "---|" * 10]
    for m in ("kan", "kan_nospline", "m1"):
        e = edge[edge.model == m].groupby(["seed", "channel"]).edge_importance.sum().unstack()
        sh = e.div(e.sum(1), axis=0).mean()
        L.append(f"| {m} | " + " | ".join(f"{sh[c]:.3f}" for c in CH + ["missing_day", "hrv_missing"]) + " |")
    L += ["\nSpearman correlation between first-layer edge importance and window-shift sensitivity over the 8 value channels (per seed, mean ±SD):\n", "| model | head | rho |", "|---|---|---|"]
    for m in ("kan", "kan_nospline", "m1"):
        e = edge[edge.model == m].groupby(["seed", "channel"]).edge_importance.sum().unstack()[CH]
        for h in ("sev_dep_t0", "sev_anx_t0"):
            rho = []
            for sd in e.index:
                sv = sens[(sens.model == m) & (sens.seed == sd) & (sens["head"] == h)].set_index("channel").sens[CH]
                rho.append(e.loc[sd].rank().corr(sv.rank()))
            L.append(f"| {m} | {h} | {np.mean(rho):+.2f} ±{np.std(rho, ddof=1):.2f} |")

    # ---- 6. lag profile
    L += ["\n## 6. Lag profile: mean change of the severity prediction for +1 SD on ONE day t-lag (mean over 5 seeds); planted lags are 1-2 days and the latent state integrates them, so a profile concentrated on recent days is plausible but not a lag estimate\n"]
    for h, c in (("sev_dep_t0", "sleep_variability"), ("sev_dep_t0", "steps"), ("sev_anx_t0", "steps"), ("sev_anx_t0", "hrv")):
        L += [f"\n{h} / {c}\n", "| model | " + " | ".join(f"lag {l}" for l in sorted(lag.lag.unique())) + " |", "|---|" + "---|" * lag.lag.nunique()]
        for m in ORDER:
            x = lag[(lag.model == m) & (lag["head"] == h) & (lag.channel == c)].groupby("lag").effect.mean()
            if len(x):
                L.append(f"| {m} | " + " | ".join(f"{v:+.3f}" for v in x.values) + " |")
    open(f"{D}/interpretability_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---- figures
    dl = np.linspace(-2, 2, 9)
    fam = {"B2 family": ["kan", "mlp_b2", "kan_nospline", "gru"], "M1 family": ["m1", "mlp_m1", "gru"]}
    fig, ax = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    for i, ((k, c), sg) in enumerate(PLANTED.items()):
        for j, (fn, ms) in enumerate(fam.items()):
            a = ax[i, j]
            for m in ms:
                cur = []
                for sd in range(5):
                    p = pdp[(pdp.model == m) & (pdp.seed == sd) & (pdp["head"] == f"sev_{k}_t0") & (pdp.channel == c)].sort_values("delta").value.values
                    vc = p - p[4]
                    cur.append(vc / (np.abs(vc).max() + 1e-12))
                cur = np.array(cur)
                a.plot(dl, cur.mean(0), color=COL[m], ls=LS.get(m, "-"), label=m, lw=1.8)
                a.fill_between(dl, cur.min(0), cur.max(0), color=COL[m], alpha=0.12)
            ref = sg * G[c](dl)
            a.plot(dl, ref / np.abs(ref).max(), color="green", ls=":", lw=2.5, label="planted g (sign x shape)")
            a.axhline(0, color="#bbbbbb", lw=0.6)
            a.set_title(f"sev_{k}_t0 vs {c} ({fn})", fontsize=9)
            if i == 3:
                a.set_xlabel("window-wide shift of channel (SD units)")
            if j == 0:
                a.set_ylabel("normalised partial dependence")
            if i == 0:
                a.legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: learned window-shift response vs planted transform (curves scaled to max |.| = 1; band = min-max over 5 seeds). Association learned by the predictor, not a causal effect", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_pdp_shapes.png", dpi=130)
    plt.close(fig)

    npz = np.load(f"{D}/kan_edge_functions.npz")
    grid = npz["grid"]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for i, (lab, key) in enumerate((("B2 KAN, seed 0, 5-day encoder", "kan|0|5"), ("M1 KAN, seed 0, 5-day encoder", "m1|0|5"))):
        phi = npz[key]                                                                   # (grid, o, C, k)
        for j, c in enumerate(("sleep_variability", "steps", "hrv", "sleep_duration")):
            a = ax[i, j]
            Phi = phi[:, :, CH.index(c), :].sum(-1)                                       # response of each encoder output to a window-wide shift
            for o in range(Phi.shape[1]):
                a.plot(grid, Phi[:, o], lw=1.1)
            a.axhline(0, color="#bbbbbb", lw=0.6)
            a.set_title(f"{lab}\nchannel: {c}", fontsize=8)
            a.set_xlabel("input value (standardised)")
    fig.suptitle("SYNTHETIC data: first-layer KAN edge functions summed over the 5 days of the patch, one curve per encoder output. Shapes of learned univariate functions; not causal effects", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_kan_edges.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, h in zip(ax, ("sev_dep_t0", "sev_anx_t0")):
        M = []
        rows = [m for m in ORDER + ["mlp_m1_untrained", "kan_untrained"] if not sens[(sens.model == m) & (sens["head"] == h)].empty]
        for m in rows:
            v = sens[(sens.model == m) & (sens["head"] == h)].groupby("channel").sens.mean()[CH]
            M.append((v / v.sum()).values)
        ac = assoc[assoc["head"] == h].set_index("channel").pearson_r.abs()[CH]
        M.append((ac / ac.sum()).values)
        rows.append("association |r| (no model)")
        im = a.imshow(np.array(M), aspect="auto", cmap="viridis")
        a.set_xticks(range(8))
        a.set_xticklabels([("* " if (h[4:7], c) in PLANTED else "") + c for c in CH], rotation=40, ha="right", fontsize=8)
        a.set_yticks(range(len(rows)))
        a.set_yticklabels(rows, fontsize=8)
        a.set_title(f"{h}: share of window-shift sensitivity (rows sum to 1); * = planted", fontsize=9)
        plt.colorbar(im, ax=a)
    fig.suptitle("SYNTHETIC data: which channels each model reacts to. Calories (= 0.7 x steps) and circadian regularity (= -0.3 x sleep variability) are planted collinear correlates", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_sensitivity.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
    for a, (h, c) in zip(ax, (("sev_dep_t0", "sleep_variability"), ("sev_dep_t0", "steps"), ("sev_anx_t0", "steps"), ("sev_anx_t0", "hrv"))):
        for m in ("gru", "kan", "mlp_b2", "m1", "mlp_m1"):
            x = lag[(lag.model == m) & (lag["head"] == h) & (lag.channel == c)].groupby("lag").effect.agg(["mean", "std"])
            a.errorbar(x.index, x["mean"], yerr=x["std"], color=COL[m], label=m, lw=1.4, capsize=2)
        a.axhline(0, color="#bbbbbb", lw=0.6)
        a.set_title(f"{h} / {c}", fontsize=9)
        a.set_xlabel("lag (days before the origin)")
    ax[0].set_ylabel("prediction change for +1 SD on that day")
    ax[0].legend(fontsize=7)
    fig.suptitle("SYNTHETIC data: single-day lag profile (mean ±SD over 5 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_lag_profile.png", dpi=130)
    plt.close(fig)
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
