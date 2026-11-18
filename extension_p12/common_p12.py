"""P1-2 (EXTENSION): multitask vs single-task GRU ablation. SYNTHETIC. Shared definitions + training helpers. Existing models/results are never modified.
Frozen GRUNet is reused unchanged; only the set of output heads (and hence the loss terms) differs between conditions:
  MT              : existing multitask GRU (results/stage3 main 16 heads; results/stage3c_forecast 12 heads) - NOT retrained
  st_task         : one GRU per head (n_out = 1), frozen early stopping on its OWN calibration loss
  st_task_matched : one GRU per head trained for EXACTLY the number of epochs the same-seed MT run selected (no early stopping, last epoch)
                    -> equal gradient-step budget, no selection advantage
  family          : one GRU per task FAMILY (severity / status / sustained onset), several heads sharing one representation
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
import baselines as bl  # noqa: E402
import models as md  # noqa: E402

KS = ("dep", "anx")
H = (7, 14, 21)
PARTS = {
    "main": dict(
        st=[f"sev_{k}_t0" for k in KS] + [f"status_{k}_t0" for k in KS] + [f"onset_sust_{k}_h{h}" for k in KS for h in H],
        family={"severity": [f"sev_{k}_t0" for k in KS], "status": [f"status_{k}_t0" for k in KS], "onset_sust": [f"onset_sust_{k}_h{h}" for k in KS for h in H]},
        mt_dir="results/stage3"),
    "forecast": dict(
        st=[f"sev_{k}_h{h}" for k in KS for h in H] + [f"status_{k}_h{h}" for k in KS for h in H],
        family={"severity": [f"sev_{k}_h{h}" for k in KS for h in H], "status": [f"status_{k}_h{h}" for k in KS for h in H]},
        mt_dir="results/stage3c_forecast"),
}
CFG = dict(epochs=25, patience=4, bs=256, lr=2e-3, wd=1e-5, epoch_frac=0.5)       # = frozen fit_nn defaults


def family_of(head):
    return "severity" if head.startswith("sev_") else "status" if head.startswith("status_") else "onset_sust" if head.startswith("onset_sust") else "onset_cross"


def unit_data(data_all, heads):
    """data dicts built with the 28-head layout -> same rows, Y restricted to `heads` (regression heads first, as fit_nn requires)."""
    cols = [bl.ALL_HEADS.index(h) for h in heads]
    out = {}
    for s, d in data_all.items():
        d2 = dict(d)
        d2["Y"] = d["Y"][:, cols].copy()
        out[s] = d2
    return out


def configure_unit(heads):
    heads = sorted(heads, key=lambda h: 0 if h.startswith("sev_") else 1)             # stable: regression heads first
    n_reg = sum(h.startswith("sev_") for h in heads)
    bl.TASKSETS["unit"] = (list(heads), n_reg)
    bl.configure("unit")
    return list(heads)


def fit_fixed(seed, tr, ca, epochs, bs=256, lr=2e-3, wd=1e-5, epoch_frac=0.5, log=print):
    """Same loop/RNG as baselines.fit_nn (copied, not edited) but trains EXACTLY `epochs` epochs and returns the LAST state (no selection)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = md.MODELS["gru"](n_out=len(bl.HEADS))
    y = tr["Y"][:, :bl.N_REG]
    ymean = torch.tensor(np.nanmean(y, 0), dtype=torch.float32)
    ystd = torch.tensor(np.nanstd(y, 0), dtype=torch.float32)
    Tt, Tc = bl.to_t(tr), bl.to_t(ca)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    hist, n = [], len(tr["pi"])
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)[:int(epoch_frac * n)]
        m_, run = len(perm), 0.0
        for a in range(0, m_, bs):
            x, st, yb = bl.batch(Tt, perm[a:a + bs])
            loss = bl.masked_loss(model(x, st), yb, ymean, ystd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(yb)
        cl = bl.eval_loss(model, Tc, ymean, ystd)
        hist.append({"epoch": ep, "train_loss": run / m_, "cal_loss": cl})
        log(f"  gru-fixed seed{seed} ep{ep} train {run / m_:.4f} cal {cl:.4f}")
    return model, ymean, ystd, {"best_epoch": epochs, "epochs_run": epochs, "history": hist,
                                "hparams": dict(epochs=epochs, patience=None, batch_size=bs, lr=lr, weight_decay=wd, epoch_frac=epoch_frac, selection="none (last epoch)")}


def mt_best_epochs(part):
    import json
    d = json.load(open(os.path.join(PARTS[part]["mt_dir"], "run_log.json")))
    return {r["seed"]: r["best_epoch"] for r in d["runs"] if r["model"] == "gru"}
