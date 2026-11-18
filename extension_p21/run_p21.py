"""P2-1 EXTENSION pilot (SYNTHETIC data): hybrid KAN-ODE, no-flow ablation and GRU-D-style GRU, ONE seed, CALIBRATION ONLY.
The test split is never loaded by this script. Frozen protocol = baselines.fit_nn (train-only fitting, epoch chosen on calibration loss).
Subcommands (run from anywhere; paths are relative to the project root):
  manifest --out F            sha256 of every pre-existing project file (P2-1 files excluded)
  verify   --manifest F       re-hash and compare
  bench    --out F            forward+backward timing vs M1 and GRU (runtime gate)
  train    --model M --taskset T --out DIR    (refuses a non-empty DIR)
  analyze  --pilot DIR        calibration-only comparison tables
Config: configs/p2_kanode_pilot.json"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p21")]
import baselines as bl  # noqa: E402
import metrics as mt  # noqa: E402
import models as md  # noqa: E402
import models_p21 as mp  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "configs", "p2_kanode_pilot.json")))
EXCLUDE = ("extension_p21", os.path.join("results", "p2_kanode"), os.path.join("tests", "test_p21.py"),
           os.path.join("configs", "p2_kanode_pilot.json"), "__pycache__", ".claude")
HASH_DIRS = ("data", "results", "configs", "src", "tests", "final_submission", "references") + tuple(
    d for d in sorted(os.listdir(ROOT)) if d.startswith("extension_") and d != "extension_p21")


# ------------------------------------------------------------------ integrity
def file_hashes():
    out = {}
    for top in HASH_DIRS + tuple(f for f in os.listdir(ROOT) if f.endswith(".md")):
        p0 = os.path.join(ROOT, top)
        files = [p0] if os.path.isfile(p0) else [os.path.join(r, f) for r, _, fs in os.walk(p0) for f in fs]
        for p in files:
            rel = os.path.relpath(p, ROOT)
            if any(e in rel for e in EXCLUDE) or rel in ("PROJECT_STATUS.md", "EXPERIMENT_LOG.md"):   # the two logs get appended entries
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            out[rel.replace("\\", "/")] = h.hexdigest()
    return out


# ------------------------------------------------------------------ benchmark
def bench(out):
    torch.set_num_threads(bl.THREADS)
    bl.configure("all")
    d = bl.load_data("data/processed_v1", {"train": 100, "cal": 10, "test": 0}, splits=("train",))["train"]
    T = bl.to_t(d)
    idx = torch.arange(256)
    x, s, _ = bl.batch(T, idx)
    H = list(bl.HEADS)
    specs = {"m1": lambda: md.MODELS["m1"](n_out=28, heads=H, dropout=0.3),
             "kanode_S1": lambda: mp.KANODENet(H, n_out=28, flow=True, substeps=1, dropout=0.3),
             "kanode_S2": lambda: mp.KANODENet(H, n_out=28, flow=True, substeps=2, dropout=0.3),
             "kanode_noflow": lambda: mp.KANODENet(H, n_out=28, flow=False, dropout=0.3),
             "gru": lambda: md.GRUNet(n_out=16), "grud": lambda: mp.GRUDNet(n_out=16)}
    res = {}
    for k, f in specs.items():
        torch.manual_seed(0)
        m = f()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        ts = []
        for r in range(6):
            t0 = time.perf_counter()
            loss = m(x, s).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ts.append(time.perf_counter() - t0)
        res[k] = {"sec_per_batch_median": float(np.median(ts[1:])), "n_params": md.n_params(m)}
    m1_runs = json.load(open("results/stage4_m1/run_log.json"))["runs"]
    m1_wall_max, m1_wall_mean = max(r["wall_s"] for r in m1_runs), float(np.mean([r["wall_s"] for r in m1_runs]))
    for k in res:
        ratio = res[k]["sec_per_batch_median"] / res["m1"]["sec_per_batch_median"]
        res[k]["ratio_vs_m1"] = ratio
        res[k]["projected_wall_min_per_seed_if_m1_epochs_meanM1"] = ratio * m1_wall_mean / 60
        res[k]["projected_wall_min_per_seed_if_m1_epochs_maxM1"] = ratio * m1_wall_max / 60
    gate = CFG["gates"]["G2_max_wall_min_per_seed"]
    proj = res["kanode_S1"]["projected_wall_min_per_seed_if_m1_epochs_maxM1"]
    summary = {"SYNTHETIC": True, "threads": bl.THREADS, "batch": 256, "results": res, "m1_saved_wall_s_mean": m1_wall_mean,
               "m1_saved_wall_s_max": m1_wall_max, "G2_projection_kanode_S1_min": proj, "G2_limit_min": gate, "G2_projection_pass": proj <= gate,
               "note": "projection = (sec/batch ratio vs M1) x saved M1 wall time per seed; assumes a similar number of epochs; the measured wall time of the real run is the binding G2 value"}
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ training (seed 0, calibration only)
@torch.no_grad()
def diagnostics(model, Tc, n=512):
    """kappa, latent range/extrapolation over a fixed calibration batch, and RK4 solver error S=1 vs S=4 (eval mode: no dropout, no RNG use)."""
    if not isinstance(model, mp.KANODENet):
        return {}
    was = model.training
    model.eval()
    saved = copy.deepcopy(model.stats)
    x, s, _ = bl.batch(Tc, torch.arange(n))
    model.reset_stats()
    o1, z1, _ = model(x, s, return_latent=True)
    st = copy.deepcopy(model.stats["eval"])
    out = {"fixed_batch_max_abs_z": st["max_abs_z"], "fixed_batch_frac_out_f": st["out_f"] / max(st["n_f"], 1),
           "fixed_batch_frac_out_g": st["out_g"] / max(st["n_g"], 1), "nonfinite_outputs": int((~torch.isfinite(o1)).sum())}
    if model.flow:
        S0 = model.S
        model.S = 4
        o4, z4, _ = model(x, s, return_latent=True)
        model.S = S0
        k = model.kappa()
        out.update({"kappa": [round(float(v), 5) for v in k], "kappa_min": float(k.min()), "kappa_max": float(k.max()),
                    "kappa_h_max": float(k.max()) / S0, "solver_S1_vs_S4_max_abs_dz": float((z1 - z4).abs().max()),
                    "solver_S1_vs_S4_max_abs_dout": float((o1 - o4).abs().max())})
    model.stats = saved
    model.train(was)
    return out


def cal_metrics(P, d):
    rows = []
    for j, task in enumerate(bl.HEADS):
        y = d["Y"][:, j]
        ok = ~np.isnan(y)
        m = mt.reg_metrics(y[ok], P[ok, j]) if task in bl.REG else mt.clf_metrics(y[ok], P[ok, j])
        rows += [dict(task=task, split="cal", metric=k, value=v) for k, v in m.items()]
    return rows


def train(name, taskset, out, quick=False, seed=None):
    seed = CFG["seed"] if seed is None else seed          # default 0 = pilot behaviour
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite.")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(bl.THREADS)
    mp.register(CFG["kanode"]["substeps"])
    bl.configure(taskset)
    proto = CFG["protocol_kan"] if name.startswith("kanode") else CFG["protocol_gru"]
    data = bl.load_data("data/processed_v1", {"train": 40, "cal": 20, "test": 0} if quick else None, splits=("train", "cal"))
    assert "test" not in data
    Tc = bl.to_t(data["cal"])
    logf = open(os.path.join(out, "train_log.txt"), "w", encoding="utf-8")
    epoch_diag = []

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()
        if " ep" in msg and "model" in mp.LAST:
            m = mp.LAST["model"]
            row = {"epoch": int(msg.split(" ep")[1].split()[0])}
            if isinstance(m, mp.KANODENet):
                row.update({f"{mode}_{k}": v for mode in ("train", "eval") for k, v in m.stats[mode].items()})
                m.reset_stats()
                row.update(diagnostics(m, Tc))
            epoch_diag.append(row)
            logf.write("  diag " + json.dumps(row) + "\n")

    mk = {"dropout": proto["dropout"], "heads": list(bl.HEADS)} if name.startswith("kanode") else None
    t0, c0 = time.perf_counter(), time.process_time()
    model, ym, ys, info = bl.fit_nn(name, seed, data["train"], data["cal"], epochs=2 if quick else proto["epochs"],
                                    patience=proto["patience"], bs=proto["batch_size"], lr=proto["lr"], wd=proto["weight_decay"],
                                    epoch_frac=proto["epoch_frac"], model_kwargs=mk, log=log)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    if isinstance(model, mp.KANODENet):
        model.reset_stats()
    P = bl.predict_nn(model, Tc, ym, ys)
    final = {}
    if isinstance(model, mp.KANODENet):
        st = model.stats["eval"]
        final = {"full_cal_max_abs_z": st["max_abs_z"], "full_cal_frac_out_f": st["out_f"] / max(st["n_f"], 1),
                 "full_cal_frac_out_g": st["out_g"] / max(st["n_g"], 1), "full_cal_nonfinite_latent": st["nonfinite"],
                 **diagnostics(model, Tc)}
    N = len(data["cal"]["pid"])
    full = np.full((N, 120, len(bl.HEADS)), np.nan, np.float32)
    full[data["cal"]["pi"], data["cal"]["ti"]] = P
    os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "predictions", f"{name}_seed{seed}_cal.npz"), pred=full, pid=data["cal"]["pid"], heads=np.array(bl.HEADS))
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "ymean": ym, "ystd": ys, "heads": bl.HEADS, "seed": seed, "best_epoch": info["best_epoch"]},
               os.path.join(out, "checkpoints", f"{name}_seed{seed}.pt"))
    pd.DataFrame(cal_metrics(P, data["cal"])).assign(model=name, seed=seed).to_csv(os.path.join(out, "metrics_cal.csv"), index=False)
    json.dump({"SYNTHETIC": True, "test_split_loaded": False, "model": name, "taskset": taskset, "heads": bl.HEADS, "seed": seed,
               "wall_s": wall, "cpu_s": cpu, "n_params": md.n_params(model), "torch_threads": bl.THREADS, "quick": quick,
               "best_cal_loss": min(h["cal_loss"] for h in info["history"]), "final_diagnostics": final, "epoch_diagnostics": epoch_diag,
               **info, "torch": torch.__version__}, open(os.path.join(out, "run_log.json"), "w"), indent=1, default=str)
    logf.close()
    print(f"done {name} wall {wall:.0f}s -> {out}")


# ------------------------------------------------------------------ analysis (calibration only)
def gap_age(d):
    """Days since the last observed (missing_day = 0) day at each origin, causal, full record."""
    miss = d["X"][..., mp.MD]
    age = np.zeros_like(miss)
    a = np.full(miss.shape[0], 99.0)
    for t in range(miss.shape[1]):
        a = np.where(miss[:, t] == 0, 0.0, a + 1)
        age[:, t] = a
    return age


def analyze(pilot):
    rows, loss = [], []
    ref = {"main": ("results/stage3", "gru"), "forecast": ("results/stage3c_forecast", "gru"), "all": ("results/stage4_m1", "m1")}
    runs = {"kanode": "kanode_all", "kanode_noflow": "kanode_noflow_all", "grud_main": "grud_main", "grud_forecast": "grud_forecast"}
    for key, sub in runs.items():
        rl = json.load(open(os.path.join(pilot, sub, "run_log.json")))
        rdir, rname = ref[rl["taskset"]]
        refl = [min(h["cal_loss"] for h in r["history"]) for r in json.load(open(os.path.join(rdir, "run_log.json")))["runs"] if r["model"] == rname]
        mu, sd = float(np.mean(refl)), float(np.std(refl, ddof=1))
        loss.append(dict(model=key, taskset=rl["taskset"], reference=f"{rname} (5 seeds)", best_cal_loss=rl["best_cal_loss"], ref_mean=mu, ref_sd=sd,
                         ref_min=min(refl), ref_max=max(refl), z_vs_ref=(rl["best_cal_loss"] - mu) / sd, pct_vs_ref_mean=100 * (rl["best_cal_loss"] - mu) / mu,
                         best_epoch=rl["best_epoch"], epochs_run=rl["epochs_run"], wall_min=rl["wall_s"] / 60, n_params=rl["n_params"]))
        mine = pd.read_csv(os.path.join(pilot, sub, "metrics_cal.csv"))
        refm = pd.read_csv(os.path.join(rdir, "metrics_long.csv"))
        refm = refm[(refm.model == rname) & (refm.split == "cal")]
        for (task, met), g in refm.groupby(["task", "metric"]):
            if met not in ("rmse", "ap", "auroc") or task not in set(mine.task):
                continue
            v = mine[(mine.task == task) & (mine.metric == met)].value.iloc[0]
            rows.append(dict(model=key, reference=rname, task=task, metric=met, pilot_seed0=v, ref_mean=g.value.mean(), ref_sd=g.value.std(ddof=1),
                             ref_seed0=g[g.seed == 0].value.iloc[0]))
    L, T = pd.DataFrame(loss), pd.DataFrame(rows)
    os.makedirs(os.path.join(pilot, "analysis"), exist_ok=True)
    L.to_csv(os.path.join(pilot, "analysis", "cal_loss_vs_reference.csv"), index=False)
    T.to_csv(os.path.join(pilot, "analysis", "cal_metrics_vs_reference.csv"), index=False)

    def fam(t):
        if t.startswith("sev"):
            return "severity (t0+forecast)" if "_t0" not in t else "severity nowcast"
        if t.startswith("status"):
            return "status nowcast" if "_t0" in t else "status forecast"
        return "sustained onset" if "sust" in t else "first-crossing onset"
    T["family"] = T.task.map(fam)
    F_ = T.groupby(["model", "reference", "family", "metric"])[["pilot_seed0", "ref_mean", "ref_sd", "ref_seed0"]].mean().reset_index()
    F_.to_csv(os.path.join(pilot, "analysis", "cal_family_means.csv"), index=False)

    # gap-stratified (days since last observed day at the origin), nowcast heads, calibration only
    import data as dd
    cal = dd.load_split("cal")
    age = gap_age(cal)
    strata = {"0": lambda a: a == 0, "1-2": lambda a: (a >= 1) & (a <= 2), ">=3": lambda a: a >= 3}
    G = []
    srcs = {"kanode": [os.path.join(pilot, "kanode_all", "predictions", "kanode_seed0_cal.npz")],
            "kanode_noflow": [os.path.join(pilot, "kanode_noflow_all", "predictions", "kanode_noflow_seed0_cal.npz")],
            "m1": [f"results/stage4_m1/predictions/m1_seed{s}_cal.npz" for s in range(5)],
            "grud": [os.path.join(pilot, "grud_main", "predictions", "grud_seed0_cal.npz")],
            "gru": [f"results/stage3/predictions/gru_seed{s}_cal.npz" for s in range(5)]}
    for model, files in srcs.items():
        for si, f in enumerate(files):
            z = np.load(f)
            heads = list(z["heads"])
            for task in ("sev_dep_t0", "sev_anx_t0", "status_dep_t0", "status_anx_t0"):
                y, p = cal["y_" + task], z["pred"][..., heads.index(task)]
                for sname, fn in strata.items():
                    ok = ~np.isnan(y) & fn(age)
                    m = mt.reg_metrics(y[ok], p[ok])["rmse"] if task.startswith("sev") else mt.clf_metrics(y[ok], p[ok])["ap"]
                    G.append(dict(model=model, seed=si, task=task, stratum=sname, n=int(ok.sum()), metric="rmse" if task.startswith("sev") else "ap", value=m))
    G = pd.DataFrame(G)
    G.to_csv(os.path.join(pilot, "analysis", "cal_gap_strata_long.csv"), index=False)
    Gs = G.groupby(["model", "task", "stratum", "metric"]).agg(n=("n", "first"), mean=("value", "mean"), sd=("value", "std"), n_seeds=("seed", "nunique")).reset_index()
    Gs.to_csv(os.path.join(pilot, "analysis", "cal_gap_strata_summary.csv"), index=False)
    print(L.to_string())
    print(F_.to_string())
    print(Gs.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "verify", "bench", "train", "analyze"])
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--model")
    ap.add_argument("--taskset", default="all")
    ap.add_argument("--pilot", default="results/p2_kanode/pilot")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.cmd == "manifest":
        if os.path.exists(a.out):
            raise SystemExit("manifest exists; refusing to overwrite")
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        h = file_hashes()
        json.dump(h, open(a.out, "w"), indent=0)
        print(len(h), "files hashed")
    elif a.cmd == "verify":
        pre, now = json.load(open(a.manifest)), file_hashes()
        changed = [k for k in pre if now.get(k) != pre[k]]
        print(f"{len(pre)} pre-existing files; changed or missing: {len(changed)}", changed[:20])
        sys.exit(1 if changed else 0)
    elif a.cmd == "bench":
        bench(a.out)
    elif a.cmd == "train":
        train(a.model, a.taskset, a.out, a.quick, a.seed)
    else:
        analyze(a.pilot)


if __name__ == "__main__":
    main()
