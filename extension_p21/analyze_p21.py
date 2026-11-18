"""P2-1 five-seed analysis and the single test evaluation (SYNTHETIC data). Plan: results/p2_kanode/ANALYSIS_PLAN.md.
  python extension_p21/analyze_p21.py lock                 -> hash plan + this file into results/p2_kanode/plan_lock.json (refuses to overwrite)
  python extension_p21/analyze_p21.py analyze --split cal  -> calibration analysis
  python extension_p21/analyze_p21.py test                 -> ONE-TIME test prediction (refuses if done or plan changed), then analysis on test"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "extension_p21")]
import baselines as bl  # noqa: E402
import data as dd  # noqa: E402
import metrics as mt  # noqa: E402
import models_p21 as mp  # noqa: E402

BASE = "results/p2_kanode"
SEEDS = (0, 1, 2, 3, 4)
LOCK = os.path.join(BASE, "plan_lock.json")
LOCKED = (os.path.join(BASE, "ANALYSIS_PLAN.md"), "extension_p21/analyze_p21.py")
NEW = {"kanode": ("kanode_all", "kanode"), "kanode_noflow": ("kanode_noflow_all", "kanode_noflow"),
       "grud_main": ("grud_main", "grud"), "grud_forecast": ("grud_forecast", "grud")}
REF = {"m1": ("results/stage4_m1", "m1"), "gru_main": ("results/stage3", "gru"), "gru_forecast": ("results/stage3c_forecast", "gru")}
KS = ("dep", "anx")
FAMS = {"sev_now": ("rmse", ["sev_{k}_t0"]), "sev_fc": ("rmse", ["sev_{k}_h7", "sev_{k}_h14", "sev_{k}_h21"]),
        "status_now": ("ap", ["status_{k}_t0"]), "status_fc": ("ap", ["status_{k}_h7", "status_{k}_h14", "status_{k}_h21"]),
        "onset_sust": ("ap", ["onset_sust_{k}_h7", "onset_sust_{k}_h14", "onset_sust_{k}_h21"]),
        "onset_cross": ("ap", ["onset_cross_{k}_h7", "onset_cross_{k}_h14", "onset_cross_{k}_h21"])}
STRATA = {"0": lambda a: a == 0, "1-2": lambda a: (a >= 1) & (a <= 2), ">=3": lambda a: a >= 3}
PAIRS = [("kanode", "kanode_noflow"), ("kanode", "m1"), ("kanode_noflow", "m1"), ("grud", "gru")]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_dir(key, seed):
    sub = NEW[key][0]
    return os.path.join(BASE, "pilot", sub) if seed == 0 else os.path.join(BASE, "full", f"{sub}_seed{seed}")


def pred_file(key, seed, split):
    if key in NEW:
        name = NEW[key][1]
        if split == "cal":
            return os.path.join(run_dir(key, seed), "predictions", f"{name}_seed{seed}_cal.npz")
        return os.path.join(BASE, "test", "predictions", f"{key}_seed{seed}_test.npz")
    d, name = REF[key]
    return os.path.join(d, "predictions", f"{name}_seed{seed}_{split}.npz")


def gap_age(miss):
    age, a = np.zeros_like(miss), np.full(miss.shape[0], 99.0)
    for t in range(miss.shape[1]):
        a = np.where(miss[:, t] == 0, 0.0, a + 1)
        age[:, t] = a
    return age


def metric(name, y, p):
    return mt.reg_metrics(y, p)["rmse"] if name == "rmse" else mt.clf_metrics(y, p)[name]


def source(model, fam):
    """Which saved model provides a family: 28-head models provide all; GRU/GRU-D forecast families come from the forecast models."""
    if model in ("kanode", "kanode_noflow", "m1"):
        return model
    return f"{model}_forecast" if fam in ("sev_fc", "status_fc") else f"{model}_main"


def collect(split):
    lab = dd.load_split(split)
    age = gap_age(lab["X"][..., mp.MD])
    fam_rows, strat_rows = [], []
    cache = {}
    for model in ("kanode", "kanode_noflow", "m1", "grud", "gru"):
        for seed in SEEDS:
            for fam, (met, tpl) in FAMS.items():
                src = source(model, fam)
                key = (src, seed)
                if key not in cache:
                    z = np.load(pred_file(src, seed, split))
                    cache[key] = (z["pred"], list(z["heads"]))
                pred, heads = cache[key]
                mets = [met] + (["auroc"] if met == "ap" else [])
                for m_ in mets:
                    vals = {}
                    for k in KS:
                        v = []
                        for t in tpl:
                            h = t.format(k=k)
                            y = lab["y_" + h]
                            ok = ~np.isnan(y)
                            v.append(metric(m_, y[ok], pred[..., heads.index(h)][ok]))
                        vals[k] = float(np.mean(v))
                    vals["both"] = float(np.mean([vals["dep"], vals["anx"]]))
                    for k, v in vals.items():
                        fam_rows.append(dict(split=split, model=model, seed=seed, family=fam, disease=k, metric=m_, value=v))
                if fam in ("sev_now", "status_now"):
                    for k in KS:
                        h = tpl[0].format(k=k)
                        y = lab["y_" + h]
                        for sname, fn in STRATA.items():
                            ok = ~np.isnan(y) & fn(age)
                            strat_rows.append(dict(split=split, model=model, seed=seed, family=fam, disease=k, stratum=sname, metric=met, n=int(ok.sum()),
                                                   value=metric(met, y[ok], pred[..., heads.index(h)][ok])))
    return pd.DataFrame(fam_rows), pd.DataFrame(strat_rows)


def summarise(df, keys):
    g = df.groupby(keys + ["model"]).value
    s = g.agg(mean="mean", sd=lambda v: v.std(ddof=1), min="min", max="max", n="count").reset_index()
    per = df.pivot_table(index=keys + ["model"], columns="seed", values="value").add_prefix("seed").reset_index()
    return s.merge(per, on=keys + ["model"])


def paired(df, keys):
    rows = []
    for a, b in PAIRS:
        A = df[df.model == a].set_index(keys + ["seed"]).value
        B = df[df.model == b].set_index(keys + ["seed"]).value
        d = (A - B).dropna().reset_index()
        for idx, g in d.groupby(keys):
            met = idx[keys.index("metric")]
            v = g.sort_values("seed").value.values
            better = int((v < 0).sum()) if met == "rmse" else int((v > 0).sum())
            sd = float(np.std(v, ddof=1))
            same = better in (0, len(v))
            verdict = "consistent" if same and abs(v.mean()) > sd else ("leaning" if better in (1, len(v) - 1) or same else "not distinguishable")
            rows.append(dict(zip(keys, idx), A=a, B=b, mean_diff=float(v.mean()), sd_diff=sd, min_diff=float(v.min()), max_diff=float(v.max()),
                             A_better_n=better, n=len(v), verdict=verdict, **{f"d_seed{s}": float(x) for s, x in zip(g.sort_values("seed").seed, v)}))
    return pd.DataFrame(rows)


def stability():
    rows = []
    for key in ("kanode", "kanode_noflow", "grud_main", "grud_forecast"):
        for seed in SEEDS:
            rl = json.load(open(os.path.join(run_dir(key, seed), "run_log.json")))
            r = dict(model=key, seed=seed, best_epoch=rl["best_epoch"], epochs_run=rl["epochs_run"], wall_min=rl["wall_s"] / 60, cpu_s=rl["cpu_s"],
                     best_cal_loss=rl["best_cal_loss"], n_params=rl["n_params"])
            E, F_ = rl["epoch_diagnostics"], rl["final_diagnostics"]
            if key.startswith("kanode"):
                r.update(train_max_abs_z=max(e["train_max_abs_z"] for e in E), cal_max_abs_z_final=F_["full_cal_max_abs_z"],
                         max_frac_out_f=max(e["train_out_f"] / max(e["train_n_f"], 1) for e in E), max_frac_out_g=max(e["train_out_g"] / max(e["train_n_g"], 1) for e in E),
                         final_frac_out=F_["full_cal_frac_out_f"] + F_["full_cal_frac_out_g"],
                         nonfinite=sum(e["train_nonfinite"] + e["eval_nonfinite"] for e in E) + F_["full_cal_nonfinite_latent"] + F_["nonfinite_outputs"])
                if key == "kanode":
                    r.update(kappa_min_final=F_["kappa_min"], kappa_max_final=F_["kappa_max"], kappa_max_any_epoch=max(e["kappa_max"] for e in E),
                             kappa_h_max_final=F_["kappa_h_max"], solver_dz_final=F_["solver_S1_vs_S4_max_abs_dz"], solver_dout_final=F_["solver_S1_vs_S4_max_abs_dout"],
                             solver_dz_max_any_epoch=max(e["solver_S1_vs_S4_max_abs_dz"] for e in E))
            rows.append(r)
    return pd.DataFrame(rows)


def loss_table():
    rows = []
    for key, (ref, tset) in {"kanode": ("m1", "all"), "kanode_noflow": ("m1", "all"), "grud_main": ("gru_main", "main"), "grud_forecast": ("gru_forecast", "forecast")}.items():
        refl = {r["seed"]: min(h["cal_loss"] for h in r["history"]) for r in json.load(open(os.path.join(REF[ref][0], "run_log.json")))["runs"] if r["model"] == REF[ref][1]}
        for seed in SEEDS:
            v = json.load(open(os.path.join(run_dir(key, seed), "run_log.json")))["best_cal_loss"]
            rows.append(dict(model=key, reference=ref, seed=seed, best_cal_loss=v, ref_best_cal_loss=refl[seed], diff=v - refl[seed]))
    return pd.DataFrame(rows)


def write_md(out, F, S, P, PS, title):
    def tbl(d, cols):
        d = d[cols]
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
            "| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.itertuples(index=False))
    sd = [c for c in F.columns if c.startswith("seed")]
    parts = [f"# {title} (SYNTHETIC)\n", "## Family means over seeds\n", tbl(F, ["family", "disease", "metric", "model", "mean", "sd", "min", "max"] + sd),
             "\n## Paired differences (A - B, seed-matched)\n",
             tbl(P, ["family", "disease", "metric", "A", "B", "mean_diff", "sd_diff", "min_diff", "max_diff", "A_better_n", "verdict"]),
             "\n## Missing-day strata (nowcast)\n", tbl(S, ["family", "disease", "stratum", "metric", "model", "mean", "sd", "min", "max"]),
             "\n## Missing-day strata: paired differences\n",
             tbl(PS, ["family", "disease", "stratum", "metric", "A", "B", "mean_diff", "sd_diff", "A_better_n", "verdict"])]
    open(out, "w", encoding="utf-8").write("\n".join(parts))


def analyze(split):
    out = os.path.join(BASE, "analysis_" + split)
    os.makedirs(out, exist_ok=True)
    fam, strat = collect(split)
    fam.to_csv(os.path.join(out, "family_per_seed.csv"), index=False)
    strat.to_csv(os.path.join(out, "strata_per_seed.csv"), index=False)
    fk, sk = ["family", "disease", "metric"], ["family", "disease", "stratum", "metric"]
    F, S = summarise(fam, fk), summarise(strat, sk)
    P, PS = paired(fam, fk), paired(strat, sk)
    for n, d in (("family_summary", F), ("strata_summary", S), ("paired", P), ("strata_paired", PS)):
        d.to_csv(os.path.join(out, n + ".csv"), index=False)
    if split == "cal":
        stability().to_csv(os.path.join(out, "stability_runtime.csv"), index=False)
        loss_table().to_csv(os.path.join(out, "cal_loss_per_seed.csv"), index=False)
    write_md(os.path.join(out, f"p21_{split}_tables.md"), F, S, P, PS, f"P2-1 five-seed results, {split} split")
    print("written", out)


def test_eval():
    lock = json.load(open(LOCK))
    for p in LOCKED:
        if sha(p) != lock[p]:
            raise SystemExit(f"{p} changed after locking; refusing test evaluation")
    tdir = os.path.join(BASE, "test")
    if os.path.exists(tdir):
        raise SystemExit("test evaluation already done; refusing to repeat")
    for key in NEW:
        for seed in SEEDS:
            assert os.path.exists(os.path.join(run_dir(key, seed), "run_log.json")), (key, seed)
    assert os.path.exists(os.path.join(BASE, "analysis_cal", "p21_cal_tables.md")), "calibration analysis must be written first"
    os.makedirs(os.path.join(tdir, "predictions"))
    torch.set_num_threads(bl.THREADS)
    log = []
    for key, (sub, name) in NEW.items():
        ts = {"kanode": "all", "kanode_noflow": "all", "grud_main": "main", "grud_forecast": "forecast"}[key]
        bl.configure(ts)
        data = bl.load_data("data/processed_v1", None, splits=("test",))
        T = bl.to_t(data["test"])
        for seed in SEEDS:
            ck = torch.load(os.path.join(run_dir(key, seed), "checkpoints", f"{name}_seed{seed}.pt"), weights_only=False)
            assert list(ck["heads"]) == bl.HEADS
            m = mp.KANODENet(list(bl.HEADS), n_out=28, flow=(key == "kanode"), substeps=1, dropout=0.3) if key.startswith("kanode") else mp.GRUDNet(n_out=len(bl.HEADS))
            m.load_state_dict(ck["state_dict"], strict=True)
            P = bl.predict_nn(m, T, ck["ymean"], ck["ystd"])
            N = len(data["test"]["pid"])
            full = np.full((N, 120, len(bl.HEADS)), np.nan, np.float32)
            full[data["test"]["pi"], data["test"]["ti"]] = P
            np.savez_compressed(os.path.join(tdir, "predictions", f"{key}_seed{seed}_test.npz"), pred=full, pid=data["test"]["pid"], heads=np.array(bl.HEADS))
            fin = {}
            if key.startswith("kanode"):
                st = m.stats["eval"]
                fin = dict(test_max_abs_z=st["max_abs_z"], test_frac_out=(st["out_f"] + st["out_g"]) / max(st["n_f"] + st["n_g"], 1), test_nonfinite=st["nonfinite"])
            log.append(dict(model=key, seed=seed, best_epoch=ck["best_epoch"], nonfinite_pred=int((~np.isfinite(P)).sum()), **fin))
            print("test predicted", key, seed, flush=True)
    pd.DataFrame(log).to_csv(os.path.join(tdir, "test_prediction_log.csv"), index=False)
    json.dump({"SYNTHETIC": True, "test_evaluated_once": True, "plan_lock": lock}, open(os.path.join(tdir, "test_eval_meta.json"), "w"), indent=1)
    analyze("test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "analyze", "test"])
    ap.add_argument("--split", default="cal", choices=["cal"])        # test analysis only via the one-time 'test' command
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.cmd == "lock":
        if os.path.exists(LOCK):
            raise SystemExit("plan already locked")
        json.dump({p: sha(p) for p in LOCKED}, open(LOCK, "w"), indent=1)
        print("locked", LOCK)
    elif a.cmd == "analyze":
        analyze(a.split)
    else:
        test_eval()


if __name__ == "__main__":
    main()
