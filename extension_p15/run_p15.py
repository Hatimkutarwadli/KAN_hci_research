"""P1-5 compute step (SYNTHETIC; EXTENSION): extended early-warning evaluation from SAVED predictions (no retraining, no inference). Writes results/p1_early_warning/{sust,cross}/ ...
 sust  = PRIMARY analysis (sustained onset, horizons h7/h14/h21 with W = h, Stage 7 protocol; h21 reproduces Stage 7 exactly - see tests/test_p15.py)
 cross = SECONDARY analysis (first-crossing onset), kept in a separate directory; only for models that have first-crossing heads.
Per (model, seed): Stage 7 metrics, supplementary exposure-matched metrics, participant-trajectory shuffle null (200 shared permutations), iid random null (Stage 7 draws).
Then: stratified participant bootstrap (B = 1000, shared indices -> paired contrasts vs GRU, seed-pooled) and threshold-uncertainty bootstrap (calibration never-onset resampling; seed 0; sustained only).
Thresholds come from calibration only; no decision here uses test outcomes. Refuses to overwrite a non-empty output directory.
Usage: python extension_p15/run_p15.py"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ew15 as ew  # noqa: E402

OUT = "results/p1_early_warning"
B_BOOT, BOOT_SEED, B_THR, THR_SEED, N_SHUFFLE, NULL_DRAWS = 1000, 20261116, 300, 20261117, 200, 5
KINDS = ("sust", "cross")


def main():
    os.chdir(ew.ROOT)
    if os.path.isdir(OUT) and os.listdir(OUT):
        raise SystemExit(f"{OUT} exists and is not empty; refusing to overwrite.")
    for kind in KINDS:
        os.makedirs(f"{OUT}/{kind}", exist_ok=True)
    perms = ew.shuffle_perms(N_SHUFFLE)
    cohorts = {(kind, k, h): (ew.load_cohort("cal", k, kind, h), ew.load_cohort("test", k, kind, h)) for kind in KINDS for k in ew.KS for h in ew.HORIZONS}
    rows, nullrows, parr, thr_rows = [], [], {}, []
    t_start = time.time()
    runs = [(m, s) for m, v in ew.SRC.items() for s in v[2]] + [("random_null_iid", s) for s in range(NULL_DRAWS)]
    for mi, (model, seed) in enumerate(runs):
        for kind in KINDS:
            if model != "random_null_iid" and not ew.has_head(model, kind):
                continue
            for k in ew.KS:
                for h in ew.HORIZONS:
                    cal, test = cohorts[(kind, k, h)]
                    if model == "random_null_iid":
                        Scal, Stest = ew.null_scores_iid(seed)
                    else:
                        Scal, Stest = (ew.load_scores(model, seed, sp, k, kind, h, c["pid"]) for sp, c in (("cal", cal), ("test", test)))
                        assert np.isfinite(Stest[:, ew.T0:test["T1"] + 1]).all() and np.isfinite(Scal[:, ew.T0:cal["T1"] + 1]).all()
                    auc = ew.participant_auc(Stest, test)
                    for f in ew.TARGETS:
                        res, st = ew.evaluate_one(Scal, Stest, cal, test, f)
                        res.update(kind=kind, outcome=k, h=h, model=model, seed=seed, target_fap=f, auc_participant=auc)
                        rows.append(res)
                        parr.setdefault((kind, k, h, f, model), []).append(ew.participant_arrays(st, test))
                        if model != "random_null_iid":
                            nm = ew.shuffle_null(Stest, res["tau"], test, perms)
                            for met, v in nm.items():
                                obs = res[met]
                                nullrows.append(dict(kind=kind, outcome=k, h=h, model=model, seed=seed, target_fap=f, metric=met, observed=obs, null_mean=float(np.nanmean(v)), null_lo=float(np.nanpercentile(v, 2.5)),
                                                     null_hi=float(np.nanpercentile(v, 97.5)), frac_null_ge_observed=float(np.nanmean(v >= obs - 1e-12)), n_shuffles=len(v)))
                            if kind == "sust" and seed == 0:
                                for r in ew.threshold_bootstrap(Scal, Stest, cal, test, f, B_THR, THR_SEED):
                                    thr_rows.append(dict(outcome=k, h=h, model=model, seed=0, target_fap=f, **r))
        print(f"[{mi + 1}/{len(runs)}] {model} seed{seed}  {time.time() - t_start:.0f}s", flush=True)
    per = pd.DataFrame(rows)
    nulls = pd.DataFrame(nullrows)
    thr = pd.DataFrame(thr_rows)
    for kind in KINDS:
        per[per.kind == kind].to_csv(f"{OUT}/{kind}/per_seed_metrics.csv", index=False)
        nulls[nulls.kind == kind].to_csv(f"{OUT}/{kind}/shuffle_null.csv", index=False)
    thr.to_csv(f"{OUT}/sust/threshold_bootstrap_draws.csv", index=False)

    # ---- participant bootstrap (stratified, shared indices => paired), seed-pooled, thresholds fixed
    brow = []
    pm_cols = None
    for kind in KINDS:
        for k in ew.KS:
            for h in ew.HORIZONS:
                cal, test = cohorts[(kind, k, h)]
                on, nv, bon, bnv = ew.stratified_indices(test, B_BOOT, BOOT_SEED)
                for f in ew.TARGETS:
                    boots, points = {}, {}
                    for (kk, k2, h2, f2, model), arrs in parr.items():
                        if (kk, k2, h2, f2) != (kind, k, h, f):
                            continue
                        boots[model] = ew.boot_metrics(arrs, bon, bnv, test)
                        sub = per[(per.kind == kind) & (per.outcome == k) & (per.h == h) & (per.target_fap == f) & (per.model == model)]
                        points[model] = sub.mean(numeric_only=True)
                    for model, bm in boots.items():
                        for met, v in bm.items():
                            row = dict(kind=kind, outcome=k, h=h, target_fap=f, model=model, metric=met, point_seed_mean=points[model].get(met, np.nan), ci95_lo=float(np.nanpercentile(v, 2.5)),
                                       ci95_hi=float(np.nanpercentile(v, 97.5)), n_boot=len(v), n_onset=len(on), n_never=len(nv))
                            if model != "gru" and "gru" in boots:
                                d = v - boots["gru"][met]
                                row.update(diff_vs_gru_point=float(points[model].get(met, np.nan) - points["gru"].get(met, np.nan)), diff_vs_gru_lo=float(np.nanpercentile(d, 2.5)),
                                           diff_vs_gru_hi=float(np.nanpercentile(d, 97.5)), frac_resamples_better_than_gru=float(np.nanmean(d > 0)))
                            brow.append(row)
    bt = pd.DataFrame(brow)
    for kind in KINDS:
        bt[bt.kind == kind].to_csv(f"{OUT}/{kind}/bootstrap_ci.csv", index=False)
    json.dump({"SYNTHETIC": True, "EXTENSION": "P1-5 extended early-warning evaluation", "horizons": list(ew.HORIZONS), "W": "= h", "T1": "sust 117-h, cross 119-h", "targets": list(ew.TARGETS), "leads": list(ew.LEADS),
               "episode_gap": ew.EP_GAP, "B_boot": B_BOOT, "boot_seed": BOOT_SEED, "B_threshold_boot": B_THR, "threshold_boot_seed": THR_SEED, "n_shuffles": N_SHUFFLE, "shuffle_perm_seed": 15015,
               "iid_null_draws": NULL_DRAWS, "models": {m: [v[0], list(v[2]), v[3]] for m, v in ew.SRC.items()},
               "primary_contrast": "sust, h21, target FAP 0.20, det_lead7 (overall), per outcome, paired stratified participant bootstrap vs GRU (seed-pooled)",
               "matched_exposure": "complete non-overlapping W-day blocks of the never-onset window; leftover days at the end are not used"},
              open(f"{OUT}/config.json", "w"), indent=1)
    print("done", f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
