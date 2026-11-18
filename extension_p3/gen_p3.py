"""P3: generate + validate R4/R5 with the UNCHANGED P2-2 pipeline (gen_p22.generate / gen_p22.validate).
Validation output is redirected to results/p3_small_data/datasets/<ds>/validation.json (train/cal + simulator-level only; test never read).
  python extension_p3/gen_p3.py R4 R5"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extension_p22"))
import common_p22 as C  # noqa: E402
import gen_p22 as G  # noqa: E402

if __name__ == "__main__":
    os.chdir(ROOT)
    for ds in sys.argv[1:]:
        assert ds in ("R4", "R5")
        G.generate(ds)
        C.RES = "results/p3_small_data/datasets"          # redirect validation output only (module attribute; no file edited)
        G.validate(ds)
        C.RES = "results/p2_datasets"
