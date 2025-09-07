#!/usr/bin/env python
import sys, pathlib, pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]
targets = [
  ("baseline", root/"data/C3C4_polytrope_n_sweep_baseline.csv"),
  ("box6", root/"data/C3C4_polytrope_n_sweep_box6.csv"),
  ("N128", root/"data/C3C4_polytrope_n_sweep_N128.csv"),
  ("ring075_box6_N128", root/"data/C3C4_polytrope_n_sweep_ring075_box6_N128.csv"),
]
dfs = []
for label, path in targets:
    df = pd.read_csv(path)
    # tolerate column names; ensure n,C3,C4
    ncol = "n" if "n" in df else df.columns[0]
    out = df[[ncol, "C3", "C4"]].copy()
    out.columns = ["n", f"C3_{label}", f"C4_{label}"]
    dfs.append(out)

comp = dfs[0]
for d in dfs[1:]:
    comp = comp.merge(d, on="n", how="inner")

# compute relative shifts vs baseline
for label in ["box6","N128","ring075_box6_N128"]:
    comp[f"dC3_{label}"] = (comp[f"C3_{label}"] - comp["C3_baseline"]) / comp["C3_baseline"]
    comp[f"dC4_{label}"] = (comp[f"C4_{label}"] - comp["C4_baseline"]) / comp["C4_baseline"]

out = root/"data"/"C3C4_polytrope_convergence_compare.csv"
comp.to_csv(out, index=False)
print(f"Wrote {out}")
print(comp.to_string(index=False, float_format=lambda v: f'{v:.3e}'))
