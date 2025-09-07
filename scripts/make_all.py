#!/usr/bin/env python
# ensure project root on sys.path
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from codes.pn_solver import compute_shape_coeffs
from codes.gauge_invariants import delta_K_shape, delta_z1_shape

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def sweep_plummer():
    rows=[]
    for eps in [0.10, 0.05, 0.02]:
        rows.append(compute_shape_coeffs(N=96, L=3.0, a=1.0, eps=eps))
    df = pd.DataFrame(rows).sort_values("eps", ascending=False)
    df.to_csv("data/C3C4_plummer_sweep.csv", index=False)
    plt.figure()
    plt.plot(df["eps"], df["C3"], 'o-', label="C3")
    plt.plot(df["eps"], df["C4"], 'o-', label="C4")
    plt.gca().invert_xaxis()
    plt.xlabel("epsilon / a"); plt.ylabel("coeff")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/C3C4_plummer_sweep.png", dpi=200)

def make_gauge_invariant_shapes(C3=14.08, C4=32.79):
    x = np.logspace(-8, -1, 200)
    df = pd.DataFrame({
        "x": x,
        "deltaK_shape": delta_K_shape(x, C3, C4),
        "deltaz1_shape": delta_z1_shape(x, C3, C4),
    })
    df.to_csv("data/gauge_invariant_shapes.csv", index=False)
    for col, fname, ylabel in [("deltaK_shape","dK_vs_x.png","δK(x) per (α/M²)"),
                               ("deltaz1_shape","dz1_vs_x.png","δz₁(x) per (α/M²)")]:
        plt.figure(); plt.loglog(df["x"], df[col])
        plt.xlabel("x=(GMΩ/c³)^{2/3}"); plt.ylabel(ylabel)
        plt.tight_layout(); plt.savefig(f"figures/{fname}", dpi=200)

if __name__ == "__main__":
    sweep_plummer()
    make_gauge_invariant_shapes()
    print("OK: data & figures written.")

import subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
py = sys.executable

# Renormalized fits (reads data/C3C4_finitesphere_sweep_Rstar.csv)
subprocess.check_call([py, str(root / "scripts" / "make_renormalized_fits.py")])

