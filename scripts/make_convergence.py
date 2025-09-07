#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convergence studies for the uniform finite sphere source:
1) Ring-radius convergence at fixed box size
2) Box/domain-size convergence at fixed ring radius

Outputs:
  data/C3C4_ring_radius_convergence.csv
  figures/C3C4_ring_vs_Rring.png
  data/C3C4_domain_size_convergence.csv
  figures/C3C4_domain_convergence.png
"""

from __future__ import annotations
import sys, pathlib, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on sys.path (so `codes/` is importable when run from anywhere)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from codes.sources_finite_sphere import C3C4_for_uniform_sphere  # see helper in section 2 below

def _ensure_dirs():
    root = pathlib.Path(__file__).resolve().parents[1]
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root

def ring_radius_convergence(R_star: float,
                            box_radius_factor: float,
                            ring_radius_factors: list[float],
                            n_grid: int) -> pd.DataFrame:
    rows = []
    for f in ring_radius_factors:
        out = C3C4_for_uniform_sphere(R_star=R_star,
                                      ring_radius_factor=f,
                                      box_radius_factor=box_radius_factor,
                                      n_grid=n_grid)
        rows.append({
            "R_star": out["R_star"],
            "ring_radius": out["ring_radius"],
            "box_radius": out["box_radius"],
            "ring_radius_factor": f,
            "C3": out["C3"],
            "C4": out["C4"],
        })
    df = pd.DataFrame(rows).sort_values("ring_radius_factor").reset_index(drop=True)

    # Take the largest ring as proxy for asymptotic reference (more benign image-charge effects)
    ref = df.iloc[-1]
    for key in ["C3", "C4"]:
        df[f"d{key}"] = df[key] - ref[key]
        df[f"reld{key}"] = np.where(ref[key] != 0.0, df[f"d{key}"] / ref[key], np.nan)
    return df

def domain_size_convergence(R_star: float,
                            ring_radius_factor: float,
                            box_radius_factors: list[float],
                            n_grid: int) -> pd.DataFrame:
    rows = []
    for f in box_radius_factors:
        out = C3C4_for_uniform_sphere(R_star=R_star,
                                      ring_radius_factor=ring_radius_factor,
                                      box_radius_factor=f,
                                      n_grid=n_grid)
        rows.append({
            "R_star": out["R_star"],
            "ring_radius": out["ring_radius"],
            "box_radius": out["box_radius"],
            "box_radius_factor": f,
            "C3": out["C3"],
            "C4": out["C4"],
        })
    df = pd.DataFrame(rows).sort_values("box_radius_factor").reset_index(drop=True)

    # Use largest domain as reference (least boundary contamination)
    ref = df.iloc[-1]
    for key in ["C3", "C4"]:
        df[f"d{key}"] = df[key] - ref[key]
        df[f"reld{key}"] = np.where(ref[key] != 0.0, df[f"d{key}"] / ref[key], np.nan)
    return df

def plot_ring(df: pd.DataFrame, out_png: pathlib.Path):
    plt.figure(figsize=(6.0, 4.2), dpi=150)
    x = df["ring_radius_factor"].values
    plt.plot(x, df["C3"].values, marker="o", label="C3")
    plt.plot(x, df["C4"].values, marker="s", label="C4")
    plt.xlabel(r"Ring radius factor $R_{\rm ring}/R_\star$")
    plt.ylabel(r"$C_3,\; C_4$  (code units)")
    plt.title("Ring-radius convergence (box radius fixed)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_domain(df: pd.DataFrame, out_png: pathlib.Path):
    plt.figure(figsize=(6.0, 4.2), dpi=150)
    x = df["box_radius_factor"].values
    plt.plot(x, df["C3"].values, marker="o", label="C3")
    plt.plot(x, df["C4"].values, marker="s", label="C4")
    plt.xlabel(r"Box radius factor $R_{\rm box}/R_\star$")
    plt.ylabel(r"$C_3,\; C_4$  (code units)")
    plt.title("Domain-size convergence (ring radius fixed)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate convergence studies for finite-sphere source.")
    parser.add_argument("--Rstar", type=float, default=1.0, help="Physical sphere radius (code units).")
    parser.add_argument("--ngrid", type=int, default=32768, help="Grid size (power of two recommended).")
    parser.add_argument("--ring_factors", type=str,
                        default="0.30,0.40,0.50,0.60,0.75,1.00,1.25",
                        help="Comma-separated ring radius factors (×Rstar).")
    parser.add_argument("--box_factors", type=str,
                        default="3.0,4.0,5.0,6.0,8.0,10.0",
                        help="Comma-separated box radius factors (×Rstar).")
    parser.add_argument("--ring_box_factor", type=float, default=5.0,
                        help="Box factor used during ring-radius convergence.")
    parser.add_argument("--domain_ring_factor", type=float, default=0.6,
                        help="Ring factor used during domain-size convergence.")
    args = parser.parse_args()

    root = _ensure_dirs()
    ring_radius_factors = [float(s) for s in args.ring_factors.split(",") if s.strip()]
    box_radius_factors  = [float(s) for s in args.box_factors.split(",") if s.strip()]

    # --- 1) Ring-radius convergence ---
    print(f"[1/2] Ring-radius convergence: R_star={args.Rstar}, box_factor={args.ring_box_factor}, n_grid={args.ngrid}")
    df_ring = ring_radius_convergence(R_star=args.Rstar,
                                      box_radius_factor=args.ring_box_factor,
                                      ring_radius_factors=ring_radius_factors,
                                      n_grid=args.ngrid)
    csv1 = root / "data" / "C3C4_ring_radius_convergence.csv"
    df_ring.to_csv(csv1, index=False)
    print(f"  -> wrote {csv1}")

    png1 = root / "figures" / "C3C4_ring_vs_Rring.png"
    plot_ring(df_ring, png1)
    print(f"  -> wrote {png1}")

    # --- 2) Domain-size convergence ---
    print(f"[2/2] Domain-size convergence: R_star={args.Rstar}, ring_factor={args.domain_ring_factor}, n_grid={args.ngrid}")
    df_dom = domain_size_convergence(R_star=args.Rstar,
                                     ring_radius_factor=args.domain_ring_factor,
                                     box_radius_factors=box_radius_factors,
                                     n_grid=args.ngrid)
    csv2 = root / "data" / "C3C4_domain_size_convergence.csv"
    df_dom.to_csv(csv2, index=False)
    print(f"  -> wrote {csv2}")

    png2 = root / "figures" / "C3C4_domain_convergence.png"
    plot_domain(df_dom, png2)
    print(f"  -> wrote {png2}")

    print("OK: convergence CSVs and figures written.")

if __name__ == "__main__":
    main()
