#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit C3(R*/a), C4(R*/a) with quadratics and extract renormalized intercepts at R*/a -> 0.
Reads:   data/C3C4_finitesphere_sweep_Rstar.csv
Writes:  data/C3C4_renormalized_fits.csv
         figures/renormalized_fit_report.png
"""

from __future__ import annotations
import sys, pathlib, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def _ensure_dirs():
    root = pathlib.Path(__file__).resolve().parents[1]
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root

def _read_finite_sphere_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Be tolerant to column naming variations
    # Expected: Rstar_over_a, maybe 'R_over_a' or 'R/a'
    if "Rstar_over_a" in df:
        x = df["Rstar_over_a"].astype(float)
    elif "R_over_a" in df:
        x = df["R_over_a"].astype(float)
    elif "R/a" in df:
        x = df["R/a"].astype(float)
    else:
        raise ValueError("Could not find Rstar_over_a (or R_over_a / R/a) in CSV.")

    # Ring factor might vary; prefer a single track (e.g., Rring_over_a closest to 0.6 or 1.0)
    if "Rring_over_a" in df:
        df["Rring_over_a"] = df["Rring_over_a"].astype(float)
        # choose the densest/most common ring radius
        ring_val = df["Rring_over_a"].round(3).mode().iloc[0]
        sel = (df["Rring_over_a"].round(3) == ring_val)
        df = df.loc[sel].copy()
    # else: assume single ring used

    # Choose the columns for C3, C4 (prefer interaction-only if present)
    if "C3_int" in df and "C4_int" in df:
        y3 = df["C3_int"].astype(float)
        y4 = df["C4_int"].astype(float)
        y3_label = "C3_int"; y4_label = "C4_int"
    else:
        y3 = df["C3"].astype(float)
        y4 = df["C4"].astype(float)
        y3_label = "C3"; y4_label = "C4"

    # Keep only finite values and sort by x
    keep = np.isfinite(x) & np.isfinite(y3) & np.isfinite(y4)
    df = df.loc[keep].copy()
    df["x"] = x[keep]
    df[y3_label] = y3[keep]
    df[y4_label] = y4[keep]
    df.sort_values("x", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, y3_label, y4_label

def _polyfit_with_report(x, y, deg=2):
    # unweighted quadratic fit: y = c0 + c1 x + c2 x^2
    coeffs = np.polyfit(x, y, deg)
    p = np.poly1d(coeffs)
    yhat = p(x)

    # simple R^2
    ss_res = np.sum((y - yhat)**2.0)
    ss_tot = np.sum((y - np.mean(y))**2.0)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0.0 else np.nan)

    # crude intercept error via residual scatter + design matrix (for quick referee plot)
    # (Not a full GLS; good enough for viz. If needed we can switch to statsmodels.)
    X = np.vstack([np.ones_like(x), x, x**2]).T
    # covariance ~ sigma^2 (X^T X)^{-1}
    dof = max(len(x) - (deg + 1), 1)
    sigma2 = ss_res / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    var = np.diag(sigma2 * XtX_inv)
    # intercept is the first coefficient in 1D poly1d representation: c0 = coeffs[-1] if poly1d is descending
    # numpy.polyfit returns [c2, c1, c0]; intercept = coeffs[-1]
    c0 = coeffs[-1]
    c0_err = np.sqrt(var[-1]) if var[-1] >= 0.0 else np.nan

    return coeffs, r2, c0, c0_err, p

def main():
    ap = argparse.ArgumentParser(description="Quadratic renormalized fits for finite-sphere sweep.")
    ap.add_argument("--csv", type=str, default=str(pathlib.Path(__file__).resolve().parents[1] / "data" / "C3C4_finitesphere_sweep_Rstar.csv"))
    ap.add_argument("--deg", type=int, default=2)
    args = ap.parse_args()

    root = _ensure_dirs()
    df, y3_label, y4_label = _read_finite_sphere_csv(pathlib.Path(args.csv))

    x = df["x"].values
    y3 = df[y3_label].values
    y4 = df[y4_label].values

    coeffs3, r2_3, c3_ren, c3_err, p3 = _polyfit_with_report(x, y3, deg=args.deg)
    coeffs4, r2_4, c4_ren, c4_err, p4 = _polyfit_with_report(x, y4, deg=args.deg)

    # Write CSV with summary
    out_csv = root / "data" / "C3C4_renormalized_fits.csv"
    rows = [{
        "fit_degree": args.deg,
        "n_points": len(x),
        "ring_track": float(df.get("Rring_over_a", pd.Series([np.nan])).iloc[0]) if "Rring_over_a" in df else np.nan,
        "C3_intercept": c3_ren,
        "C3_intercept_err": c3_err,
        "C3_c1": coeffs3[-2],
        "C3_c2": coeffs3[-3] if len(coeffs3) >= 3 else 0.0,
        "C3_R2": r2_3,
        "C4_intercept": c4_ren,
        "C4_intercept_err": c4_err,
        "C4_c1": coeffs4[-2],
        "C4_c2": coeffs4[-3] if len(coeffs4) >= 3 else 0.0,
        "C4_R2": r2_4,
        "C4_over_C3_intercept": (c4_ren / c3_ren) if c3_ren != 0.0 else np.nan,
    }]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Make figure
    xx = np.linspace(min(x), max(x), 400)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150, constrained_layout=True)

    ax[0].scatter(x, y3, s=18, label=f"{y3_label} data")
    ax[0].plot(xx, np.poly1d(np.polyfit(x, y3, args.deg))(xx), label=f"fit deg {args.deg}\nC3_ren={c3_ren:.5g}±{c3_err:.2g}\nR²={r2_3:.3f}")
    ax[0].set_xlabel(r"$R_\star/a$"); ax[0].set_ylabel(r"$C_3$")
    ax[0].grid(True, alpha=0.35); ax[0].legend(loc="best", fontsize=9)

    ax[1].scatter(x, y4, s=18, label=f"{y4_label} data")
    ax[1].plot(xx, np.poly1d(np.polyfit(x, y4, args.deg))(xx), label=f"fit deg {args.deg}\nC4_ren={c4_ren:.5g}±{c4_err:.2g}\nR²={r2_4:.3f}")
    ax[1].set_xlabel(r"$R_\star/a$"); ax[1].set_ylabel(r"$C_4$")
    ax[1].grid(True, alpha=0.35); ax[1].legend(loc="best", fontsize=9)

    fig.suptitle("Renormalized quadratic fits (intercepts at $R_\star/a \\to 0$)")
    out_png = root / "figures" / "renormalized_fit_report.png"
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    print("Done.")

if __name__ == "__main__":
    main()
