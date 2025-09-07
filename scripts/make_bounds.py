#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 6.67430e-11; c = 299792458.0; Msun = 1.98847e30

def pick(df, names):
    for n in names:
        if n in df.columns:
            return float(df[n].iloc[0])
    raise KeyError(f"None of {names} in {list(df.columns)}")

def alpha_bound(M_msun, P_seconds, C3, C4, delta):
    M = M_msun * Msun
    Omega = 2*np.pi / P_seconds
    x = (G*M*Omega/c**3)**(2/3)
    denom = C3*(x**3) + C4*(x**4)
    if denom <= 0:
        return np.nan, x
    return delta * (M**4) / denom, x

def load_pulsars(root: pathlib.Path):
    pf = root/"data"/"pulsars_input.csv"
    if pf.exists():
        df = pd.read_csv(pf)
        if "system" not in df: raise KeyError("pulsars_input.csv must have 'system'")
        if "P_seconds" not in df:
            for alt in ["P","period_s","period"]:
                if alt in df: df=df.rename(columns={alt:"P_seconds"}); break
        if "M_Msun" not in df:
            for alt in ["M","Mtot_Msun","M_total_Msun","Mtot"]:
                if alt in df: df=df.rename(columns={alt:"M_Msun"}); break
        assert "P_seconds" in df and "M_Msun" in df
        return df[["system","P_seconds","M_Msun"]].copy()
    return pd.DataFrame([
        {"system":"Hulse–Taylor (B1913+16)", "P_seconds":27906.98, "M_Msun":2.828},
        {"system":"Double Pulsar (J0737-3039)", "P_seconds":8834.5,  "M_Msun":2.587},
    ])

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    (root/"data").mkdir(parents=True, exist_ok=True)
    (root/"figures").mkdir(parents=True, exist_ok=True)

    # renormalized fits
    ren_paths = [root/"data"/"C3C4_renormalized_fits.csv", root/"C3C4_renormalized_fits.csv"]
    ren = None
    for p in ren_paths:
        if p.exists():
            ren = pd.read_csv(p); break
    if ren is None:
        raise FileNotFoundError("C3C4_renormalized_fits.csv not found.")

    C3 = pick(ren, ["C3_intercept","C3_ren","C3ren","C3_0"])
    C4 = pick(ren, ["C4_intercept","C4_ren","C4ren","C4_0"])

    pulsars = load_pulsars(root)

    rows = []
    for _, r in pulsars.iterrows():
        for delta in (1e-6, 1e-7):
            amax, x = alpha_bound(r["M_Msun"], r["P_seconds"], C3, C4, delta)
            rows.append({"system":r["system"],"P_seconds":r["P_seconds"],"M_Msun":r["M_Msun"],
                         "delta":delta,"x":x,"alpha_max_m4":amax})
    out_csv = root/"data"/"pulsar_bounds.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    m = np.isfinite(df["alpha_max_m4"]) & (df["alpha_max_m4"]>0)
    dfp = df.loc[m].copy()
    systems = dfp["system"].unique().tolist()
    plt.figure(figsize=(8.2,4.8), dpi=150)
    xticks = np.arange(1, len(systems)+1)
    for j, delta in enumerate(sorted(dfp["delta"].unique())):
        sel = dfp["delta"]==delta
        y = dfp[sel].set_index("system")["alpha_max_m4"].reindex(systems).values
        xplot = xticks + 0.12*j
        plt.scatter(xplot, y, s=42, label=f"δ = {delta:g}")
    plt.xticks(xticks+0.06, systems, rotation=25, ha="right")
    plt.yscale("log"); plt.grid(True, which="both", alpha=0.35)
    plt.ylabel(r"$\alpha_{\max}\ [{\rm m}^4]$")
    plt.title(r"Pulsar bounds from $C_3^{\rm ren}, C_4^{\rm ren}$")
    plt.legend()
    plt.tight_layout()
    out_png = root/"figures"/"alpha_bounds.png"
    plt.savefig(out_png); plt.close()
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()
