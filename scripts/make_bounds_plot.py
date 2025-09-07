#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    csv = root / "data" / "pulsar_bounds.csv"
    df = pd.read_csv(csv)

    # try to detect alpha columns (allow multiple tolerances)
    alpha_cols = [c for c in df.columns if c.lower().startswith("alpha") or "m4" in c.lower()]
    if not alpha_cols:
        raise RuntimeError("No alpha_* columns found in pulsar_bounds.csv")
    labels = df.get("system", df.get("name", pd.Series([f"row{i}" for i in range(len(df))])))

    plt.figure(figsize=(7.5, 4.5), dpi=150)
    for j, col in enumerate(alpha_cols):
        y = df[col].values.astype(float)
        x = np.arange(len(y)) + 1 + 0.1*j
        plt.scatter(x, y, label=col, s=32)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=30, ha="right")
    plt.ylabel(r"$\alpha_{\max}\ [{\rm length}^4]$")
    plt.yscale("log"); plt.grid(True, which="both", alpha=0.35)
    plt.title("Pulsar bounds on $\\alpha$")
    plt.legend()
    out = root / "figures" / "alpha_bounds.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
