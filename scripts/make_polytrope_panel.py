#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    csv = root / "data" / "C3C4_polytrope_n_sweep.csv"
    df = pd.read_csv(csv)

    # tolerant column names
    ncol = "n" if "n" in df else "polytropic_index"
    x = df[ncol].values.astype(float)
    C3 = df.get("C3", df.get("C3_int", df["C3_raw"])).values.astype(float)
    C4 = df.get("C4", df.get("C4_int", df["C4_raw"])).values.astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=150, constrained_layout=True)
    ax[0].plot(x, C3, marker="o"); ax[0].set_xlabel("polytrope index n"); ax[0].set_ylabel(r"$C_3$")
    ax[0].grid(True, alpha=0.35); ax[0].set_title("EOS sweep — $C_3$")
    ax[1].plot(x, C4, marker="s"); ax[1].set_xlabel("polytrope index n"); ax[1].set_ylabel(r"$C_4$")
    ax[1].grid(True, alpha=0.35); ax[1].set_title("EOS sweep — $C_4$")
    out = root / "figures" / "C3C4_polytrope_n_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
