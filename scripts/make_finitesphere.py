#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from codes.fft_poisson import fft_poisson
from codes.sources_finite_sphere import (
    build_two_spheres_fields, build_single_sphere_fields,
    build_sources, shape_coefficients_from_phi
)

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def sweep_finite_spheres(N=64, L=3.0, a=1.0, Rlist=(0.05,0.10,0.20,0.30), Rrings=(0.8,1.0)):
    rows=[]
    for Rstar in Rlist:
        # two-body fields
        x, U2, gx2, gy2, gz2 = build_two_spheres_fields(N=N, L=L, a=a, Rstar=Rstar, m1=1.0, m2=1.0)
        S3_2, S4_2 = build_sources(U2, gx2, gy2, gz2)
        phi3_2 = fft_poisson(S3_2, L=L)
        phi4_2 = fft_poisson(S4_2, L=L)

        # single-body fields (for inner-zone subtraction)
        _, U1A, g1Ax, g1Ay, g1Az = build_single_sphere_fields('A', N=N, L=L, a=a, Rstar=Rstar, m=1.0)
        _, U1B, g1Bx, g1By, g1Bz = build_single_sphere_fields('B', N=N, L=L, a=a, Rstar=Rstar, m=1.0)
        S3_1A, S4_1A = build_sources(U1A, g1Ax, g1Ay, g1Az)
        S3_1B, S4_1B = build_sources(U1B, g1Bx, g1By, g1Bz)
        phi3_1A = fft_poisson(S3_1A, L=L); phi3_1B = fft_poisson(S3_1B, L=L)
        phi4_1A = fft_poisson(S4_1A, L=L); phi4_1B = fft_poisson(S4_1B, L=L)

        # interaction-only potentials
        phi3_int = phi3_2 - (phi3_1A + phi3_1B)
        phi4_int = phi4_2 - (phi4_1A + phi4_1B)

        for Rring in Rrings:
            C3_raw, C4_raw = shape_coefficients_from_phi(phi3_2,   phi4_2,   x, L, a, Rring*a)
            C3_int, C4_int = shape_coefficients_from_phi(phi3_int, phi4_int, x, L, a, Rring*a)
            rows.append({
                "Rstar_over_a": Rstar,
                "Rring_over_a": Rring,
                "N": N, "L": L, "a": a,
                "C3_raw": C3_raw, "C4_raw": C4_raw,
                "C3_int": C3_int, "C4_int": C4_int,
                "ratio_int": C4_int / C3_int if C3_int != 0 else np.nan
            })

    df = pd.DataFrame(rows).sort_values(["Rring_over_a","Rstar_over_a"])
    df.to_csv("data/C3C4_finitesphere_sweep_Rstar.csv", index=False)
    return df

def plot_finite_radius(df):
    plt.figure(figsize=(6.2,4.4))
    for Rring in sorted(df["Rring_over_a"].unique()):
        sub = df[df["Rring_over_a"]==Rring]
        plt.plot(sub["Rstar_over_a"], sub["C3_int"], 'o-', label=f"C3_int (Rring/a={Rring})")
    plt.xlabel("R*/a"); plt.ylabel("C3 (interaction-only)")
    plt.tight_layout(); plt.savefig("figures/C3C4_finite_radius_vs_Rstar.png", dpi=200)

if __name__ == "__main__":
    df = sweep_finite_spheres()
    plot_finite_radius(df)
    print("OK: data/C3C4_finitesphere_sweep_Rstar.csv and figures/C3C4_finite_radius_vs_Rstar.png written.")
