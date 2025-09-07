#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polytrope physics generator:
- Solve Lane–Emden for n in a list (default 0.5, 1.0, 1.5).
- Build single-star polytrope U, ∇U on a 3D grid (centered).
- Compute S3 = U|∇U|^2, S4 = |∇U|^4.
- Solve ∇^2 φ3 = S3, ∇^2 φ4 = S4 via fft_poisson.
- Extract C3, C4 by ring averaging at radius R_ring in the orbital plane.
- Normalize exactly like your sphere script by choosing a = 2 R_ring so U_* = 1/R_ring.

Outputs:
  data/C3C4_polytrope_n_sweep.csv
  figures/C3C4_polytrope_n_panel.png
"""

from __future__ import annotations
import sys, pathlib, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from codes.fft_poisson import fft_poisson
from codes.pn_solver import ring_average  # your existing ring sampler

# ------------------------------
# Lane–Emden solver (RK4)
# ------------------------------

def lane_emden(n: float, dxi: float = 1e-3, xi_max: float = 30.0):
    """
    Solve θ'' + 2/ξ θ' + θ^n = 0,  θ(0)=1, θ'(0)=0, stop at first zero.
    Uses RK4 with series start near ξ=0 to avoid singular term.

    Returns:
        xi : (m,) array
        theta : (m,) array
        dtheta : (m,) array
        xi1 : first zero of theta (float)
        dtheta_at_xi1 : θ'(xi1) estimated by last step
    """
    # series start
    xi0 = dxi
    theta0 = 1.0 - (xi0*xi0)/6.0
    dtheta0 = -xi0/3.0

    xi = [0.0, xi0]
    theta = [1.0, theta0]
    dtheta = [0.0, dtheta0]

    def f(xi, y, yp):
        # y  = θ, yp = θ'
        # θ'' = -2/ξ θ' - θ^n
        return -2.0*yp/xi - (max(y, 0.0))**n  # guard negative θ for fractional n

    x = xi0
    y = theta0
    yp = dtheta0

    while x < xi_max:
        # RK4 step for θ and θ'
        k1 = dxi * yp
        l1 = dxi * f(x, y, yp)

        k2 = dxi * (yp + 0.5*l1)
        l2 = dxi * f(x + 0.5*dxi, y + 0.5*k1, yp + 0.5*l1)

        k3 = dxi * (yp + 0.5*l2)
        l3 = dxi * f(x + 0.5*dxi, y + 0.5*k2, yp + 0.5*l2)

        k4 = dxi * (yp + l3)
        l4 = dxi * f(x + dxi, y + k3, yp + l3)

        y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6.0
        yp_new = yp + (l1 + 2*l2 + 2*l3 + l4)/6.0
        x_new = x + dxi

        xi.append(x_new); theta.append(y_new); dtheta.append(yp_new)
        x, y, yp = x_new, y_new, yp_new

        if y_new <= 0.0 and len(theta) > 3:
            break

    xi = np.array(xi, dtype=float)
    theta = np.array(theta, dtype=float)
    dtheta = np.array(dtheta, dtype=float)

    # linear interpolation to zero crossing
    if theta[-1] < 0.0:
        x1, x2 = xi[-2], xi[-1]
        y1, y2 = theta[-2], theta[-1]
        t = -y1 / (y2 - y1)
        xi1 = x1 + t*(x2 - x1)
        dtheta_at_xi1 = dtheta[-2] + t*(dtheta[-1] - dtheta[-2])
    else:
        xi1 = xi[-1]
        dtheta_at_xi1 = dtheta[-1]

    return xi, theta, dtheta, xi1, dtheta_at_xi1


def polytrope_profiles(n: float, R_star: float, M_tot: float = 1.0,
                       Nr: int = 2000):
    """
    Build spherical profiles ρ(r), M(r), U(r) for a Lane–Emden polytrope with total mass M_tot and radius R_star.
    Scaling:
        r = α ξ, with α = R_star / ξ1.
        M(r) = M_tot * m(ξ)/m(ξ1), with m(ξ) = -ξ^2 θ'(ξ).
    Potential:
        U(r) = M_tot/r for r ≥ R_star; inside: integrate dU/dr = - M(r)/r^2 with U(R_star)=M_tot/R_star.

    Returns:
        r (Nr,), rho (Nr,), M (Nr,), U (Nr,), g_r (Nr,)  (g_r = ∂U/∂r)
    """
    xi, theta, dtheta, xi1, dtheta1 = lane_emden(n)
    alpha = R_star / xi1

    # dimensionless mass m(ξ) = - ξ^2 θ'
    m_xi = -xi**2 * dtheta
    m1 = -xi1**2 * dtheta1

    # map to physical r-grid
    r = np.linspace(0.0, R_star, Nr)
    # avoid r=0 division later
    r[0] = 0.0

    # interpolate θ(ξ), θ'(ξ) on ξ(r)
    xi_r = np.where(R_star > 0, r/alpha, 0.0)
    theta_r = np.interp(xi_r, xi, theta, left=1.0, right=0.0)
    dtheta_r = np.interp(xi_r, xi, dtheta, left=0.0, right=dtheta1)

    m_r = np.interp(xi_r, xi, m_xi, left=0.0, right=m1)
    M_r = M_tot * m_r / m1
    rho = np.clip(theta_r, 0.0, None)**n  # central density absorbed into scaling (only shapes matter)

    # gravitational field: g_r = ∂U/∂r = - M(r) / r^2 (radial derivative)
    g_r = np.zeros_like(r)
    # inside r>0
    mask = r > 0.0
    g_r[mask] = - M_r[mask] / (r[mask]**2)
    # outside (for completeness): g_r_out = - M_tot / r^2; potential will handle with boundary

    # potential: integrate inward from boundary with U(R) = M_tot/R
    U = np.zeros_like(r)
    U[-1] = M_tot / R_star
    # integrate backward
    for i in range(Nr-2, -1, -1):
        dr = r[i+1] - r[i]
        # midpoint slope (simple trapezoid on dU/dr)
        dU = 0.5 * (g_r[i+1] + g_r[i]) * dr
        U[i] = U[i+1] + dU  # dU/dr = g_r

    return r, rho, M_r, U, g_r


# ------------------------------
# Map spherical profiles to 3D grid and compute C3,C4
# ------------------------------

def polytrope_C3C4_single(n: float, R_star: float,
                          N: int, L: float,
                          R_ring: float) -> dict:
    """
    Build single polytrope centered at origin on a cubic grid:
      - Grid: x ∈ [-L, L], N points per axis.
      - Use mass normalization M_tot=1 so U→1/r outside (consistent with your sphere code).
      - Compute S3, S4 -> φ3, φ4 via fft_poisson.
      - Extract C3, C4 using ring_average with a = 2 R_ring (so U_* = 1/R_ring).
    """
    # radial profiles
    r1d, rho1d, M1d, U1d, g_r1d = polytrope_profiles(n=n, R_star=R_star, M_tot=1.0, Nr=max(2000, 10*N))

    # 3D grid
    x = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X*X + Y*Y + Z*Z, dtype=np.float64)

    # sample U(r), g_r(r)
    # outside region: U = 1/r, g_r = -1/r^2
    U = np.empty_like(r)
    g_r = np.empty_like(r)

    inside = r <= R_star
    outside = ~inside
    # interpolate inside
    U[inside] = np.interp(r[inside], r1d, U1d)
    g_r[inside] = np.interp(r[inside], r1d, g_r1d)
    # analytic outside
    with np.errstate(divide="ignore", invalid="ignore"):
        U[outside] = 1.0 / np.where(r[outside] > 0, r[outside], np.inf)
        g_r[outside] = -1.0 / np.where(r[outside] > 0, r[outside]**2, np.inf)

    # components of ∇U = (∂U/∂r) * r̂  = g_r * (x/r, y/r, z/r)
    # note: r=0 handled by setting components to 0 there
    invr = np.zeros_like(r)
    mask = r > 0
    invr[mask] = 1.0 / r[mask]
    gx = g_r * X * invr
    gy = g_r * Y * invr
    gz = g_r * Z * invr

    # sources & Poisson
    g2 = gx*gx + gy*gy + gz*gz
    S3 = U * g2
    S4 = g2 * g2

    phi3 = fft_poisson(S3, L=L)
    phi4 = fft_poisson(S4, L=L)

    # normalization: choose a = 2*R_ring => U_* = 2/a = 1/R_ring
    a = 2.0 * R_ring
    C3 = (a**3) * (ring_average(phi3, x, L, R_ring) * (R_ring**3))  # since 1/U_*^3 = (R_ring)^3
    C4 = (a**6) * (ring_average(phi4, x, L, R_ring) * (R_ring**4))  # since 1/U_*^4 = (R_ring)^4

    return {
        "n": float(n),
        "R_star": float(R_star),
        "R_ring": float(R_ring),
        "L": float(L),
        "N": int(N),
        "C3": float(C3),
        "C4": float(C4),
        "ratio": float(C4/C3) if C3 != 0.0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate polytrope EOS sweep (C3,C4) with full physics.")
    ap.add_argument("--n_list", type=str, default="0.5,1.0,1.5", help="Comma-separated polytrope indices.")
    ap.add_argument("--Rstar_over_a", type=float, default=0.30, help="Star radius in units of a.")
    ap.add_argument("--ring_factor", type=float, default=0.60, help="R_ring / a.")
    ap.add_argument("--box_factor", type=float, default=5.0, help="L / a (half-box size in units of a).")
    ap.add_argument("--N", type=int, default=96, help="Grid points per axis (3D).")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)

    n_list = [float(s) for s in args.n_list.split(",") if s.strip()]

    a = 1.0  # arbitrary unit length for this computation; all ratios use a
    R_star = args.Rstar_over_a * a
    R_ring = args.ring_factor * a
    L = args.box_factor * a

    rows = []
    for n in n_list:
        print(f"[polytrope] n={n:.3g}  R*/a={args.Rstar_over_a:.3f}  Rring/a={args.ring_factor:.3f}  N={args.N}  L/a={args.box_factor:.3f}")
        out = polytrope_C3C4_single(n=n, R_star=R_star, N=args.N, L=L, R_ring=R_ring)
        rows.append(out)

    df = pd.DataFrame(rows)
    csv_path = root / "data" / "C3C4_polytrope_n_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Figure (panel)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=150, constrained_layout=True)
    ax[0].plot(df["n"], df["C3"], marker="o"); ax[0].set_xlabel("polytrope index n"); ax[0].set_ylabel(r"$C_3$")
    ax[0].grid(True, alpha=0.35); ax[0].set_title("EOS sweep — $C_3$")
    ax[1].plot(df["n"], df["C4"], marker="s"); ax[1].set_xlabel("polytrope index n"); ax[1].set_ylabel(r"$C_4$")
    ax[1].grid(True, alpha=0.35); ax[1].set_title("EOS sweep — $C_4$")
    out_png = root / "figures" / "C3C4_polytrope_n_panel.png"
    fig.savefig(out_png); plt.close(fig)
    print(f"Wrote {out_png}")
    print("OK: polytrope physics CSV+figure written.")

if __name__ == "__main__":
    main()
