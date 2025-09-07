import numpy as np
from codes.fft_poisson import fft_poisson

def plummer_binary_potential(N=96, L=3.0, a=1.0, eps=0.05, m1=1.0, m2=1.0):
    x = np.linspace(-L, L, N)
    X,Y,Z = np.meshgrid(x,x,x, indexing='ij')
    r1 = np.sqrt((X + a/2)**2 + Y**2 + Z**2 + eps**2)
    r2 = np.sqrt((X - a/2)**2 + Y**2 + Z**2 + eps**2)
    U  = m1/r1 + m2/r2
    gx = -m1*(X + a/2)/r1**3 - m2*(X - a/2)/r2**3
    gy = -m1*Y/r1**3        - m2*Y/r2**3
    gz = -m1*Z/r1**3        - m2*Z/r2**3
    return x, U, gx, gy, gz

def build_sources(U, gx, gy, gz):
    g2 = gx*gx + gy*gy + gz*gz
    return U*g2, g2**2  # S3, S4

def ring_average(field, xgrid, L, R, ntheta=720):
    N = xgrid.size
    dx = (2*L)/N
    vals = []
    for th in np.linspace(0, 2*np.pi, ntheta, endpoint=False):
        px, py, pz = R*np.cos(th), R*np.sin(th), 0.0
        ix = int(round((px - xgrid[0]) / dx)); ix = np.clip(ix,0,N-1)
        iy = int(round((py - xgrid[0]) / dx)); iy = np.clip(iy,0,N-1)
        iz = int(round((pz - xgrid[0]) / dx)); iz = np.clip(iz,0,N-1)
        vals.append(field[ix, iy, iz])
    return float(np.mean(vals))

def compute_shape_coeffs(N=96, L=3.0, a=1.0, eps=0.05):
    x, U, gx, gy, gz = plummer_binary_potential(N=N, L=L, a=a, eps=eps)
    S3, S4 = build_sources(U, gx, gy, gz)
    phi3 = fft_poisson(S3, L=L)
    phi4 = fft_poisson(S4, L=L)
    R = 0.5*a; Ustar = (2.0)/a
    C3 = (a**3 / Ustar**3) * ring_average(phi3, x, L, R)
    C4 = (a**6 / Ustar**4) * ring_average(phi4, x, L, R)
    return {"N":N, "L":L, "a":a, "eps":eps, "C3":C3, "C4":C4, "ratio": C4/C3}
