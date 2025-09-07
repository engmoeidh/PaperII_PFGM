import numpy as np
from codes.fft_poisson import fft_poisson
from codes.pn_solver import ring_average  # reuse ring sampler

def _sphere_U_grad_centered(X, Y, Z, m=1.0, R=0.2):
    """
    Exact uniform-sphere Newtonian potential U and gradient ∇U at points (X,Y,Z),
    where the sphere is centered at the origin. G=c=1.
    """
    r2 = X*X + Y*Y + Z*Z
    r = np.sqrt(r2, dtype=np.float64)

    U  = np.empty_like(r)
    gx = np.empty_like(r); gy = np.empty_like(r); gz = np.empty_like(r)

    # outside
    outside = r > R
    invr = np.zeros_like(r); invr[outside] = 1.0/r[outside]
    U[outside] = m*invr[outside]
    fac_out = np.zeros_like(r); fac_out[outside] = -m*invr[outside]**3
    gx[outside] = fac_out[outside]*X[outside]
    gy[outside] = fac_out[outside]*Y[outside]
    gz[outside] = fac_out[outside]*Z[outside]

    # inside
    inside = ~outside
    # U(r) = m (3R^2 - r^2) / (2R^3);  ∇U = - m r_vec / R^3
    U[inside]  = m * (3.0*R*R - r2[inside]) / (2.0*R**3)
    gx[inside] = -m * X[inside] / (R**3)
    gy[inside] = -m * Y[inside] / (R**3)
    gz[inside] = -m * Z[inside] / (R**3)

    return U, gx, gy, gz

def build_two_spheres_fields(N=64, L=3.0, a=1.0, Rstar=0.3, m1=1.0, m2=1.0):
    """
    Two identical uniform spheres at (-a/2,0,0) and (+a/2,0,0).
    Returns x-grid, total U, and total gradient components (gx,gy,gz).
    """
    x = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Sphere centers
    X1 = X + a/2; Y1 = Y; Z1 = Z
    X2 = X - a/2; Y2 = Y; Z2 = Z

    U1, g1x, g1y, g1z = _sphere_U_grad_centered(X1, Y1, Z1, m=m1, R=Rstar)
    U2, g2x, g2y, g2z = _sphere_U_grad_centered(X2, Y2, Z2, m=m2, R=Rstar)

    U  = U1 + U2
    gx = g1x + g2x; gy = g1y + g2y; gz = g1z + g2z
    return x, U, gx, gy, gz

def build_single_sphere_fields(which='A', N=64, L=3.0, a=1.0, Rstar=0.3, m=1.0):
    """
    One sphere only (for inner-zone subtraction). 'A' at (-a/2,0,0), 'B' at (+a/2,0,0).
    """
    x = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    if which.upper() == 'A':
        Xc = X + a/2; Yc = Y; Zc = Z
    else:
        Xc = X - a/2; Yc = Y; Zc = Z
    U, gx, gy, gz = _sphere_U_grad_centered(Xc, Yc, Zc, m=m, R=Rstar)
    return x, U, gx, gy, gz

def build_sources(U, gx, gy, gz):
    """Return S3=U|∇U|^2 and S4=|∇U|^4."""
    g2 = gx*gx + gy*gy + gz*gz
    return U*g2, g2**2

def shape_coefficients_from_phi(phi3, phi4, xgrid, L, a, Rring):
    """
    Dimensionless shape coefficients at a ring of radius Rring in the orbital plane.
    """
    Ustar = (2.0)/a
    C3 = (a**3 / Ustar**3) * ring_average(phi3, xgrid, L, Rring)
    C4 = (a**6 / Ustar**4) * ring_average(phi4, xgrid, L, Rring)
    return C3, C4
