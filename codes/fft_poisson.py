import numpy as np
import numpy.fft as nfft

def fft_poisson(S, L):
    """Solve ∇²φ = -4π S on a periodic cube [-L, L]^3 with N^3 samples; enforce <φ>=0."""
    S = np.asarray(S, dtype=np.float64)
    N = S.shape[0]
    kvec = nfft.fftfreq(N, d=(2*L)/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(kvec, kvec, kvec, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz
    S_k = nfft.fftn(S)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_k = np.where(k2!=0.0, (4.0*np.pi/k2)*S_k, 0.0)
    return np.real(nfft.ifftn(phi_k))
