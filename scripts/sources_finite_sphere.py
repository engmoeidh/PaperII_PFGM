# --- Helper for scripts/make_convergence.py ---
def C3C4_for_uniform_sphere(R_star: float,
                            ring_radius_factor: float = 0.6,
                            box_radius_factor: float = 5.0,
                            n_grid: int = 32768) -> dict:
    """
    Thin wrapper returning {R_star, ring_radius, box_radius, C3, C4} for a uniform-density sphere.

    Parameters
    ----------
    R_star : float
        Physical radius of the sphere (code units).
    ring_radius_factor : float
        Ring radius as a multiple of R_star; ring_radius = ring_radius_factor * R_star.
        This controls the deconvolution/FFT padding ring in the Poisson solver.
    box_radius_factor : float
        Simulation box half-size as a multiple of R_star; box_radius = box_radius_factor * R_star.
    n_grid : int
        Grid size (power of two recommended).

    Returns
    -------
    out : dict
        Keys: 'R_star', 'ring_radius', 'box_radius', 'C3', 'C4'.
    """
    ring_radius = ring_radius_factor * R_star
    box_radius  = box_radius_factor * R_star

    # Prefer an existing implementation if present (the one used by make_finitesphere.py)
    try:
        # If your file already defines a lower-level solver, reuse it:
        # expected signature (R_star, ring_radius, box_radius, n_grid) -> dict with C3, C4
        compute = globals().get("compute_C3C4_uniform_sphere", None)
        if compute is None:
            raise AttributeError("compute_C3C4_uniform_sphere not found in sources_finite_sphere.py")

        res = compute(R_star=R_star,
                      ring_radius=ring_radius,
                      box_radius=box_radius,
                      n_grid=n_grid)

        # Sanity & standard keys
        C3 = float(res["C3"])
        C4 = float(res["C4"])
        return {
            "R_star": float(R_star),
            "ring_radius": float(ring_radius),
            "box_radius": float(box_radius),
            "C3": C3,
            "C4": C4,
        }
    except Exception as e:
        raise RuntimeError(
            "C3C4_for_uniform_sphere could not dispatch to your solver.\n"
            "Please expose `compute_C3C4_uniform_sphere(R_star, ring_radius, box_radius, n_grid)`\n"
            f"or update this helper to call your internal routine. Original error: {e}"
        ) from e
