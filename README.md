# Proto-Field Gravity II — PN & Binary Phenomenology

This repo hosts **reproducibility assets** for Paper II:
- 3PN/4PN conservative sector from PFGM (near-zone Poisson solves for \(S_3=U|\nabla U|^2\), \(S_4=|\nabla U|^4\))
- Gauge-invariant diagnostics \(\delta K(x)\), \(\delta z_1(x)\)
- 2.5PN dissipative mapping (SPA phasing)
- Binary-pulsar/LVK template bounds on \(\alpha\)

> The LaTeX manuscript is authored on **Overleaf**. This repo is for **code, CSVs, and figures**.

## Quickstart
```bash
conda env create -f environment.yml
conda activate pfgm_paper2

# regenerate all tables/figures:
python scripts/make_all.py

Artifacts

codes/ — FFT Poisson solver + PN helpers

data/ — CSVs (coefficients, sweeps, derived bounds)

figures/ — plots used in the paper

scripts/ — orchestration (make_all.py)

Reproducibility

A single entrypoint, scripts/make_all.py, regenerates the figures and CSVs used in the paper from scratch.

License

MIT (code) and CC-BY-4.0 (data).

How to cite

Please cite the Paper II preprint/journal when available; a CITATION.cff will be added with DOI details.
