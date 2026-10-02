# Reproducibility (R2)

Commit: record `git rev-parse HEAD` at freeze.

## Environment
- Python 3.11+ (project used 3.13)
- `numpy`, `scipy`, `pandas`, `pytest`, `statsmodels` (probes only)

## Core tests
```bash
pytest tests/ -q
python -m experiments.run_r2_smoke