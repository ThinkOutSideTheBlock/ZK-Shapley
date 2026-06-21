# ZK-Shapley: Privacy-Preserving Mechanism Design for Federated Supply Chain Forecasting

Reference implementation and experiment suite supporting the preprint
*"ZK-Shapley: Privacy-Preserving Mechanism Design for Incentive-Compatible
Federated Demand Forecasting in Competitive Supply Chains."*

## What this is

A coalitional game between competing firms who can pool demand data via
federated learning to forecast better, but who must be paid fairly (Shapley
value) and cannot profit from misreporting (free-riding, data-size inflation,
noise injection) -- enforced by a cryptographic commit-reveal layer standing
in for a zk-SNARK honesty proof, whose proving/verification cost is estimated
from published zkML benchmarks rather than implemented from scratch.

## Layout

```
src/
  demand.py        Multi-firm correlated demand generator (AR(1) + shared latent factor)
  federated.py      Exact federated ridge regression (sufficient-statistics aggregation)
  shapley.py        Exact + Monte Carlo Shapley value engine, axiom verification
  commitment.py     Hash commit-reveal scheme (real, tested) + ZK cost model (literature-calibrated estimate)
  mechanism.py      ZK-Shapley protocol, baseline mechanisms, deviation/attack simulation
  inventory.py      Multi-echelon (R,S) base-stock supply chain simulator
tests/              34 tests, 97% line coverage on src/ -- see "What is actually verified" below
experiments/
  run_main_experiment.py   Produces every numerical result in results/*.csv, *.json
  generate_figures.py       Produces every figure in figures/*.pdf, *.png
results/            Numerical outputs (regenerate via run_main_experiment.py)
figures/            7 publication figures (regenerate via generate_figures.py)
```

## Reproducing everything

```bash
pip install numpy scipy pandas matplotlib pytest pytest-cov --break-system-packages
python3 -m pytest tests/ -v --cov=src           # 34 tests, ~2s
python3 -m experiments.run_main_experiment       # writes results/*.csv, *.json
python3 -m experiments.generate_figures          # writes figures/*.pdf, *.png
```

All randomness is seeded; results are bit-reproducible.

## What is actually verified (not just smoke-tested)

- **Federated ridge regression is mathematically exact**, not approximate:
  `test_federated_equals_centralized_ridge` proves the sufficient-statistics
  aggregation produces a bit-identical model to centralizing raw data.
- **Shapley engine validated against three toy games with closed-form
  solutions** (unanimity game, additive game, glove game) -- not just the
  complex FL pipeline, so a math bug can't hide behind pipeline complexity.
- **Efficiency, symmetry, and null-player axioms verified numerically**, both
  on toy games and end-to-end on the real demand-forecasting game.
- **Inventory simulator validated against a deterministic zero-uncertainty
  case** with an analytically known correct answer (zero backorders).
  Initial implementation had a genuine off-by-one pipeline-delay bug, caught
  by this test and fixed (see git history / inline comments in `inventory.py`).
- **Commitment scheme binding and hiding properties tested directly**
  (tamper detection, nonce-based unlinkability).
- **Mechanism budget-balance (efficiency axiom) verified through the full
  protocol stack**, not just the abstract Shapley formula.

## Honest scoping decisions (stated, not hidden)

1. **No zk-SNARK circuit is implemented.** `commitment.py` implements a real
   hash commit-reveal scheme (binding + hiding, fully tested) for
   anti-equivocation, and estimates SNARK proving/verification cost from the
   ZKML EuroSys 2024 benchmark (Chen et al.) rather than building a circuit.
   Building one is a separate, multi-month systems contribution.
2. **No universal incentive-compatibility theorem is claimed.** Shapley
   mechanisms are fair (efficiency/symmetry/null-player, proven and tested)
   but not provably strategy-proof against arbitrary manipulation in general
   cooperative games. What is claimed and demonstrated: (a) the mechanism is
   structurally immune to data-size misreporting (no self-reported size
   parameter exists to misreport), and (b) free-riding and noise-injection
   are empirically penalized via realized-performance-based payment.
3. **Federated model is ridge regression, not a deep forecaster.** This is a
   deliberate choice for mathematical tractability (exact, not SGD-approximate,
   aggregation) and clean Shapley computation. Extension to nonlinear models
   (TFT, N-HiTS) is future work, noted explicitly in the paper's limitations.
