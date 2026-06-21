"""
The ZK-Shapley mechanism and baseline payment mechanisms, plus the deviation
(attack) simulation used to empirically characterize incentive properties.

Coalitional value function
---------------------------
v(S) = u(S) - u(emptyset), where u(S) = -RMSE(model trained on coalition S's
sufficient statistics, evaluated on the FIXED global holdout set = union of
all firms' private chronological holdout splits). u is monotone in expectation
(more correlated-but-noisy data weakly improves out-of-sample accuracy via
variance reduction -- see demand.py docstring) but not guaranteed monotone
pointwise on any single finite sample; this is reported, not assumed, via the
monotonicity-violation-rate diagnostic in experiments/run_main_experiment.py.

Payment mechanisms compared
----------------------------
1. ZK_SHAPLEY (proposed): T_i = phi_i(v), the exact/MC Shapley value of the
   verified-realized-performance game. Budget-balanced by the efficiency
   axiom: sum_i T_i = v(N), exactly.
2. EQUAL_SPLIT (baseline): T_i = v(N) / n. Ignores realized contribution
   entirely -- the standard naive "everyone gets an equal cut" rule used in
   many real-world data-pooling consortia. Structurally vulnerable to
   free-riding (a contributing-nothing firm is paid the same as the top
   contributor).
3. PROPORTIONAL_TO_REPORTED_SIZE (baseline): T_i = (n_i_reported / sum_j
   n_j_reported) * v(N). This is the common "pay by data volume" heuristic.
   Structurally vulnerable to data-quantity misreporting: n_i_reported is a
   SELF-REPORTED number with no verification step in this baseline, so a firm
   can claim more rows than it has and capture a larger share of v(N) without
   contributing more value. ZK_SHAPLEY has no such attack surface BY
   CONSTRUCTION, because it never takes any self-reported quantity as an
   input -- payment depends only on independently verified realized
   performance.

Deviation (attack) simulation
-------------------------------
For each firm, we compute payoff under:
  - HONEST: true statistics submitted.
  - FREE_RIDE: firm submits all-zero sufficient statistics (zero marginal
    effort) but still claims protocol membership.
  - NOISE_INJECTION: firm computes statistics from a deliberately corrupted
    version of its own training data (label noise injected), i.e. it pays the
    same effort cost as being honest but degrades the data quality it
    contributes.
  - DATA_INFLATION (PROPORTIONAL mechanism only -- structurally inapplicable
    to ZK_SHAPLEY, which is the point): firm claims a larger n_i_reported than
    its true row count, with no corresponding increase in actual data.

This produces an empirical, non-overclaimed characterization of incentive
robustness: ZK_SHAPLEY weakly dominates the tested deviations for the firm
under consideration, and is structurally immune to size-misreporting in a way
the baseline is not -- without asserting a blanket Bayesian-Nash-equilibrium
theorem over an unbounded strategy space.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .federated import FirmDataset, fit_and_evaluate_coalition
from .shapley import CachedGame, exact_shapley, monte_carlo_shapley
from .commitment import commit, verify, estimate_zk_cost


def make_value_function(datasets: dict[str, FirmDataset], lam: float = 1.0):
    """
    Returns v: frozenset -> float, normalized so v(emptyset) = 0, where the
    raw utility u(S) = -RMSE(S) is evaluated on the fixed global holdout
    (union of all firms' private holdout splits).
    """
    all_firms = frozenset(datasets.keys())
    baseline_rmse = fit_and_evaluate_coalition(datasets, frozenset(), all_firms, lam=lam)

    def v(S: frozenset) -> float:
        rmse_S = fit_and_evaluate_coalition(datasets, S, all_firms, lam=lam)
        return baseline_rmse - rmse_S  # utility gain over the no-sharing baseline

    return v, baseline_rmse


@dataclass
class MechanismResult:
    name: str
    payments: dict[str, float]
    total_budget: float
    rmse_grand_coalition: float
    rmse_baseline: float


def run_zk_shapley_mechanism(
    datasets: dict[str, FirmDataset],
    lam: float = 1.0,
    exact: bool = True,
    mc_permutations: int = 2000,
    mc_seed: int = 0,
    verify_commitments: bool = True,
) -> tuple[MechanismResult, dict]:
    """
    Full protocol simulation: (1) each firm computes sufficient statistics and
    commits to them, (2) commitments are revealed and verified (binding check
    -- catches any post-hoc tampering), (3) the smart-contract-equivalent
    computes v(S) from verified statistics for the Shapley calculation, (4)
    payments are issued. Returns the mechanism result plus a diagnostics dict
    (commitment verification outcomes, ZK cost estimate, Shapley computation
    cost in terms of characteristic-function evaluations).
    """
    rng = np.random.default_rng(mc_seed)
    firms = list(datasets.keys())

    # --- (1)+(2): commit-reveal round, one commitment per firm to its sufficient statistics ---
    from .federated import sufficient_statistics
    commitments, revealed_ok = {}, {}
    total_train_rows = 0
    for name in firms:
        A, b = sufficient_statistics(datasets[name])
        c = commit(A, b, rng=rng)
        commitments[name] = c
        if verify_commitments:
            revealed_ok[name] = verify(c, A, b)
        total_train_rows += datasets[name].n_train
    assert all(revealed_ok.values()) if verify_commitments else True, "Commitment verification failed"

    # --- (3): characteristic function + Shapley computation ---
    v, baseline_rmse = make_value_function(datasets, lam=lam)
    game = CachedGame(firms, v)

    if exact and len(firms) <= 12:
        phi = exact_shapley(game)
        mc_diag = None
    else:
        phi, mc_diag = monte_carlo_shapley(game, n_permutations=mc_permutations, seed=mc_seed)

    rmse_grand = fit_and_evaluate_coalition(datasets, frozenset(firms), frozenset(firms), lam=lam)
    total_budget = sum(phi.values())

    result = MechanismResult(
        name="ZK_SHAPLEY",
        payments=phi,
        total_budget=total_budget,
        rmse_grand_coalition=rmse_grand,
        rmse_baseline=baseline_rmse,
    )

    # average ZK cost estimate across firms, using each firm's true row count
    from .federated import N_FEATURES
    zk_costs = {name: estimate_zk_cost(datasets[name].n_train, N_FEATURES) for name in firms}

    diagnostics = {
        "commitments_verified": revealed_ok,
        "n_characteristic_function_evaluations": game.n_evaluations,
        "monte_carlo_diagnostics": mc_diag,
        "zk_cost_per_firm": zk_costs,
        "total_train_rows": total_train_rows,
    }
    return result, diagnostics


def run_equal_split_mechanism(datasets: dict[str, FirmDataset], lam: float = 1.0) -> MechanismResult:
    """Baseline: split the grand-coalition surplus equally regardless of contribution."""
    firms = list(datasets.keys())
    v, baseline_rmse = make_value_function(datasets, lam=lam)
    v_N = v(frozenset(firms))
    payments = {name: v_N / len(firms) for name in firms}
    rmse_grand = fit_and_evaluate_coalition(datasets, frozenset(firms), frozenset(firms), lam=lam)
    return MechanismResult(
        name="EQUAL_SPLIT", payments=payments, total_budget=sum(payments.values()),
        rmse_grand_coalition=rmse_grand, rmse_baseline=baseline_rmse,
    )


def run_proportional_mechanism(
    datasets: dict[str, FirmDataset],
    lam: float = 1.0,
    reported_sizes: dict[str, int] | None = None,
) -> MechanismResult:
    """
    Baseline: split the grand-coalition surplus proportionally to SELF-REPORTED
    sample counts. If `reported_sizes` is None, uses true counts (the honest
    case); pass an inflated dict to simulate the data-inflation attack.
    """
    firms = list(datasets.keys())
    v, baseline_rmse = make_value_function(datasets, lam=lam)
    v_N = v(frozenset(firms))
    sizes = reported_sizes or {name: datasets[name].n_train for name in firms}
    total = sum(sizes.values())
    payments = {name: (sizes[name] / total) * v_N for name in firms}
    rmse_grand = fit_and_evaluate_coalition(datasets, frozenset(firms), frozenset(firms), lam=lam)
    return MechanismResult(
        name="PROPORTIONAL", payments=payments, total_budget=sum(payments.values()),
        rmse_grand_coalition=rmse_grand, rmse_baseline=baseline_rmse,
    )


def run_no_sharing_baseline(datasets: dict[str, FirmDataset], lam: float = 1.0) -> MechanismResult:
    """Baseline: no federation at all. Zero payments; reported for the RMSE/cost comparison only."""
    firms = list(datasets.keys())
    _, baseline_rmse = make_value_function(datasets, lam=lam)
    return MechanismResult(
        name="NO_SHARING", payments={name: 0.0 for name in firms}, total_budget=0.0,
        rmse_grand_coalition=baseline_rmse, rmse_baseline=baseline_rmse,
    )


# ---------------------------------------------------------------------------
# Deviation (attack) simulation
# ---------------------------------------------------------------------------

def free_ride_dataset(ds: FirmDataset) -> FirmDataset:
    """
    Firm contributes zero-effort: all-zero training rows (still correctly shaped).

    Empirical note (verified in experiments/run_main_experiment.py, Experiment
    3): under ZK_SHAPLEY this typically yields a NEGATIVE payment, not merely
    a zero payment. This is a correct, non-obvious consequence of the value
    function, not an artifact: when the free-riding firm is the sole member
    of a coalition (the "going first" case in the Shapley permutation sum),
    training a model on all-zero sufficient statistics collapses to the
    trivial zero-weight predictor (every demand forecast = 0), which performs
    far WORSE on the held-out evaluation set than the no-coalition autarky
    baseline (each firm predicting its own historical mean). Since the
    all-zero contribution is actively harmful in that scenario (and neutral,
    i.e. zero marginal contribution, in every other coalition the firm could
    join), the Shapley-weighted average payoff is pulled negative. This
    strengthens rather than weakens the incentive argument: free-riding is
    not merely unrewarded under ZK_SHAPLEY, it is penalized in expectation.
    """
    zeroed = copy.deepcopy(ds)
    zeroed.X_train = np.zeros_like(ds.X_train)
    zeroed.y_train = np.zeros_like(ds.y_train)
    return zeroed


def inject_noise_dataset(ds: FirmDataset, noise_std_multiplier: float, seed: int) -> FirmDataset:
    """
    Firm submits statistics computed from its OWN real feature rows but with
    labels corrupted by additive Gaussian noise scaled relative to the
    in-sample label standard deviation -- modeling a firm that puts in the
    same nominal "effort" (real data, real computation, a valid ZK proof of
    correct computation over THAT corrupted data) but degrades signal quality,
    e.g. to deny competitors full benefit while still claiming participation credit.
    """
    rng = np.random.default_rng(seed)
    corrupted = copy.deepcopy(ds)
    label_std = ds.y_train.std() if ds.y_train.shape[0] > 0 else 1.0
    noise = rng.normal(0.0, noise_std_multiplier * label_std, size=ds.y_train.shape)
    corrupted.y_train = ds.y_train + noise
    return corrupted


def deviation_payoff_analysis(
    datasets: dict[str, FirmDataset],
    target_firm: str,
    lam: float = 1.0,
    noise_std_multiplier: float = 2.0,
    inflation_factor: float = 3.0,
    seed: int = 0,
) -> dict:
    """
    For `target_firm`, compute ZK_SHAPLEY and PROPORTIONAL payoffs under
    HONEST, FREE_RIDE, NOISE_INJECTION, and (proportional-mechanism-only)
    DATA_INFLATION deviations, holding all other firms' behavior fixed at
    honest. This is the core empirical evidence for the paper's incentive
    analysis section.
    """
    results = {}

    # HONEST baseline for both mechanisms
    zk_honest, _ = run_zk_shapley_mechanism(datasets, lam=lam, mc_seed=seed)
    prop_honest = run_proportional_mechanism(datasets, lam=lam)
    results["honest"] = {
        "zk_shapley_payment": zk_honest.payments[target_firm],
        "proportional_payment": prop_honest.payments[target_firm],
    }

    # FREE_RIDE: replace target firm's dataset with zero-effort data
    ds_freeride = copy.deepcopy(datasets)
    ds_freeride[target_firm] = free_ride_dataset(datasets[target_firm])
    zk_fr, _ = run_zk_shapley_mechanism(ds_freeride, lam=lam, mc_seed=seed)
    prop_fr = run_proportional_mechanism(ds_freeride, lam=lam)
    results["free_ride"] = {
        "zk_shapley_payment": zk_fr.payments[target_firm],
        "proportional_payment": prop_fr.payments[target_firm],
    }

    # NOISE_INJECTION: corrupt target firm's labels
    ds_noisy = copy.deepcopy(datasets)
    ds_noisy[target_firm] = inject_noise_dataset(datasets[target_firm], noise_std_multiplier, seed)
    zk_noisy, _ = run_zk_shapley_mechanism(ds_noisy, lam=lam, mc_seed=seed)
    prop_noisy = run_proportional_mechanism(ds_noisy, lam=lam)
    results["noise_injection"] = {
        "zk_shapley_payment": zk_noisy.payments[target_firm],
        "proportional_payment": prop_noisy.payments[target_firm],
    }

    # DATA_INFLATION: only meaningful for PROPORTIONAL (ZK_SHAPLEY takes no
    # self-reported size input at all -- this row is intentionally absent
    # from the zk_shapley column to make the structural point explicit).
    inflated_sizes = {name: datasets[name].n_train for name in datasets}
    inflated_sizes[target_firm] = int(inflated_sizes[target_firm] * inflation_factor)
    prop_inflated = run_proportional_mechanism(datasets, lam=lam, reported_sizes=inflated_sizes)
    results["data_inflation"] = {
        "zk_shapley_payment": None,  # structurally not applicable -- no size input exists to inflate
        "proportional_payment": prop_inflated.payments[target_firm],
    }

    return results
