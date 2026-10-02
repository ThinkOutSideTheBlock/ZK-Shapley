"""
ZK-Shapley mechanism, baseline allocation rules, and deviation/attack analyses.

This module is the integration point: it wires demand -> federated ->
inventory -> shapley into the two coalitional games compared throughout the
paper (plan Section F.2):

    v_operational(S) : inventory-cost SAVINGS from coalition S's pooled
                        forecast, relative to the no-sharing (autarky)
                        baseline, evaluated via simulate_serial_supply_chain.
    v_accuracy(S)     : forecast-loss REDUCTION from coalition S's pooled
                        ridge model, relative to autarky, evaluated via
                        evaluate_holdout_rmse.

Both value functions share the SAME underlying federated ridge machinery and
differ only in what final scalar they report for a coalition -- exactly the
"only the characteristic function changes" design constraint from the plan.

Individual rationality (plan Section C.4, member-relative form): a firm's
allocation is evaluated against ITS OWN autarky payoff, not zero, since a
firm's outside option is operating alone (possibly with a nonzero baseline
inventory cost), not "earning nothing." `apply_participation_costs` and the
per-mechanism IR check enforce this member-relative notion explicitly.
"""
from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .demand import DemandSimulationResult
from .federated import FirmDataset, fit_and_evaluate_coalition, aggregate_coalition_statistics, fit_ridge_from_statistics
from .inventory import EchelonConfig, simulate_serial_supply_chain
from .shapley import CachedGame, exact_shapley, monte_carlo_shapley


@dataclass
class MechanismResult:
    """
    Full output of running an allocation mechanism over a fixed set of firms:
    the per-firm payment/allocation vector, each firm's autarky (standalone)
    baseline value, member-relative individual-rationality flags, and the
    grand-coalition value the payments must sum to (efficiency check target).
    """
    payments: dict[str, float]
    autarky_baseline: dict[str, float]
    grand_coalition_value: float
    ir_satisfied: dict[str, bool]
    mechanism_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def all_ir_satisfied(self) -> bool:
        return all(self.ir_satisfied.values())

    @property
    def efficiency_gap(self) -> float:
        return abs(sum(self.payments.values()) - self.grand_coalition_value)


# ---------------------------------------------------------------------------
# Value function factories
# ---------------------------------------------------------------------------

def _stable_seed(*parts) -> int:
    """
    Process-stable seed derivation.

    Python randomizes str.__hash__ per process unless PYTHONHASHSEED is fixed,
    so seeding an RNG off hash(name) or hash((coalition, name)) silently
    produces a DIFFERENT stream on every invocation, defeating the common-
    random-numbers reproducibility required by plan Section F.1. SHA-256 is
    stable across processes, platforms, and Python versions.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (frozenset, set)):
            h.update(b"|".join(sorted(str(s).encode() for s in part)))
        else:
            h.update(str(part).encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], byteorder="big")


def make_operational_value_function(
    datasets: dict[str, FirmDataset],
    demand_paths: dict[str, np.ndarray],
    echelons: list[EchelonConfig],
    lam: float = 1.0,
    rng_seed: int = 0,
) -> Callable[[frozenset[str]], float]:
    """
    v_operational(S) = autarky_total_cost(S's members) - coalition_total_cost(S),
    i.e. inventory-cost SAVINGS relative to each member operating alone,
    summed over S's members and evaluated with COMMON RANDOM NUMBERS (a fixed
    rng seed per coalition-member-set, per plan Section F.1) so that
    differences across coalitions reflect forecast-quality differences only.

    For each member i in S, i's demand is driven by the coalition-S ridge
    forecast (if |S| > 0) versus i's own autarky ridge forecast, both fed
    through simulate_serial_supply_chain over member i's holdout demand path.
    v(emptyset) = 0 by convention (no coalition, no simulated system).
    """
    firm_names = list(datasets.keys())
    _autarky_cache: dict[str, float] = {}

    def _autarky_cost_for_member(name: str) -> float:
        # Memoized: without this, each firm's autarky simulation is recomputed
        # once per coalition containing it -- roughly n * 2^(n-1) redundant
        # multi-echelon simulations across a full exact-Shapley sweep.
        if name in _autarky_cache:
            return _autarky_cache[name]
        rng = np.random.default_rng([rng_seed, _stable_seed("autarky", name)])
        ds = datasets[name]
        if ds.X_train.shape[0] == 0 or ds.X_holdout.shape[0] == 0:
            _autarky_cache[name] = 0.0
            return 0.0
        A, b = aggregate_coalition_statistics(datasets, frozenset({name}))
        w = fit_ridge_from_statistics(A, b, lam=lam)
        forecast_path = ds.X_holdout @ w
        residual_std = float(np.std(ds.y_train - ds.X_train @ w)
                             ) if ds.X_train.shape[0] > 1 else 1.0
        demand_path = demand_paths[name][-ds.X_holdout.shape[0]:]
        out = simulate_serial_supply_chain(
            demand_path, forecast_path, residual_std, echelons, rng=rng)
        _autarky_cache[name] = out.total_cost
        return out.total_cost

    def v(coalition: frozenset[str]) -> float:
        if len(coalition) == 0:
            return 0.0
        A_S, b_S = aggregate_coalition_statistics(datasets, coalition)
        w_S = fit_ridge_from_statistics(A_S, b_S, lam=lam)

        total_savings = 0.0
        for name in coalition:
            ds = datasets[name]
            if ds.X_holdout.shape[0] == 0:
                continue
            rng = np.random.default_rng(
                [rng_seed, _stable_seed(coalition, name)])
            forecast_path = ds.X_holdout @ w_S
            residual_std = float(
                np.std(ds.y_train - ds.X_train @ w_S)) if ds.X_train.shape[0] > 0 else 1.0
            demand_path = demand_paths[name][-ds.X_holdout.shape[0]:]
            out = simulate_serial_supply_chain(
                demand_path, forecast_path, residual_std, echelons, rng=rng)
            autarky_cost = _autarky_cost_for_member(name)
            total_savings += (autarky_cost - out.total_cost)
        return total_savings

    return v


def make_accuracy_value_function(
    datasets: dict[str, FirmDataset], lam: float = 1.0
) -> Callable[[frozenset[str]], float]:
    """
    v_accuracy(S) = sum_{i in S} [autarky_RMSE_i - coalition_RMSE(S on i)],
    where autarky_RMSE_i is the firm’s own singleton ridge model.
    This guarantees v({i}) = 0 by construction (plan Section F.2 / model_spec).
    """
    def v(coalition: frozenset[str]) -> float:
        if len(coalition) == 0:
            return 0.0
        total_reduction = 0.0
        for name in coalition:
            # Singleton baseline (own model) — NOT the empty-coalition mean
            autarky_rmse = fit_and_evaluate_coalition(
                datasets, frozenset({name}), frozenset({name}), lam=lam)
            coalition_rmse = fit_and_evaluate_coalition(
                datasets, coalition, frozenset({name}), lam=lam)
            if np.isnan(autarky_rmse) or np.isnan(coalition_rmse):
                continue
            total_reduction += (autarky_rmse - coalition_rmse)
        return total_reduction
    return v


# ---------------------------------------------------------------------------
# Individual rationality (member-relative)
# ---------------------------------------------------------------------------

def _member_relative_ir_check(
    payments: dict[str, float], autarky_baseline: dict[str, float], tol: float = 1e-9
) -> dict[str, bool]:
    """
    Member-relative IR: firm i's allocation is compared to i's own autarky
    VALUE-FUNCTION baseline (i.e. v({i}), which by construction of both
    v_operational and v_accuracy above is 0 -- since both are already defined
    as savings/reduction RELATIVE to autarky). IR here means payments[i] >= 0,
    i.e. no firm is asked to pay to join relative to its own outside option of
    operating alone with cost/RMSE = autarky_baseline[i].
    """
    return {name: (payments.get(name, 0.0) >= autarky_baseline.get(name, 0.0) - tol) for name in payments}


def apply_participation_costs(
    payments: dict[str, float], participation_costs: dict[str, float]
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Net a per-firm participation cost (e.g. commitment-protocol overhead,
    computational cost of producing sufficient statistics) against gross
    Shapley payments, and re-check member-relative IR against the NET payoff
    (net_i = payments[i] - participation_costs[i] >= 0). This operationalizes
    the plan's requirement that IR be checked net of any real cost the
    mechanism itself imposes on participants, not just gross of it.
    """
    net_payments = {
        name: payments[name] - participation_costs.get(name, 0.0) for name in payments}
    ir_net = {name: net_payments[name] >= -1e-9 for name in net_payments}
    return net_payments, ir_net


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

def run_zk_shapley_mechanism(
    firm_names: list[str],
    v_func: Callable[[frozenset[str]], float],
    exact: bool = True,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> MechanismResult:
    """
    The paper's core mechanism: allocate v(N) via Shapley value of the
    coalitional game (v_func, firm_names). Uses exact enumeration for
    n <= 12 firms (tractable, per plan Section F.1), Monte Carlo permutation
    sampling otherwise.
    """
    game = CachedGame(players=firm_names, v_func=v_func)
    if exact and len(firm_names) <= 12:
        phi = exact_shapley(game)
    else:
        phi = monte_carlo_shapley(game, n_permutations=n_permutations, rng=rng)

    autarky_baseline = {name: game.v(frozenset({name})) for name in firm_names}
    v_grand = game.v(frozenset(firm_names))
    ir = _member_relative_ir_check(phi, autarky_baseline)

    return MechanismResult(
        payments=phi, autarky_baseline=autarky_baseline, grand_coalition_value=v_grand,
        ir_satisfied=ir, mechanism_name="ZK-Shapley (exact)" if exact and len(firm_names) <= 12
        else "ZK-Shapley (Monte Carlo)",
        metadata={"n_permutations": n_permutations if not (
            exact and len(firm_names) <= 12) else None},
    )


def run_accuracy_shapley_mechanism(
    firm_names: list[str], v_accuracy_func: Callable[[frozenset[str]], float],
    exact: bool = True, n_permutations: int = 2000, rng: np.random.Generator | None = None,
) -> MechanismResult:
    """Identical Shapley machinery applied to the accuracy game -- the head-to-head comparator."""
    result = run_zk_shapley_mechanism(
        firm_names, v_accuracy_func, exact=exact, n_permutations=n_permutations, rng=rng)
    result.mechanism_name = "Federated Shapley Value (accuracy-based)" + \
        result.mechanism_name.split("ZK-Shapley")[-1]
    return result


def run_equal_split_mechanism(
    firm_names: list[str], v_func: Callable[[frozenset[str]], float]
) -> MechanismResult:
    """Baseline: split v(N) equally regardless of contribution."""
    game = CachedGame(players=firm_names, v_func=v_func)
    v_grand = game.v(frozenset(firm_names))
    n = len(firm_names)
    payments = {name: v_grand / n for name in firm_names}
    autarky_baseline = {name: game.v(frozenset({name})) for name in firm_names}
    ir = _member_relative_ir_check(payments, autarky_baseline)
    return MechanismResult(payments=payments, autarky_baseline=autarky_baseline, grand_coalition_value=v_grand,
                           ir_satisfied=ir, mechanism_name="Equal Split")


def run_proportional_mechanism(
    firm_names: list[str], v_func: Callable[[frozenset[str]], float], size_proxy: dict[str, float]
) -> MechanismResult:
    """
    Baseline: split v(N) proportionally to a size proxy (e.g. declared sample
    count) -- the contrasting mechanism used alongside sample_size_baseline_payment
    in theory.py to make Proposition A's content legible.
    """
    game = CachedGame(players=firm_names, v_func=v_func)
    v_grand = game.v(frozenset(firm_names))
    total = sum(size_proxy[name] for name in firm_names)
    payments = {name: v_grand * size_proxy[name] / total for name in firm_names} if total > 0 else \
        {name: v_grand / len(firm_names) for name in firm_names}
    autarky_baseline = {name: game.v(frozenset({name})) for name in firm_names}
    ir = _member_relative_ir_check(payments, autarky_baseline)
    return MechanismResult(payments=payments, autarky_baseline=autarky_baseline, grand_coalition_value=v_grand,
                           ir_satisfied=ir, mechanism_name="Proportional (size-weighted)")


def run_no_sharing_baseline(firm_names: list[str], v_func: Callable[[frozenset[str]], float]) -> MechanismResult:
    """
    Baseline: no coalition forms at all -- every firm gets its own autarky
    value (0 by construction of v_operational/v_accuracy's savings/reduction
    definition). Serves as the "genuine collaboration gains exist" check:
    if v(N) via any real mechanism is not reliably > 0, pooling is not adding
    value and the whole exercise is moot (plan Section F.1 sanity check).
    """
    game = CachedGame(players=firm_names, v_func=v_func)
    payments = {name: game.v(frozenset({name})) for name in firm_names}
    autarky_baseline = dict(payments)
    v_grand = game.v(frozenset(firm_names))
    ir = _member_relative_ir_check(payments, autarky_baseline)
    return MechanismResult(payments=payments, autarky_baseline=autarky_baseline, grand_coalition_value=v_grand,
                           ir_satisfied=ir, mechanism_name="No Sharing (autarky)")


# ---------------------------------------------------------------------------
# Deviation / attack analyses
# ---------------------------------------------------------------------------

def scale_statistics_attack(
    A_i: np.ndarray, b_i: np.ndarray, scale_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    A genuine data-fabrication attack: a firm submits (alpha * A_i, alpha *
    b_i) for some alpha != 1, i.e. fabricated sufficient statistics
    inconsistent with any real dataset the firm actually holds (since real
    A_i, b_i are rank-tied to actual observation counts and cannot generally
    be rescaled this way while remaining realizable by any (X, y)). THIS
    ATTACK IS EXPECTED, AND REQUIRED BY THE PLAN, TO CHANGE PAYMENTS -- it is
    fundamentally different from, and must not be conflated with, the
    standalone-declared-size invariance tested by Proposition A / plan Section
    F.4. Use this function together with mechanism-level payment computation
    to empirically characterize HOW payments move under fabrication, which
    then motivates why a future proof layer (costed but not built,
    proof_cost.py) would need to constrain (A_i, b_i) to be relation-
    consistent with a committed raw dataset.
    """
    return scale_factor * A_i, scale_factor * b_i


def inject_noise_dataset(
    y: np.ndarray, sigma_eta: float, rng: np.random.Generator
) -> np.ndarray:
    """Contaminate a response vector with independent, mean-zero Gaussian noise (the empirical counterpart to theory.noise_injection_risk_bound)."""
    return y + rng.normal(0.0, sigma_eta, size=y.shape[0])


def deviation_payoff_analysis(
    firm_names: list[str],
    v_func_factory: Callable[[dict[str, FirmDataset]], Callable[[frozenset[str]], float]],
    datasets: dict[str, FirmDataset],
    deviator: str,
    sigma_eta_grid: list[float],
    n_mc_reps: int = 20,
    seed: int = 0,
) -> dict:
    """
    Empirical Monte Carlo sweep: for each sigma_eta in sigma_eta_grid,
    contaminate `deviator`'s TRAINING response vector with independent noise
    of that scale (holding all other firms' data fixed and truthful),
    recompute the deviator's ZK-Shapley payment under the resulting
    coalitional game, and average over n_mc_reps independent noise draws.
    Reports the payment trajectory in sigma_eta and whether it is (on
    average, empirically) non-increasing -- the empirical counterpart to
    theory.noise_injection_risk_bound's analytic guarantee, per plan Section
    F.5's requirement to pair the analytic bound with an empirical sweep
    rather than resting the claim on the analytic result alone.
    """
    rng = np.random.default_rng(seed)
    mean_payments = []
    std_payments = []

    for sigma_eta in sigma_eta_grid:
        rep_payments = []
        for _ in range(n_mc_reps):
            contaminated = dict(datasets)
            if sigma_eta > 0:
                ds = datasets[deviator]
                noisy_y_train = inject_noise_dataset(
                    ds.y_train, sigma_eta, rng)
                contaminated[deviator] = FirmDataset(
                    name=ds.name, X_train=ds.X_train, y_train=noisy_y_train,
                    X_val=ds.X_val, y_val=ds.y_val, X_holdout=ds.X_holdout, y_holdout=ds.y_holdout,
                    gamma=ds.gamma, sigma=ds.sigma,
                )
            v_func = v_func_factory(contaminated)
            result = run_zk_shapley_mechanism(
                firm_names, v_func, exact=(len(firm_names) <= 12))
            rep_payments.append(result.payments[deviator])
        mean_payments.append(float(np.mean(rep_payments)))
        std_payments.append(float(np.std(rep_payments)))

    diffs = np.diff(mean_payments)
    empirically_non_increasing = bool(np.all(diffs <= 1e-6))

    return {
        "sigma_eta_grid": sigma_eta_grid,
        "mean_payments": mean_payments,
        "std_payments": std_payments,
        "empirically_non_increasing": empirically_non_increasing,
        "n_mc_reps": n_mc_reps,
        "deviator": deviator,
    }


def compute_divergence_metrics(
    payments_operational: dict[str, float], payments_accuracy: dict[str, float]
) -> dict:
    """
    Quantify how much the operational (inventory-cost) and accuracy-based
    Shapley allocations diverge for the head-to-head comparison (plan Section
    F.2): rank-order (Spearman-style via simple rank comparison) agreement,
    max absolute normalized payment difference, and per-firm rank shifts.
    """
    firms = sorted(payments_operational.keys())
    op_vals = np.array([payments_operational[f] for f in firms])
    acc_vals = np.array([payments_accuracy[f] for f in firms])

    op_ranks = op_vals.argsort().argsort()
    acc_ranks = acc_vals.argsort().argsort()
    rank_shifts = {f: int(op_ranks[i] - acc_ranks[i])
                   for i, f in enumerate(firms)}

    n = len(firms)
    if n > 1:
        d_sq = np.sum((op_ranks - acc_ranks) ** 2)
        spearman_rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
    else:
        spearman_rho = float("nan")

    op_total = np.sum(np.abs(op_vals)) + 1e-12
    acc_total = np.sum(np.abs(acc_vals)) + 1e-12
    normalized_diff = np.abs(op_vals / op_total - acc_vals / acc_total)

    return {
        "firms": firms,
        "spearman_rho": float(spearman_rho),
        "rank_shifts": rank_shifts,
        "max_normalized_payment_difference": float(np.max(normalized_diff)),
        "mean_normalized_payment_difference": float(np.mean(normalized_diff)),
        "any_rank_reversal": any(v != 0 for v in rank_shifts.values()),
    }
