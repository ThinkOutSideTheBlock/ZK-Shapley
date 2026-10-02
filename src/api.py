"""
ZK-Shapley public API surface.

Experiments, notebooks, and external callers MUST import only from this
module (or from the top-level package re-exports). Direct imports of
internal helpers (e.g. private caches, encoding utilities) are forbidden
and will be caught by tests/test_api_contract.py.

This thin layer freezes the signatures that the rest of the research
instrument depends on.
"""
from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# Demand generation
# ---------------------------------------------------------------------------
from .demand import (
    DemandSimulationConfig,
    DemandSimulationResult,
    FirmRawSeries,
    generate_multi_firm_demand,
)

# ---------------------------------------------------------------------------
# Federated ridge
# ---------------------------------------------------------------------------
from .federated import (
    FirmDataset,
    N_FEATURES,
    build_firm_datasets,
    aggregate_coalition_statistics,
    fit_ridge_from_statistics,
    evaluate_holdout_rmse,
    fit_and_evaluate_coalition,
    verify_exact_aggregation,
)

# ---------------------------------------------------------------------------
# Inventory simulator
# ---------------------------------------------------------------------------
from .inventory import (
    EchelonConfig,
    SimulationOutput,
    simulate_serial_supply_chain,
    default_three_echelon_config,
)

# ---------------------------------------------------------------------------
# Shapley engine
# ---------------------------------------------------------------------------
from .shapley import (
    CachedGame,
    exact_shapley,
    monte_carlo_shapley,
    check_efficiency,
    check_null_player,
    check_symmetry,
)

# ---------------------------------------------------------------------------
# Commitment (anti-equivocation only)
# ---------------------------------------------------------------------------
from .commitment import (
    Commitment,
    commit_statistics,
    verify_opening,
    verify_opening_or_raise,
    CommitmentError,
)

# ---------------------------------------------------------------------------
# Proof-cost model
# ---------------------------------------------------------------------------
from .proof_cost import (
    FixedPointSpec,
    quantize_statistics,
    dequantize_ridge_solution,
    quantization_error_report,
    relation_multiplication_count,
    run_mb1_benchmark,
    extrapolate_mb2,
    extrapolate_mb3,
    fit_scaling_model,
    predict_with_interval,
)

# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------
from .theory import (
    noise_injection_risk_bound,
    size_declaration_is_unused,
    sample_size_baseline_payment,
    sample_size_baseline_monotonicity_check,
)

# ---------------------------------------------------------------------------
# Mechanism (core public entry points)
# ---------------------------------------------------------------------------
from .mechanism import (
    MechanismResult,
    make_operational_value_function,
    make_accuracy_value_function,
    run_zk_shapley_mechanism,
    run_accuracy_shapley_mechanism,
    run_equal_split_mechanism,
    run_proportional_mechanism,
    run_no_sharing_baseline,
    apply_participation_costs,
    scale_statistics_attack,
    inject_noise_dataset,
    deviation_payoff_analysis,
    compute_divergence_metrics,
)

__all__ = [
    # demand
    "DemandSimulationConfig",
    "DemandSimulationResult",
    "FirmRawSeries",
    "generate_multi_firm_demand",
    # federated
    "FirmDataset",
    "N_FEATURES",
    "build_firm_datasets",
    "aggregate_coalition_statistics",
    "fit_ridge_from_statistics",
    "evaluate_holdout_rmse",
    "fit_and_evaluate_coalition",
    "verify_exact_aggregation",
    # inventory
    "EchelonConfig",
    "SimulationOutput",
    "simulate_serial_supply_chain",
    "default_three_echelon_config",
    # shapley
    "CachedGame",
    "exact_shapley",
    "monte_carlo_shapley",
    "check_efficiency",
    "check_null_player",
    "check_symmetry",
    # commitment
    "Commitment",
    "commit_statistics",
    "verify_opening",
    "verify_opening_or_raise",
    "CommitmentError",
    # proof_cost
    "FixedPointSpec",
    "quantize_statistics",
    "dequantize_ridge_solution",
    "quantization_error_report",
    "relation_multiplication_count",
    "run_mb1_benchmark",
    "extrapolate_mb2",
    "extrapolate_mb3",
    "fit_scaling_model",
    "predict_with_interval",
    # theory
    "noise_injection_risk_bound",
    "size_declaration_is_unused",
    "sample_size_baseline_payment",
    "sample_size_baseline_monotonicity_check",
    # mechanism
    "MechanismResult",
    "make_operational_value_function",
    "make_accuracy_value_function",
    "run_zk_shapley_mechanism",
    "run_accuracy_shapley_mechanism",
    "run_equal_split_mechanism",
    "run_proportional_mechanism",
    "run_no_sharing_baseline",
    "apply_participation_costs",
    "scale_statistics_attack",
    "inject_noise_dataset",
    "deviation_payoff_analysis",
    "compute_divergence_metrics",
]
