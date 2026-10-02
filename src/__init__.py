"""
ZK-Shapley: Operational Surplus Allocation for Privacy-Aware Federated
Supply-Chain Forecasting -- reference implementation.

Companion codebase for "Valuing Federated Forecasts by Inventory Savings:
The ZK-Shapley Mechanism for Competing Firms."

Module map
----------
demand         : synthetic multi-firm demand-generating process
federated      : exact federated ridge aggregation via sufficient statistics
inventory      : multi-echelon base-stock simulator (operational value channel)
shapley        : exact / Monte Carlo Shapley engine + axiom verification
commitment     : hash commit-reveal protocol (anti-equivocation only, see H.2)
proof_cost     : relation-specific zk cost model and micro-benchmarks (Section E)
theory         : closed-form analytic results (Proposition A, noise-injection risk)
mechanism      : ZK-Shapley mechanism, baselines, and deviation/attack analyses

Scope-of-guarantees reminder (see plan Section H): this codebase does NOT
implement or claim (a) dominant-strategy truthfulness, (b) a deployed
zero-knowledge correctness proof for sufficient-statistic computation, or
(c) confidentiality of released sufficient statistics themselves. See each
module's docstring for the precise guarantee it does establish.
"""
from .demand import (
    DemandSimulationConfig,
    DemandSimulationResult,
    FirmRawSeries,
    generate_multi_firm_demand,
)
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
from .inventory import (
    EchelonConfig,
    SimulationOutput,
    simulate_serial_supply_chain,
    default_three_echelon_config,
)
from .shapley import (
    CachedGame,
    exact_shapley,
    monte_carlo_shapley,
    check_efficiency,
    check_null_player,
    check_symmetry,
)
from .commitment import (
    Commitment,
    commit_statistics,
    verify_opening,
    CommitmentError,
)
from .proof_cost import (
    FixedPointSpec,
    quantize_statistics,
    dequantize_ridge_solution,
    quantization_error_report,
    relation_multiplication_count,
    IPAProof,
    InnerProductArgument,
    run_mb1_benchmark,
    extrapolate_mb2,
    extrapolate_mb3,
    fit_scaling_model,
    predict_with_interval,
)
from .theory import (
    noise_injection_risk_bound,
    size_declaration_is_unused,
    sample_size_baseline_payment,
    sample_size_baseline_monotonicity_check,
)
from .mechanism import (
    MechanismResult,
    make_operational_value_function,
    make_accuracy_value_function,
    run_zk_shapley_mechanism,
    run_equal_split_mechanism,
    run_proportional_mechanism,
    run_no_sharing_baseline,
    run_accuracy_shapley_mechanism,
    apply_participation_costs,
    scale_statistics_attack,
    inject_noise_dataset,
    deviation_payoff_analysis,
    compute_divergence_metrics,
)

__all__ = [
    "DemandSimulationConfig", "DemandSimulationResult", "FirmRawSeries", "generate_multi_firm_demand",
    "FirmDataset", "N_FEATURES", "build_firm_datasets", "aggregate_coalition_statistics",
    "fit_ridge_from_statistics", "evaluate_holdout_rmse", "fit_and_evaluate_coalition", "verify_exact_aggregation",
    "EchelonConfig", "SimulationOutput", "simulate_serial_supply_chain", "default_three_echelon_config",
    "CachedGame", "exact_shapley", "monte_carlo_shapley", "check_efficiency", "check_null_player", "check_symmetry",
    "Commitment", "commit_statistics", "verify_opening", "CommitmentError",
    "FixedPointSpec", "quantize_statistics", "dequantize_ridge_solution", "quantization_error_report",
    "relation_multiplication_count", "IPAProof", "InnerProductArgument", "run_mb1_benchmark",
    "extrapolate_mb2", "extrapolate_mb3", "fit_scaling_model", "predict_with_interval",
    "noise_injection_risk_bound", "size_declaration_is_unused", "sample_size_baseline_payment",
    "sample_size_baseline_monotonicity_check",
    "MechanismResult", "make_operational_value_function", "make_accuracy_value_function",
    "run_zk_shapley_mechanism", "run_equal_split_mechanism", "run_proportional_mechanism",
    "run_no_sharing_baseline", "run_accuracy_shapley_mechanism", "apply_participation_costs",
    "scale_statistics_attack", "inject_noise_dataset", "deviation_payoff_analysis", "compute_divergence_metrics",
]
