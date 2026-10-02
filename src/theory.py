"""
Closed-form analytic results underpinning ZK-Shapley's strategic-robustness
claims (plan Sections D and F.5).

Two, and only two, analytic results are established here:

  1. Structural (definitional) invariance to a standalone, unsupported
     sample-count declaration (Proposition A). This is a STRUCTURAL claim
     about the mechanism's message space, not a statistical result -- it
     holds because no function in the payment pipeline ever reads a
     standalone n_i argument. `size_declaration_is_unused` operationalizes
     this by construction-check rather than by simulation.

  2. A partial analytic bound on the additional expected quadratic prediction
     risk induced by independent, mean-zero response contamination
     (plan Section F.5):

         Delta R_S = sigma_eta^2 * tr(Sigma_x M_S^{-1} A_i M_S^{-1}) >= 0

     This is an upper-level, closed-form validation target for the empirical
     noise-injection sweep in mechanism.py -- it establishes an INCREASE IN
     EXPECTED PREDICTION RISK under stated assumptions, and explicitly does
     NOT establish that inventory cost or any Shapley payment decreases in
     every realization (plan Section H.6).

Both the size-inflation baseline (`sample_size_baseline_payment`) used as the
CONTRASTING mechanism in plan Section D, and its required monotonicity check,
live here as well, since they exist specifically to make Proposition A's
content legible by comparison.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Proposition A: structural invariance to standalone sample-count declarations
# ---------------------------------------------------------------------------

def size_declaration_is_unused(
    payment_function,
    accepted_statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    declared_sizes_a: dict[str, float],
    declared_sizes_b: dict[str, float],
    tol: float = 1e-10,
) -> dict:
    """
    Operationalizes Proposition A: verifies, BY CONSTRUCTION rather than by
    inspecting arbitrary code, that a payment function which accepts a
    `declared_sizes` argument alongside accepted sufficient statistics
    produces IDENTICAL output when only the declared sizes change and the
    accepted (A_i, b_i) statistics are held fixed.

    This is the correct operationalization of Proposition A -- NOT a scaling
    attack on (A_i, b_i) itself (that is a distinct, second experiment; see
    `mechanism.scale_statistics_attack`, which is expected and REQUIRED to
    change payments, per plan Section F.4).

    Parameters
    ----------
    payment_function : callable(accepted_statistics, declared_sizes) -> dict[str, float]
        Any payment rule under test. To satisfy Proposition A it must not
        actually use declared_sizes in computing coalition values; this
        function empirically confirms that non-use for the specific rule
        passed in, it does not statically analyze the rule's source code.
    accepted_statistics : the (A_i, b_i) pairs actually accepted after
        commitment verification -- held FIXED across both calls.
    declared_sizes_a, declared_sizes_b : two different standalone sample-count
        declarations to compare.

    Returns
    -------
    dict with the two payment vectors, their max absolute difference, and a
    boolean `invariant` flag (True iff max difference <= tol).
    """
    payments_a = payment_function(accepted_statistics, declared_sizes_a)
    payments_b = payment_function(accepted_statistics, declared_sizes_b)

    common_keys = set(payments_a.keys()) & set(payments_b.keys())
    max_diff = max((abs(payments_a[k] - payments_b[k])
                   for k in common_keys), default=0.0)

    return {
        "payments_a": payments_a,
        "payments_b": payments_b,
        "max_abs_difference": max_diff,
        "invariant": max_diff <= tol,
        "declared_sizes_a": declared_sizes_a,
        "declared_sizes_b": declared_sizes_b,
    }


# ---------------------------------------------------------------------------
# Sample-size proportional baseline (contrast mechanism for Proposition A)
# ---------------------------------------------------------------------------

def sample_size_baseline_payment(
    v_grand: float, declared_sizes: dict[str, float]
) -> dict[str, float]:
    """
    p_i^size = v(N) * n_tilde_i / sum_j n_tilde_j  (plan Section D "sample-size
    baseline"). This rule DOES read a standalone declared-size argument, in
    direct contrast to ZK-Shapley -- it exists specifically to demonstrate
    what Proposition A rules out.
    """
    total = sum(declared_sizes.values())
    if total <= 0:
        raise ValueError("Sum of declared sizes must be positive.")
    return {name: v_grand * n_tilde / total for name, n_tilde in declared_sizes.items()}


def sample_size_baseline_monotonicity_check(
    v_grand: float, true_sizes: dict[str, float], deviator: str, alphas: list[float]
) -> dict:
    """
    Verifies the plan's Section D closed-form monotonicity claim: under
    unilateral inflation n_tilde_i' = alpha * n_i for the deviating firm
    (others truthful), p_i^size(alpha) is strictly increasing in alpha
    whenever v(N) > 0 and at least one other firm has positive size.
    Returns the payment trajectory and a boolean strict-monotonicity flag.
    """
    if v_grand <= 0:
        raise ValueError("Monotonicity claim requires v(N) > 0.")
    others_total = sum(v for name, v in true_sizes.items() if name != deviator)
    if others_total <= 0:
        raise ValueError(
            "At least one other firm must have positive sample size.")

    n_i = true_sizes[deviator]
    if n_i <= 0:
        raise ValueError(
            "Monotonicity claim requires the deviator's true sample size to be positive; "
            "with n_i = 0 every inflated declaration alpha * n_i remains 0 and the payment "
            "is constant in alpha."
        )

    payments = []
    for alpha in alphas:
        declared = dict(true_sizes)
        declared[deviator] = alpha * n_i
        p = sample_size_baseline_payment(v_grand, declared)
        payments.append(p[deviator])

    strictly_increasing = all(payments[k] < payments[k + 1]
                              for k in range(len(payments) - 1))
    closed_form = [v_grand * (alpha * n_i) /
                   (alpha * n_i + others_total) for alpha in alphas]

    return {
        "alphas": alphas,
        "payments": payments,
        "closed_form_payments": closed_form,
        "max_abs_gap_to_closed_form": float(np.max(np.abs(np.array(payments) - np.array(closed_form)))),
        "strictly_increasing": strictly_increasing,
    }


# ---------------------------------------------------------------------------
# Noise-injection partial analytic risk bound (plan Section F.5)
# ---------------------------------------------------------------------------

def noise_injection_risk_bound(
    sigma_eta: float, Sigma_x: np.ndarray, M_S: np.ndarray, A_i: np.ndarray
) -> float:
    """
    Delta R_S = sigma_eta^2 * tr(Sigma_x @ M_S^{-1} @ A_i @ M_S^{-1})

    the additional expected quadratic prediction risk incurred by coalition S
    when member i's response vector y_i is contaminated by independent, mean-zero noise of variance
    sigma_eta^2 (i.e. y_i_tilde = y_i + eta, eta ~ (0, sigma_eta^2 I),
    independent of X_i and of all other members' data), where

        M_S = A_S + Lambda    (the ridge coalition Gram matrix used to fit
                                beta_hat_S, with A_i one member's contribution
                                to A_S)

    Sigma_x : population second-moment matrix of the covariates the
        prediction risk is evaluated under (e.g. the empirical second-moment
        matrix of the holdout covariates, as a plug-in estimate).

    Derivation sketch (stated, not re-derived numerically here): under the
    stated contamination model, beta_hat_S_tilde - beta_hat_S = M_S^{-1} X_i^T
    eta, a mean-zero random vector with covariance sigma_eta^2 * M_S^{-1} A_i
    M_S^{-1} (since E[X_i^T eta eta^T X_i] = sigma_eta^2 * X_i^T X_i =
    sigma_eta^2 * A_i). The induced increase in expected quadratic prediction
    risk E[(x'(beta_hat_S_tilde - beta_hat_S))^2] under x ~ Sigma_x is the
    trace formula above (standard quadratic-form expectation identity:
    E[u' C u] = tr(C * Cov(u)) for mean-zero u).

    This function returns Delta R_S >= 0 ALWAYS (it is a trace of a product of
    PSD matrices under the stated assumptions), which is exactly the content
    of the partial analytic result: contamination provably cannot DECREASE
    expected quadratic prediction risk under independent, mean-zero noise.
    It does NOT claim that realized inventory cost or any single Shapley
    payment must decrease in every draw or every replication (plan Section
    H.6) -- that is an empirical question addressed by the Monte Carlo sweep
    in mechanism.inject_noise_dataset / deviation_payoff_analysis.
    """
    d = M_S.shape[0]
    if A_i.shape != (d, d) or Sigma_x.shape != (d, d):
        raise ValueError(
            "Sigma_x, M_S, and A_i must all be d x d with matching d.")

    M_S_inv = np.linalg.inv(M_S)
    cov_term = M_S_inv @ A_i @ M_S_inv
    delta_R = (sigma_eta ** 2) * float(np.trace(Sigma_x @ cov_term))

    # Numerical guard: trace of a product of PSD matrices must be >= 0 up to
    # floating-point error; clip tiny negative noise rather than raise, since
    # Sigma_x is typically an empirical (finite-sample) plug-in estimate.
    return max(delta_R, 0.0)
