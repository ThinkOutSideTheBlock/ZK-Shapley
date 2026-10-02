# tests/test_theory.py  (REPLACES previous draft — signatures were wrong)
"""
Tests for src/theory.py, against the ACTUAL signatures in the file provided:

    size_declaration_is_unused(payment_function, accepted_statistics,
                                declared_sizes_a, declared_sizes_b, tol=1e-10) -> dict
    sample_size_baseline_payment(v_grand: float, declared_sizes: dict) -> dict
    sample_size_baseline_monotonicity_check(v_grand, true_sizes, deviator, alphas) -> dict
    noise_injection_risk_bound(sigma_eta, Sigma_x, M_S, A_i) -> float   (bare float, not a dict!)

KNOWN GAP flagged, not hidden: noise_injection_risk_bound performs no symmetry
check on Sigma_x/M_S and takes no separate lam argument to validate (Lambda is
already folded into M_S by the caller). test_noise_injection_risk_bound_no_validation_is_a_gap
documents this as current behavior with an explanatory comment, rather than
asserting a ValueError that the implementation does not raise. This should be
treated as an open item for the paper's assumptions section (Sigma_x, M_S are
assumed, not verified, to be symmetric PSD), not silently passed over.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.theory import (
    size_declaration_is_unused,
    sample_size_baseline_payment,
    sample_size_baseline_monotonicity_check,
    noise_injection_risk_bound,
)


# ---------------------------------------------------------------------------
# Proposition A
# ---------------------------------------------------------------------------

def _size_blind_payment(accepted_statistics, declared_sizes):
    """Genuinely ignores declared_sizes -- splits equally over accepted_statistics keys."""
    names = list(accepted_statistics.keys())
    return {name: 1.0 / len(names) for name in names}


def _size_dependent_payment_adapter(v_grand: float):
    """
    Wraps sample_size_baseline_payment (which takes v_grand, declared_sizes)
    into the (accepted_statistics, declared_sizes) -> dict shape that
    size_declaration_is_unused expects, so we can use it as the CONTRASTING,
    genuinely size-dependent rule.
    """
    def payment_function(accepted_statistics, declared_sizes):
        return sample_size_baseline_payment(v_grand, declared_sizes)
    return payment_function


def test_size_declaration_is_unused_true_for_size_blind_rule():
    accepted = {"f1": (np.eye(2), np.ones(2)), "f2": (np.eye(2), np.ones(2))}
    result = size_declaration_is_unused(
        _size_blind_payment, accepted,
        declared_sizes_a={"f1": 10, "f2": 10},
        declared_sizes_b={"f1": 1000, "f2": 1},
    )
    assert result["invariant"] is True
    assert result["max_abs_difference"] < 1e-12


def test_size_declaration_is_unused_false_for_size_dependent_rule():
    """
    Critical negative test: without this, a checker that always reports
    invariant=True would pass the test above and go undetected as broken.
    sample_size_baseline_payment DOES depend on declared sizes by
    construction -- that's the whole point of contrasting it against
    ZK-Shapley for Proposition A.
    """
    accepted = {"f1": (np.eye(2), np.ones(2)), "f2": (np.eye(2), np.ones(2))}
    payment_function = _size_dependent_payment_adapter(v_grand=100.0)
    result = size_declaration_is_unused(
        payment_function, accepted,
        declared_sizes_a={"f1": 10, "f2": 10},
        declared_sizes_b={"f1": 1000, "f2": 1},
    )
    assert result["invariant"] is False
    assert result["max_abs_difference"] > 1.0


def test_size_declaration_is_unused_propagates_keyerror_from_incomplete_declared_sizes():
    """
    size_declaration_is_unused itself does not validate key sets (it only
    intersects payment_function's OUTPUT keys); but a size-dependent
    payment_function that indexes declared_sizes by every accepted-statistics
    key will raise KeyError if declared_sizes_b is missing one. This
    confirms the missing-key failure mode surfaces loudly rather than
    silently comparing against a default.
    """
    accepted = {"f1": (np.eye(2), np.ones(2)), "f2": (np.eye(2), np.ones(2))}

    def strict_payment_function(accepted_statistics, declared_sizes):
        total = sum(declared_sizes[name] for name in accepted_statistics)
        return {name: declared_sizes[name] / total for name in accepted_statistics}

    with pytest.raises(KeyError):
        size_declaration_is_unused(
            strict_payment_function, accepted,
            declared_sizes_a={"f1": 10, "f2": 10},
            declared_sizes_b={"f1": 10},  # missing 'f2'
        )


# ---------------------------------------------------------------------------
# sample_size_baseline_payment
# ---------------------------------------------------------------------------

def test_sample_size_baseline_payment_proportional_to_declared_size():
    declared = {"a": 100.0, "b": 300.0}
    p = sample_size_baseline_payment(v_grand=40.0, declared_sizes=declared)
    assert p["a"] == pytest.approx(10.0)
    assert p["b"] == pytest.approx(30.0)


def test_sample_size_baseline_payment_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        sample_size_baseline_payment(
            v_grand=10.0, declared_sizes={"a": 0.0, "b": 0.0})


# ---------------------------------------------------------------------------
# sample_size_baseline_monotonicity_check
# ---------------------------------------------------------------------------

def test_monotonicity_check_strictly_increasing_for_well_posed_instance():
    true_sizes = {"deviator": 40.0, "other1": 30.0, "other2": 30.0}
    result = sample_size_baseline_monotonicity_check(
        v_grand=100.0, true_sizes=true_sizes, deviator="deviator",
        alphas=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    assert result["strictly_increasing"] is True
    assert result["max_abs_gap_to_closed_form"] < 1e-9


def test_monotonicity_check_matches_hand_computed_closed_form():
    true_sizes = {"deviator": 10.0, "other1": 90.0}
    result = sample_size_baseline_monotonicity_check(
        v_grand=100.0, true_sizes=true_sizes, deviator="deviator", alphas=[1.0, 2.0]
    )
    # alpha=1: p = 100 * 10 / (10 + 90) = 10.0
    # alpha=2: p = 100 * 20 / (20 + 90) = 18.1818...
    assert result["payments"][0] == pytest.approx(10.0)
    assert result["payments"][1] == pytest.approx(100.0 * 20.0 / 110.0)


def test_monotonicity_check_rejects_zero_v_grand():
    true_sizes = {"deviator": 10.0, "other1": 90.0}
    with pytest.raises(ValueError):
        sample_size_baseline_monotonicity_check(
            v_grand=0.0, true_sizes=true_sizes, deviator="deviator", alphas=[1.0, 2.0]
        )


def test_monotonicity_check_rejects_zero_deviator_size():
    """
    Regression test for the fixed bug: n_i = 0 makes alpha * n_i = 0 for
    every alpha, so the payment is trivially constant and the monotonicity
    claim is vacuous -- must raise, not silently report "monotone" or
    "not monotone" on a degenerate instance.
    """
    true_sizes = {"deviator": 0.0, "other1": 90.0}
    with pytest.raises(ValueError):
        sample_size_baseline_monotonicity_check(
            v_grand=100.0, true_sizes=true_sizes, deviator="deviator", alphas=[1.0, 2.0]
        )


def test_monotonicity_check_rejects_zero_others_total():
    true_sizes = {"deviator": 40.0, "other1": 0.0}
    with pytest.raises(ValueError):
        sample_size_baseline_monotonicity_check(
            v_grand=100.0, true_sizes=true_sizes, deviator="deviator", alphas=[1.0, 2.0]
        )


# ---------------------------------------------------------------------------
# noise_injection_risk_bound: hand-computable diagonal cases
# ---------------------------------------------------------------------------

def test_noise_injection_risk_bound_zero_sigma_gives_zero():
    d = 3
    Sigma_x = np.eye(d)
    M_S = np.eye(d) * 5.0
    A_i = np.eye(d) * 2.0
    out = noise_injection_risk_bound(
        sigma_eta=0.0, Sigma_x=Sigma_x, M_S=M_S, A_i=A_i)
    assert out == pytest.approx(0.0, abs=1e-12)


def test_noise_injection_risk_bound_diagonal_case_matches_hand_derivation():
    """
    With Sigma_x = I, M_S = c*I, A_i = a*I:
        M_S^{-1} A_i M_S^{-1} = (a / c^2) I
        trace(Sigma_x @ that) = d * a / c^2
        Delta R = sigma_eta^2 * d * a / c^2
    Directly checks the trace formula's arithmetic, order of matrix
    multiplication, and squaring of sigma_eta are all correct.
    """
    d = 4
    a, c, sigma_eta = 3.0, 5.0, 0.7
    Sigma_x = np.eye(d)
    M_S = np.eye(d) * c
    A_i = np.eye(d) * a
    expected = (sigma_eta ** 2) * d * a / (c ** 2)
    out = noise_injection_risk_bound(sigma_eta, Sigma_x, M_S, A_i)
    assert out == pytest.approx(expected, rel=1e-10)


def test_noise_injection_risk_bound_matches_independent_numpy_recomputation_for_random_spd():
    rng = np.random.default_rng(0)
    d = 5
    # Build genuine SPD matrices from random Gram matrices + jitter.
    R1 = rng.normal(size=(d, d))
    Sigma_x = R1 @ R1.T + np.eye(d) * 0.1
    R2 = rng.normal(size=(d, d))
    M_S = R2 @ R2.T + np.eye(d) * 1.0
    R3 = rng.normal(size=(d, d))
    A_i = R3 @ R3.T
    sigma_eta = 0.4

    out = noise_injection_risk_bound(sigma_eta, Sigma_x, M_S, A_i)

    M_S_inv = np.linalg.inv(M_S)
    expected = sigma_eta ** 2 * np.trace(Sigma_x @ M_S_inv @ A_i @ M_S_inv)
    assert out == pytest.approx(expected, rel=1e-8)


def test_noise_injection_risk_bound_nondecreasing_in_sigma_eta():
    d = 3
    rng = np.random.default_rng(1)
    R = rng.normal(size=(d, d))
    Sigma_x = R @ R.T + np.eye(d) * 0.1
    M_S = np.eye(d) * 4.0
    A_i = np.eye(d) * 1.5

    vals = [noise_injection_risk_bound(
        s, Sigma_x, M_S, A_i) for s in (0.0, 0.1, 0.5, 1.0, 2.0)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_noise_injection_risk_bound_is_never_negative_for_valid_spd_inputs():
    rng = np.random.default_rng(2)
    for _ in range(20):
        d = rng.integers(2, 6)
        R1 = rng.normal(size=(d, d))
        Sigma_x = R1 @ R1.T
        R2 = rng.normal(size=(d, d))
        M_S = R2 @ R2.T + np.eye(d) * 1.0
        R3 = rng.normal(size=(d, d))
        A_i = R3 @ R3.T
        out = noise_injection_risk_bound(
            rng.uniform(0.1, 2.0), Sigma_x, M_S, A_i)
        assert out >= 0.0


def test_noise_injection_risk_bound_rejects_shape_mismatch():
    Sigma_x = np.eye(3)
    M_S = np.eye(3)
    A_i = np.eye(4)  # mismatched
    with pytest.raises(ValueError):
        noise_injection_risk_bound(0.5, Sigma_x, M_S, A_i)


def test_noise_injection_risk_bound_no_symmetry_validation_current_behavior():
    """
    DOCUMENTS A GAP, does not paper over it: as currently implemented,
    noise_injection_risk_bound performs no check that Sigma_x/M_S/A_i are
    symmetric (let alone PSD). Passing a non-symmetric M_S does NOT raise --
    it silently proceeds and can, in principle, return a value whose sign
    is not guaranteed by the stated PSD-trace argument. This test pins down
    that "does not raise" is the current behavior so a future change to add
    validation is a deliberate, visible diff rather than an unnoticed
    behavior change. If you intend to add symmetry validation (recommended,
    given the derivation explicitly assumes PSD structure), this test should
    be updated to pytest.raises(ValueError) at that time.
    """
    non_symmetric_M_S = np.array([[4.0, 1.0], [0.0, 4.0]])  # not symmetric
    Sigma_x = np.eye(2)
    A_i = np.eye(2)
    # Should not raise under current implementation:
    out = noise_injection_risk_bound(0.5, Sigma_x, non_symmetric_M_S, A_i)
    assert isinstance(out, float)
