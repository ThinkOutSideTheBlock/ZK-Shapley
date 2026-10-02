# tests/test_proof_cost.py
from __future__ import annotations

import numpy as np
import pytest

from src.proof_cost import (
    relation_multiplication_count, FixedPointSpec, quantize_statistics,
    dequantize_ridge_solution, quantization_error_report,
    IPAProof, InnerProductArgument,
    _fiat_shamir_challenge, _pedersen_stub_commit, _pad_to_power_of_two,
    _run_inner_product_proof, _verify_inner_product_proof, run_mb1_benchmark,
    fit_scaling_model, predict_with_interval, extrapolate_mb2, extrapolate_mb3,
)


# ---------------------------------------------------------------------------
# E.2: relation_multiplication_count
# ---------------------------------------------------------------------------

def test_relation_multiplication_count_matches_hand_derived_formula():
    # q(d) = d(d+1)/2 [unique A entries] + d [b entries] = d(d+3)/2
    for d in (1, 2, 3, 5, 10):
        expected_q = d * (d + 1) // 2 + d
        result = relation_multiplication_count(T_i=7, d=d)
        assert result["q_d"] == expected_q
        assert result["M_T_d"] == 7 * expected_q


def test_relation_multiplication_count_specific_values():
    result = relation_multiplication_count(T_i=5, d=3)
    assert result["q_d"] == 9   # 3*4/2 + 3 = 6+3=9
    assert result["M_T_d"] == 45


def test_relation_multiplication_count_scales_linearly_in_T():
    r1 = relation_multiplication_count(T_i=10, d=4)
    r2 = relation_multiplication_count(T_i=20, d=4)
    assert r2["M_T_d"] == 2 * r1["M_T_d"]


# ---------------------------------------------------------------------------
# E.3: FixedPointSpec / quantize / dequantize / quantization_error_report
# ---------------------------------------------------------------------------

def test_fixed_point_spec_scale_property():
    spec = FixedPointSpec(fractional_bits=10, input_bound=1.0)
    assert spec.scale == 1024


def test_accumulator_bound_matches_formula():
    spec = FixedPointSpec(fractional_bits=8, input_bound=2.0)
    T = 100
    quantized_bound = spec.scale * spec.input_bound
    expected = int(np.ceil(T * quantized_bound * quantized_bound))
    assert spec.accumulator_bound(T=T, d=5) == expected


def test_overflow_check_no_modulus_safe_for_small_problem():
    spec = FixedPointSpec(fractional_bits=8, input_bound=1.0)
    report = spec.overflow_check(T=100, d=4)
    assert report["safe"] is True
    assert report["modulus"] is None


def test_overflow_check_no_modulus_unsafe_for_huge_problem():
    spec = FixedPointSpec(fractional_bits=32, input_bound=1e6)
    report = spec.overflow_check(T=10**6, d=8)
    assert report["safe"] is False


def test_overflow_check_with_modulus_reports_headroom_and_safety_correctly():
    spec_safe = FixedPointSpec(
        fractional_bits=4, input_bound=1.0, modulus=10**12)
    report_safe = spec_safe.overflow_check(T=10, d=3)
    assert report_safe["safe"] is True
    assert report_safe["headroom"] > 0

    spec_unsafe = FixedPointSpec(
        fractional_bits=20, input_bound=100.0, modulus=97)
    report_unsafe = spec_unsafe.overflow_check(T=50, d=3)
    assert report_unsafe["safe"] is False
    assert report_unsafe["headroom"] < 0


def test_quantize_and_dequantize_recovers_reference_when_values_are_exact_multiples_of_scale():
    """
    Choosing X, y as exact multiples of 2^-fractional_bits eliminates rounding
    entirely, so the round-trip through quantize -> dequantize should match
    the direct floating-point ridge solution to numerical solver precision,
    not just approximately.
    """
    fractional_bits = 6
    scale = 1 << fractional_bits
    rng = np.random.default_rng(0)
    T, d = 40, 3
    X = np.round(rng.normal(size=(T, d)) * scale) / scale
    y = np.round(rng.normal(size=T) * scale) / scale
    spec = FixedPointSpec(fractional_bits=fractional_bits, input_bound=10.0)
    lam = 1.0

    A_ref = X.T @ X
    b_ref = X.T @ y
    beta_ref = np.linalg.solve(A_ref + lam * np.eye(d), b_ref)

    A_bar, b_bar = quantize_statistics(X, y, spec)
    beta_q = dequantize_ridge_solution(A_bar, b_bar, spec, lam)

    np.testing.assert_allclose(beta_q, beta_ref, atol=1e-8)


def test_quantize_statistics_applies_modulus_reduction():
    spec = FixedPointSpec(fractional_bits=4, input_bound=10.0, modulus=1000)
    X = np.array([[100.0, 0.0], [0.0, 100.0]])
    y = np.array([100.0, 100.0])
    A_bar, b_bar = quantize_statistics(X, y, spec)
    assert np.all(A_bar >= 0) and np.all(A_bar < 1000)
    assert np.all(b_bar >= 0) and np.all(b_bar < 1000)


def test_quantization_error_report_basic_fields_present_and_small_error_on_fine_grid():
    fractional_bits = 12
    scale = 1 << fractional_bits
    rng = np.random.default_rng(1)
    T, d = 60, 4
    X = np.round(rng.normal(size=(T, d)) * scale) / scale
    y = np.round(rng.normal(size=T) * scale) / scale
    spec = FixedPointSpec(fractional_bits=fractional_bits, input_bound=10.0)

    report = quantization_error_report(X, y, spec, lam=1.0)
    assert report["fractional_bits"] == fractional_bits
    assert report["overflow_safe"] is True
    assert report["max_coefficient_discrepancy"] < 1e-6


def test_quantization_error_report_includes_holdout_rmse_fields_when_provided():
    rng = np.random.default_rng(2)
    T, d, T_hold = 80, 3, 20
    X = rng.normal(size=(T, d))
    y = rng.normal(size=T)
    X_hold = rng.normal(size=(T_hold, d))
    y_hold = rng.normal(size=T_hold)
    spec = FixedPointSpec(fractional_bits=16, input_bound=10.0)

    report = quantization_error_report(
        X, y, spec, lam=1.0, X_holdout=X_hold, y_holdout=y_hold)
    assert "forecast_rmse_reference" in report
    assert "forecast_rmse_quantized" in report
    assert "forecast_rmse_discrepancy" in report
    assert report["forecast_rmse_discrepancy"] >= 0.0


def test_quantization_error_report_omits_holdout_fields_when_not_provided():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    spec = FixedPointSpec(fractional_bits=10, input_bound=10.0)
    report = quantization_error_report(X, y, spec, lam=1.0)
    assert "forecast_rmse_reference" not in report


def test_quantization_error_report_undersized_modulus_produces_larger_discrepancy_than_no_modulus():
    """
    Comparative rather than absolute-threshold test: an undersized modulus
    that forces A_bar/b_bar to wrap around should never produce a SMALLER
    coefficient discrepancy than the same problem with no modulus at all --
    wraparound is strictly a source of additional error, not a corrective one.
    """
    rng = np.random.default_rng(4)
    T, d = 40, 3
    X = rng.normal(size=(T, d)) * 50.0
    y = rng.normal(size=T) * 50.0

    spec_no_modulus = FixedPointSpec(
        fractional_bits=8, input_bound=100.0, modulus=None)
    spec_tiny_modulus = FixedPointSpec(
        fractional_bits=8, input_bound=100.0, modulus=97)

    report_no_mod = quantization_error_report(X, y, spec_no_modulus, lam=1.0)
    report_tiny_mod = quantization_error_report(
        X, y, spec_tiny_modulus, lam=1.0)

    assert report_tiny_mod["overflow_safe"] is False
    assert report_tiny_mod["max_coefficient_discrepancy"] >= report_no_mod["max_coefficient_discrepancy"]


# ---------------------------------------------------------------------------
# MB-1: pad / commit / challenge primitives
# ---------------------------------------------------------------------------

def test_pad_to_power_of_two_preserves_dot_product():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5, 7, 9, 16, 17):
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        u_pad, v_pad = _pad_to_power_of_two(u, v)
        assert u_pad.shape[0] == 1 << (n - 1).bit_length()
        assert np.dot(u_pad, v_pad) == pytest.approx(np.dot(u, v), abs=1e-10)


def test_pad_to_power_of_two_no_op_for_already_power_of_two_length():
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([5.0, 6.0, 7.0, 8.0])
    u_pad, v_pad = _pad_to_power_of_two(u, v)
    np.testing.assert_array_equal(u_pad, u)
    np.testing.assert_array_equal(v_pad, v)


def test_pad_to_power_of_two_rejects_empty_vector():
    with pytest.raises(ValueError):
        _pad_to_power_of_two(np.array([]), np.array([]))


def test_fiat_shamir_challenge_is_deterministic_and_bounded_in_one_to_two():
    t1 = b"some transcript bytes"
    x1 = _fiat_shamir_challenge(t1)
    x2 = _fiat_shamir_challenge(t1)
    assert x1 == x2
    assert 1.0 <= x1 < 2.0

    x3 = _fiat_shamir_challenge(b"different transcript")
    assert x3 != x1  # not a strict guarantee in general, but true for these two inputs


def test_pedersen_stub_commit_deterministic_given_same_inputs():
    vec = np.array([1.0, 2.0, 3.0])
    blind = b"0123456789abcdef"
    c1 = _pedersen_stub_commit(vec, blind)
    c2 = _pedersen_stub_commit(vec, blind)
    assert c1 == c2
    assert len(c1) == 32


def test_pedersen_stub_commit_differs_for_different_blinding():
    vec = np.array([1.0, 2.0, 3.0])
    c1 = _pedersen_stub_commit(vec, b"blinding_value_1")
    c2 = _pedersen_stub_commit(vec, b"blinding_value_2")
    assert c1 != c2


# ---------------------------------------------------------------------------
# MB-1: honest prove + verify round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 33])
def test_honest_proof_is_accepted_and_matches_reference_dot_product(n):
    rng = np.random.default_rng(0)
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    reference = float(np.dot(u, v))

    proof, prove_t, n_rounds, proof_bytes = _run_inner_product_proof(u, v, rng)
    accepted, verify_t = _verify_inner_product_proof(proof)

    assert accepted is True
    assert proof.claimed_z == pytest.approx(reference, abs=1e-9)
    assert n_rounds == (1 << (n - 1).bit_length()).bit_length() - 1
    assert prove_t >= 0.0
    assert verify_t >= 0.0
    assert proof_bytes > 0


def test_verify_rejects_tampered_claimed_z():
    rng = np.random.default_rng(1)
    u = rng.normal(size=8)
    v = rng.normal(size=8)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)

    tampered = IPAProof(
        commitments=proof.commitments, cross_terms=proof.cross_terms,
        final_u=proof.final_u, final_v=proof.final_v,
        claimed_z=proof.claimed_z + 1.0,  # lie about the inner product
        padded_length=proof.padded_length,
    )
    accepted, _ = _verify_inner_product_proof(tampered)
    assert accepted is False


def test_verify_rejects_tampered_cross_term():
    rng = np.random.default_rng(2)
    u = rng.normal(size=8)
    v = rng.normal(size=8)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)

    tampered_cross = list(proof.cross_terms)
    c_L0, c_R0 = tampered_cross[0]
    tampered_cross[0] = (c_L0 + 5.0, c_R0)  # lie about one cross term

    tampered = IPAProof(
        commitments=proof.commitments, cross_terms=tampered_cross,
        final_u=proof.final_u, final_v=proof.final_v,
        claimed_z=proof.claimed_z, padded_length=proof.padded_length,
    )
    accepted, _ = _verify_inner_product_proof(tampered)
    assert accepted is False


def test_verify_rejects_tampered_commitment():
    """
    Tampering with a commitment changes the transcript hash, which changes
    every subsequent Fiat-Shamir challenge derived from it. Since the folding
    identity check re-derives challenges from the (now-altered) transcript,
    a tampered commitment must break verification even though claimed_z and
    the cross terms are left untouched -- the prover cannot selectively lie
    about only the commitment bytes without the verifier noticing.
    """
    rng = np.random.default_rng(3)
    u = rng.normal(size=8)
    v = rng.normal(size=8)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)

    tampered_commitments = list(proof.commitments)
    commit_L0, commit_R0 = tampered_commitments[0]
    fake_commit_L0 = _pedersen_stub_commit(np.array([999.0]), b"0" * 16)
    tampered_commitments[0] = (fake_commit_L0, commit_R0)

    tampered = IPAProof(
        commitments=tampered_commitments, cross_terms=proof.cross_terms,
        final_u=proof.final_u, final_v=proof.final_v,
        claimed_z=proof.claimed_z, padded_length=proof.padded_length,
    )
    accepted, _ = _verify_inner_product_proof(tampered)
    assert accepted is False


def test_verify_rejects_tampered_final_scalar():
    rng = np.random.default_rng(4)
    u = rng.normal(size=8)
    v = rng.normal(size=8)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)

    tampered = IPAProof(
        commitments=proof.commitments, cross_terms=proof.cross_terms,
        final_u=proof.final_u + 0.5, final_v=proof.final_v,
        claimed_z=proof.claimed_z, padded_length=proof.padded_length,
    )
    accepted, _ = _verify_inner_product_proof(tampered)
    assert accepted is False


def test_verify_rejects_swapped_cross_terms():
    """
    c_L and c_R enter the folding identity with different (asymmetric)
    challenge powers (x^2 vs x^-2), so swapping them -- not just perturbing
    them -- must also be caught, confirming the check isn't accidentally
    symmetric in a way that would let a prover get away with mislabeling.
    """
    rng = np.random.default_rng(5)
    u = rng.normal(size=16)
    v = rng.normal(size=16)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)

    if proof.cross_terms[0][0] == pytest.approx(proof.cross_terms[0][1]):
        pytest.skip("degenerate draw where c_L == c_R for this seed/round")

    swapped_cross = list(proof.cross_terms)
    c_L0, c_R0 = swapped_cross[0]
    swapped_cross[0] = (c_R0, c_L0)

    tampered = IPAProof(
        commitments=proof.commitments, cross_terms=swapped_cross,
        final_u=proof.final_u, final_v=proof.final_v,
        claimed_z=proof.claimed_z, padded_length=proof.padded_length,
    )
    accepted, _ = _verify_inner_product_proof(tampered)
    assert accepted is False


def test_zero_vector_inner_product_proof_verifies():
    rng = np.random.default_rng(6)
    u = np.zeros(4)
    v = np.zeros(4)
    proof, _, _, _ = _run_inner_product_proof(u, v, rng)
    accepted, _ = _verify_inner_product_proof(proof)
    assert accepted is True
    assert proof.claimed_z == pytest.approx(0.0, abs=1e-12)


def test_single_element_vector_proof_has_zero_rounds_and_trivially_verifies():
    rng = np.random.default_rng(7)
    u = np.array([3.0])
    v = np.array([4.0])
    proof, _, n_rounds, _ = _run_inner_product_proof(u, v, rng)
    assert n_rounds == 0
    accepted, _ = _verify_inner_product_proof(proof)
    assert accepted is True
    assert proof.claimed_z == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# MB-1: run_mb1_benchmark
# ---------------------------------------------------------------------------

def test_run_mb1_benchmark_all_runs_verify_and_metadata_is_consistent():
    result = run_mb1_benchmark(vector_length=32, n_repetitions=5, seed=0)
    assert result["all_verified"] is True
    assert result["vector_length"] == 32
    assert result["padded_length"] == 32  # already a power of two
    assert result["n_rounds"] == 5  # log2(32)
    assert len(result["runs"]) == 5
    assert result["median_prove_time_s"] >= 0.0
    assert result["median_verify_time_s"] >= 0.0
    assert result["proof_size_bytes"] > 0


def test_run_mb1_benchmark_pads_non_power_of_two_length_correctly():
    result = run_mb1_benchmark(vector_length=20, n_repetitions=3, seed=1)
    assert result["padded_length"] == 32
    assert result["all_verified"] is True
    for run in result["runs"]:
        assert run.claimed_inner_product == pytest.approx(
            run.reference_inner_product, abs=1e-8)


def test_run_mb1_benchmark_prove_time_generally_increases_with_vector_length():
    """
    Loose monotonicity check averaged over repetitions rather than a single
    draw, since individual timing measurements are noisy: a much larger
    vector should not have a smaller median prove time than a much smaller
    one when both are measured with enough repetitions to average out noise.
    """
    small = run_mb1_benchmark(vector_length=16, n_repetitions=15, seed=2)
    large = run_mb1_benchmark(vector_length=1024, n_repetitions=15, seed=2)
    assert large["median_prove_time_s"] >= small["median_prove_time_s"]


def test_run_mb1_benchmark_representative_run_is_a_member_of_runs_list():
    result = run_mb1_benchmark(vector_length=8, n_repetitions=7, seed=3)
    assert result["representative_run"] in result["runs"]


def test_run_mb1_benchmark_note_field_present_and_mentions_not_zero_knowledge():
    result = run_mb1_benchmark(vector_length=8, n_repetitions=3, seed=4)
    assert "note" in result
    assert "NOT" in result["note"]


# ---------------------------------------------------------------------------
# E.8: fit_scaling_model
# ---------------------------------------------------------------------------

def test_fit_scaling_model_recovers_known_linear_relationship_exactly():
    M = [10, 20, 30, 40, 50]
    alpha_true, beta_true = 0.001, 0.0002
    y = [alpha_true + beta_true * m for m in M]
    model = fit_scaling_model(M, y)
    assert model["alpha"] == pytest.approx(alpha_true, abs=1e-10)
    assert model["beta"] == pytest.approx(beta_true, abs=1e-10)
    assert model["r_squared"] == pytest.approx(1.0, abs=1e-8)


def test_fit_scaling_model_rejects_fewer_than_three_observations():
    with pytest.raises(ValueError):
        fit_scaling_model([10, 20], [0.1, 0.2])


def test_fit_scaling_model_rejects_all_identical_M_values():
    with pytest.raises(ValueError):
        fit_scaling_model([10, 10, 10], [0.1, 0.2, 0.15])


def test_fit_scaling_model_handles_noisy_data_with_reasonable_r_squared():
    rng = np.random.default_rng(0)
    M = np.linspace(100, 1000, 20)
    y = 0.0005 + 0.00003 * M + rng.normal(0, 0.001, size=20)
    model = fit_scaling_model(list(M), list(y))
    assert model["beta"] > 0
    assert 0.0 <= model["r_squared"] <= 1.0
    assert model["n_observations"] == 20
    assert model["dof"] == 18


# ---------------------------------------------------------------------------
# E.8: predict_with_interval
# ---------------------------------------------------------------------------

def test_predict_with_interval_point_prediction_matches_linear_formula():
    M = [10, 20, 30, 40, 50]
    y = [0.001 + 0.0002 * m for m in M]
    model = fit_scaling_model(M, y)
    pred = predict_with_interval(model, M_target=100.0, level=0.90)
    expected_point = model["alpha"] + model["beta"] * 100.0
    assert pred["point_prediction_s"] == pytest.approx(
        expected_point, abs=1e-8)


def test_predict_with_interval_widens_away_from_measured_mean():
    """
    Core leverage-correction claim: the interval half-width at a target far
    from M_mean must exceed the half-width at M_mean itself, and must exceed
    the half-width at a nearer target too -- monotonic widening with
    |M_target - M_mean|, not just "nonzero at the edges."
    """
    rng = np.random.default_rng(1)
    M = np.linspace(100, 200, 10)
    y = 0.001 + 0.0001 * M + rng.normal(0, 0.0005, size=10)
    model = fit_scaling_model(list(M), list(y))

    near = predict_with_interval(model, M_target=model["M_mean"], level=0.90)
    mid = predict_with_interval(model, M_target=300.0, level=0.90)
    far = predict_with_interval(model, M_target=3000.0, level=0.90)

    assert near["interval_halfwidth_s"] < mid["interval_halfwidth_s"] < far["interval_halfwidth_s"]


def test_predict_with_interval_higher_confidence_level_gives_wider_interval():
    M = [10, 20, 30, 40, 50, 60]
    y = [0.001 + 0.0002 * m for m in M]
    model = fit_scaling_model(M, y)
    low_conf = predict_with_interval(model, M_target=35.0, level=0.50)
    high_conf = predict_with_interval(model, M_target=35.0, level=0.99)
    assert high_conf["interval_halfwidth_s"] > low_conf["interval_halfwidth_s"]


def test_predict_with_interval_lower_and_upper_bracket_point_prediction():
    M = [10, 20, 30, 40, 50]
    y = [0.001 + 0.0002 * m + eps for m,
         eps in zip(M, [0.0001, -0.0001, 0.0002, -0.0002, 0.0])]
    model = fit_scaling_model(M, y)
    pred = predict_with_interval(model, M_target=25.0, level=0.90)
    assert pred["interval_lower_s"] <= pred["point_prediction_s"] <= pred["interval_upper_s"]


# ---------------------------------------------------------------------------
# E.7/E.8: extrapolate_mb2 / extrapolate_mb3
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_scaling_model():
    M = [1000, 2000, 4000, 8000, 16000]
    alpha_true, beta_true = 0.0005, 1e-6
    rng = np.random.default_rng(0)
    y = [alpha_true + beta_true * m + rng.normal(0, 1e-6) for m in M]
    return fit_scaling_model(M, y)


def test_extrapolate_mb2_is_explicitly_labeled_as_analytical_extrapolation(fitted_scaling_model):
    result = extrapolate_mb2(fitted_scaling_model, T=100, d=8, level=0.90)
    assert result["label"] == "MB-2_ANALYTICAL_EXTRAPOLATION_NOT_MEASURED"


def test_extrapolate_mb2_M_T_d_matches_relation_multiplication_count(fitted_scaling_model):
    result = extrapolate_mb2(fitted_scaling_model, T=100, d=8, level=0.90)
    expected = relation_multiplication_count(100, 8)
    assert result["q_d"] == expected["q_d"]
    assert result["M_T_d"] == expected["M_T_d"]


def test_extrapolate_mb2_flags_warning_when_beyond_one_order_of_magnitude(fitted_scaling_model):
    # max observed M is 16000; force M(T,d) far beyond 160000
    result = extrapolate_mb2(fitted_scaling_model, T=10_000, d=50, level=0.90)
    assert result["extrapolation_ratio_to_max_measured"] > 10.0
    assert result["within_one_order_of_magnitude_bound"] is False
    assert result["warning"] is not None


def test_extrapolate_mb2_no_warning_within_measured_range(fitted_scaling_model):
    result = extrapolate_mb2(fitted_scaling_model, T=10, d=4, level=0.90)
    assert result["extrapolation_ratio_to_max_measured"] <= 10.0
    assert result["within_one_order_of_magnitude_bound"] is True
    assert result["warning"] is None


def test_extrapolate_mb3_is_explicitly_labeled_as_analytical_extrapolation(fitted_scaling_model):
    mb2 = extrapolate_mb2(fitted_scaling_model, T=50, d=6)
    mb3 = extrapolate_mb3(mb2, n_range_proofs=100,
                          per_range_proof_cost_s=0.001)
    assert mb3["label"] == "MB-3_ANALYTICAL_EXTRAPOLATION_NOT_MEASURED"


def test_extrapolate_mb3_added_cost_matches_n_times_per_proof_cost(fitted_scaling_model):
    mb2 = extrapolate_mb2(fitted_scaling_model, T=50, d=6)
    mb3 = extrapolate_mb3(mb2, n_range_proofs=200,
                          per_range_proof_cost_s=0.002)
    assert mb3["added_range_proof_cost_s"] == pytest.approx(0.4)
    assert mb3["total_extrapolated_prove_time_s"] == pytest.approx(
        mb2["extrapolated_prove_time_s"] + 0.4
    )


def test_extrapolate_mb3_rejects_negative_inputs(fitted_scaling_model):
    mb2 = extrapolate_mb2(fitted_scaling_model, T=50, d=6)
    with pytest.raises(ValueError):
        extrapolate_mb3(mb2, n_range_proofs=-1, per_range_proof_cost_s=0.001)
    with pytest.raises(ValueError):
        extrapolate_mb3(mb2, n_range_proofs=10, per_range_proof_cost_s=-0.001)


def test_extrapolate_mb3_combines_uncertainty_in_quadrature_when_se_provided(fitted_scaling_model):
    mb2 = extrapolate_mb2(fitted_scaling_model, T=50, d=6, level=0.90)
    base_halfwidth = mb2["prediction_interval_upper_s"] - \
        mb2["extrapolated_prove_time_s"]

    mb3_no_se = extrapolate_mb3(
        mb2, n_range_proofs=100, per_range_proof_cost_s=0.001, per_range_proof_cost_se_s=0.0)
    mb3_with_se = extrapolate_mb3(
        mb2, n_range_proofs=100, per_range_proof_cost_s=0.001, per_range_proof_cost_se_s=0.0005)

    hw_no_se = mb3_no_se["prediction_interval_upper_s"] - \
        mb3_no_se["total_extrapolated_prove_time_s"]
    hw_with_se = mb3_with_se["prediction_interval_upper_s"] - \
        mb3_with_se["total_extrapolated_prove_time_s"]

    assert hw_no_se == pytest.approx(base_halfwidth, abs=1e-12)
    assert hw_with_se > hw_no_se
    expected_hw_with_se = np.sqrt(base_halfwidth ** 2 + (100 * 0.0005) ** 2)
    assert hw_with_se == pytest.approx(expected_hw_with_se, abs=1e-10)


def test_extrapolate_mb3_inherits_warning_from_mb2(fitted_scaling_model):
    mb2_warned = extrapolate_mb2(fitted_scaling_model, T=10_000, d=50)
    mb3_warned = extrapolate_mb3(
        mb2_warned, n_range_proofs=5, per_range_proof_cost_s=0.001)
    assert mb3_warned["warning"] == mb2_warned["warning"]
    assert mb3_warned["warning"] is not None
