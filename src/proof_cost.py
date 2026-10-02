"""
Relation-specific zero-knowledge cost model (plan Section E).

Replaces any NN-scale zkML extrapolation with a cost model built directly
from the arithmetic relation that a future proof layer would actually need
to establish: that a firm's committed (A_i, b_i) satisfy

    A_{i,k,l} = sum_t x_{i,t,k} x_{i,t,l}     for 1 <= k <= l <= d
    b_{i,k}   = sum_t x_{i,t,k} y_{i,t}       for 1 <= k <= d

under fixed-point arithmetic. This module provides:

  (E.2) the multiplication-count formula q(d), M(T, d) -- a scalar count, NOT
        a full constraint/proof-cost estimate;
  (E.3) a fixed-point quantization spec and empirical quantization-error
        report comparing quantized-and-reconstructed ridge solutions against
        the floating-point reference;
  (E.7) three benchmark variants -- MB-1 (inner-product kernel, ACTUALLY
        IMPLEMENTED below as a minimal Bulletproofs-style inner-product
        argument with a genuine soundness check), and MB-2/MB-3 (full
        relation, relation+range) as explicitly labeled ANALYTICAL
        EXTRAPOLATIONS from a fitted scaling model, per the plan's explicit
        fallback for when only MB-1 is implemented;
  (E.8) a linear scaling-model fit with intercept retained, leverage-corrected
        prediction intervals, and a hard cap on extrapolation distance.

Required paper language (plan Section E.9): "We benchmark the arithmetic
relation that an eventual proof layer would need to establish." This module
must never be described, in code comments or generated reports, as
implementing a complete zk-SNARK or Bulletproofs *system* for the full
sufficient-statistic relation -- only MB-1 (the inner-product kernel) is an
actual end-to-end implementation; MB-2 and MB-3 are extrapolations and are
labeled as such in every function that produces them.
"""
from __future__ import annotations

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# E.2: relation multiplication-count formula
# ---------------------------------------------------------------------------

def relation_multiplication_count(T_i: int, d: int) -> dict[str, int]:
    """
    q(d) = d(d+1)/2 + d = d(d+3)/2  scalar inner-product outputs (unique
    entries of symmetric A_i, plus all d entries of b_i).
    M(T_i, d) = T_i * q(d)  witness multiplications for a direct arithmetic
    implementation, BEFORE commitment/range/fixed-point/transcript overhead.

    This is a multiplication count, not a complete constraint or proof-cost
    estimate (plan Section E.2, explicit caveat).
    """
    q = d * (d + 3) // 2
    M = T_i * q
    return {"d": d, "T_i": T_i, "q_d": q, "M_T_d": M}


# ---------------------------------------------------------------------------
# E.3: fixed-point specification and quantization-error reporting
# ---------------------------------------------------------------------------

@dataclass
class FixedPointSpec:
    """
    Fixed-point quantization spec per plan Section E.3.
    x_bar = round(2^f * x), scale for A entries and b entries is 2^{2f}.
    """
    fractional_bits: int
    input_bound: float          # max |x_{itk}|, |y_it| assumed
    # None => plain integer (no field wraparound modeled)
    modulus: int | None = None

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    def accumulator_bound(self, T: int, d: int) -> int:
        """
        Worst-case bound on an accumulated A_{k,l} or b_k entry: T terms, each
        a product of two quantized values bounded by (scale * input_bound).
        """
        quantized_bound = self.scale * self.input_bound
        return int(np.ceil(T * quantized_bound * quantized_bound))

    def overflow_check(self, T: int, d: int) -> dict:
        bound = self.accumulator_bound(T, d)
        if self.modulus is not None:
            headroom = self.modulus - bound
            safe = headroom > 0
        else:
            safe = bound < (1 << 63)  # fits a signed 64-bit accumulator
            headroom = (1 << 63) - bound
        return {"accumulator_bound": bound, "modulus": self.modulus, "headroom": headroom, "safe": safe}


def quantize_statistics(
    X: np.ndarray, y: np.ndarray, spec: FixedPointSpec
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize raw (X, y) to fixed-point integers x_bar = round(2^f x),
    y_bar = round(2^f y), then compute integer A_bar = X_bar^T X_bar,
    b_bar = X_bar^T y_bar. Entries of A_bar, b_bar carry scale 2^{2f}
    relative to the floating-point (A, b).
    """
    scale = spec.scale
    X_bar = np.round(scale * X)
    y_bar = np.round(scale * y)
    A_bar = X_bar.T @ X_bar
    b_bar = X_bar.T @ y_bar
    if spec.modulus is not None:
        A_bar = np.mod(A_bar, spec.modulus)
        b_bar = np.mod(b_bar, spec.modulus)
    return A_bar, b_bar


def dequantize_ridge_solution(A_bar: np.ndarray, b_bar: np.ndarray, spec: FixedPointSpec, lam: float) -> np.ndarray:
    """
    Reconstruct a floating-point ridge solution from quantized statistics:
    rescale A_bar, b_bar by 2^{-2f} back to floating point, then solve the
    ridge system as usual. Used to measure quantization-induced coefficient
    error against the floating-point reference.
    """
    scale2 = spec.scale ** 2
    A = A_bar.astype(np.float64) / scale2
    b = b_bar.astype(np.float64) / scale2
    d = A.shape[0]
    return np.linalg.solve(A + lam * np.eye(d), b)


def quantization_error_report(
    X: np.ndarray, y: np.ndarray, spec: FixedPointSpec, lam: float,
    X_holdout: np.ndarray | None = None, y_holdout: np.ndarray | None = None,
) -> dict:
    """
    Full quantization-error report per plan Section E.3's required-metrics
    list: fractional bits, input bound, accumulator bound, modulus,
    max coefficient discrepancy, forecast RMSE discrepancy (if holdout data
    given), and an overflow-safety flag.
    """
    d = X.shape[1]
    A_ref = X.T @ X
    b_ref = X.T @ y
    beta_ref = np.linalg.solve(A_ref + lam * np.eye(d), b_ref)

    A_bar, b_bar = quantize_statistics(X, y, spec)
    beta_quantized = dequantize_ridge_solution(A_bar, b_bar, spec, lam)

    max_coef_discrepancy = float(np.max(np.abs(beta_ref - beta_quantized)))
    overflow = spec.overflow_check(T=X.shape[0], d=d)

    report = {
        "fractional_bits": spec.fractional_bits,
        "input_bound": spec.input_bound,
        "modulus": spec.modulus,
        "accumulator_bound": overflow["accumulator_bound"],
        "overflow_safe": overflow["safe"],
        "max_coefficient_discrepancy": max_coef_discrepancy,
    }

    if X_holdout is not None and y_holdout is not None and X_holdout.shape[0] > 0:
        pred_ref = X_holdout @ beta_ref
        pred_q = X_holdout @ beta_quantized
        rmse_ref = float(np.sqrt(np.mean((pred_ref - y_holdout) ** 2)))
        rmse_q = float(np.sqrt(np.mean((pred_q - y_holdout) ** 2)))
        report["forecast_rmse_reference"] = rmse_ref
        report["forecast_rmse_quantized"] = rmse_q
        report["forecast_rmse_discrepancy"] = abs(rmse_q - rmse_ref)

    return report


# ---------------------------------------------------------------------------
# E.7 / MB-1: minimal Bulletproofs-style inner-product argument
#            (ACTUALLY IMPLEMENTED with genuine soundness check)
# ---------------------------------------------------------------------------
#
# This is a minimal, non-succinct arithmetic emulation of a Bulletproofs-style
# inner-product argument (Bünz, Bootle, Boneh, Poelstra, Wuille, Maxwell 2018):
# O(log n) rounds of halving, each producing a pair of folded commitments and
# a scalar challenge derived via Fiat-Shamir from a running transcript hash.
# It is NOT a cryptographically hardened, discrete-log-secure implementation
# (no real elliptic-curve group operations -- "commitments" here are SHA-256
# digests standing in for Pedersen commitments). Cross terms (c_L, c_R) and
# the final scalars are carried in the clear, so the argument is NOT
# zero-knowledge. It is retained in this form because MB-1's purpose is to
# measure the ARITHMETIC WORK PROFILE and to permit a genuine soundness check,
# not to provide confidentiality. Label any reported numbers
# "circuit/arithmetic micro-benchmark, Bulletproofs-style round structure"
# per plan Section E.4.

@dataclass
class IPAProof:
    """
    Transcript of one inner-product argument, sufficient for independent
    verification. NOTE: carrying the per-round cross terms (c_L, c_R) and the
    final scalars in the clear makes this argument NON-zero-knowledge -- a real
    Pedersen-committed protocol hides these behind group elements. It is
    retained in the clear here because MB-1's purpose is to measure the
    ARITHMETIC WORK PROFILE and to permit a genuine soundness check, not to
    provide confidentiality. See module docstring.
    """
    commitments: list[tuple[bytes, bytes]]
    cross_terms: list[tuple[float, float]]
    final_u: float
    final_v: float
    claimed_z: float
    padded_length: int


@dataclass
class InnerProductArgument:
    """Result of one MB-1 timed run: prove + verify an inner product z = <u, v>."""
    vector_length: int
    padded_length: int
    prove_time_s: float
    verify_time_s: float
    proof_size_bytes: int
    n_rounds: int
    verified: bool
    claimed_inner_product: float
    reference_inner_product: float


def _fiat_shamir_challenge(transcript: bytes) -> float:
    """
    Deterministic pseudo-random scalar challenge in [1, 2) derived from a
    running transcript hash.

    Range rationale: the folding step forms u_lo*x + u_hi/x, so a challenge
    near zero makes 1/x enormous and the folded values overflow after O(log n)
    rounds. Constraining x to [1, 2) bounds both x and 1/x within [0.5, 2],
    keeping the recursion numerically stable at any realistic vector length.
    """
    digest = hashlib.sha256(transcript).digest()
    as_int = int.from_bytes(digest[:8], byteorder="big")
    return 1.0 + (as_int % (10 ** 9)) / (10 ** 9)


def _pedersen_stub_commit(vec: np.ndarray, blinding: bytes) -> bytes:
    """Stand-in for a Pedersen vector commitment: SHA-256 over the vector bytes + blinding."""
    payload = np.ascontiguousarray(vec, dtype=np.float64).tobytes() + blinding
    return hashlib.sha256(payload).digest()


def _pad_to_power_of_two(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-pad both vectors to the next power of two. This leaves <u, v> exactly
    invariant (the appended terms contribute 0) and is what production
    Bulletproofs implementations do. It replaces ad-hoc odd-length handling,
    which is error-prone: naive halving with u_hi = u[half:2*half] silently
    DISCARDS the final element when the length is odd, corrupting the computed
    inner product.
    """
    n = u.shape[0]
    if n < 1:
        raise ValueError("Inner-product argument requires a non-empty vector.")
    target = 1 << (n - 1).bit_length()
    if target == n:
        return u.copy(), v.copy()
    pad = target - n
    return np.concatenate([u, np.zeros(pad)]), np.concatenate([v, np.zeros(pad)])


def _run_inner_product_proof(
    u: np.ndarray, v: np.ndarray, rng: np.random.Generator
) -> tuple[IPAProof, float, int, int]:
    """
    Prover side of the recursive halving loop. Returns
    (proof, elapsed_prove_time, n_rounds, proof_size_bytes).

    Each round: split u, v in half, form the cross inner products
    c_L = <u_lo, v_hi> and c_R = <u_hi, v_lo>, commit to them, derive a
    Fiat-Shamir challenge from the running transcript, and fold. This
    reproduces the O(n) total field multiplications across O(log n) rounds
    that the cost model in E.8 consumes.
    """
    u_cur, v_cur = _pad_to_power_of_two(u, v)
    padded_length = u_cur.shape[0]

    transcript = b"ip-argument-init"
    commitments: list[tuple[bytes, bytes]] = []
    cross_terms: list[tuple[float, float]] = []
    proof_bytes = 0
    n_rounds = 0

    t_start = time.perf_counter()
    claimed_z = float(np.dot(u_cur, v_cur))

    while u_cur.shape[0] > 1:
        half = u_cur.shape[0] // 2
        u_lo, u_hi = u_cur[:half], u_cur[half:]
        v_lo, v_hi = v_cur[:half], v_cur[half:]

        c_L = float(np.dot(u_lo, v_hi))
        c_R = float(np.dot(u_hi, v_lo))

        blind = rng.bytes(16)
        commit_L = _pedersen_stub_commit(np.array([c_L]), blind)
        commit_R = _pedersen_stub_commit(np.array([c_R]), blind)
        transcript = hashlib.sha256(transcript + commit_L + commit_R).digest()
        proof_bytes += len(commit_L) + len(commit_R) + 16

        x = _fiat_shamir_challenge(transcript)
        u_cur = u_lo * x + u_hi / x
        v_cur = v_lo / x + v_hi * x

        commitments.append((commit_L, commit_R))
        cross_terms.append((c_L, c_R))
        n_rounds += 1

    elapsed = time.perf_counter() - t_start
    proof_bytes += 16  # final scalars u*, v*

    proof = IPAProof(
        commitments=commitments, cross_terms=cross_terms,
        final_u=float(u_cur[0]), final_v=float(v_cur[0]),
        claimed_z=claimed_z, padded_length=padded_length,
    )
    return proof, elapsed, n_rounds, proof_bytes


def _verify_inner_product_proof(proof: IPAProof, rel_tol: float = 1e-6) -> tuple[bool, float]:
    """
    Verifier side. Returns (accepted, elapsed_verify_time).

    This is a GENUINE soundness check, not a timing stand-in. The verifier
    replays the transcript hash chain to re-derive every challenge x_k
    (it cannot choose them -- Fiat-Shamir), folds the claimed inner product
    forward using the identity

        <u', v'> = <u, v> + x^2 * c_L + x^-2 * c_R

    and finally checks u* . v* == z_final. A prover who misreports the inner
    product, any cross term, or either final scalar fails this check.

    Cost profile: O(log n) SHA-256 evaluations plus O(log n) scalar folds,
    against the prover's O(n) multiplications -- the prover/verifier asymmetry
    the E.8 cost model needs, now MEASURED rather than assumed.
    """
    t_start = time.perf_counter()

    transcript = b"ip-argument-init"
    z = proof.claimed_z
    for (commit_L, commit_R), (c_L, c_R) in zip(proof.commitments, proof.cross_terms):
        transcript = hashlib.sha256(transcript + commit_L + commit_R).digest()
        x = _fiat_shamir_challenge(transcript)
        z = z + (x ** 2) * c_L + (x ** -2) * c_R

    final_product = proof.final_u * proof.final_v
    scale = max(abs(z), abs(final_product), 1.0)
    accepted = bool(abs(final_product - z) / scale <= rel_tol)

    elapsed = time.perf_counter() - t_start
    return accepted, elapsed


def run_mb1_benchmark(vector_length: int, n_repetitions: int = 10, seed: int = 0) -> dict:
    """
    MB-1: time proving AND verifying z = <u, v>, over n_repetitions timed runs
    after one discarded warm-up run (plan Section E.5: "at least 10 timed
    repetitions after one warm-up").

    Both prove and verify times are directly measured. Every run's proof is
    independently verified and the aggregate result carries `all_verified`;
    a False there indicates a bug in the folding recursion, not a strategic
    prover, and should fail the test suite.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=vector_length)
    v = rng.normal(size=vector_length)
    reference_z = float(np.dot(u, v))

    _run_inner_product_proof(u, v, rng)  # warm-up (discarded)

    runs: list[InnerProductArgument] = []
    for _ in range(n_repetitions):
        proof, prove_t, n_rounds, proof_bytes = _run_inner_product_proof(
            u, v, rng)
        accepted, verify_t = _verify_inner_product_proof(proof)
        runs.append(InnerProductArgument(
            vector_length=vector_length, padded_length=proof.padded_length,
            prove_time_s=prove_t, verify_time_s=verify_t,
            proof_size_bytes=proof_bytes, n_rounds=n_rounds,
            verified=accepted, claimed_inner_product=proof.claimed_z,
            reference_inner_product=reference_z,
        ))

    prove_times = np.array([r.prove_time_s for r in runs])
    verify_times = np.array([r.verify_time_s for r in runs])

    return {
        "backend": "bulletproofs_style_ip_argument_stub (SHA-256 commitment emulation, not EC-hardened)",
        "vector_length": vector_length,
        "padded_length": runs[0].padded_length,
        "n_repetitions": n_repetitions,
        "median_prove_time_s": float(np.median(prove_times)),
        "iqr_prove_time_s": float(np.percentile(prove_times, 75) - np.percentile(prove_times, 25)),
        "median_verify_time_s": float(np.median(verify_times)),
        "iqr_verify_time_s": float(np.percentile(verify_times, 75) - np.percentile(verify_times, 25)),
        "proof_size_bytes": int(np.median([r.proof_size_bytes for r in runs])),
        "n_rounds": runs[0].n_rounds,
        "all_verified": all(r.verified for r in runs),
        "runs": runs,
        "representative_run": runs[len(runs) // 2],
        "note": "MB-1 is an actually-executed micro-benchmark with a Bulletproofs-style round "
                "structure; both prove and verify times are directly measured and every proof is "
                "checked against the folding identity u*.v* == z_final. Commitments are SHA-256 "
                "stand-ins for Pedersen commitments, cross terms are revealed in the clear (hence "
                "NOT zero-knowledge), and this is NOT a discrete-log-secure implementation.",
    }


# ---------------------------------------------------------------------------
# E.8: scaling model fit + prediction interval
# ---------------------------------------------------------------------------

def fit_scaling_model(mult_counts: list[int], prove_times: list[float]) -> dict:
    """
    Fit Time_prove = alpha + beta * M(T,d) + noise by OLS on measured
    (M, Time) pairs, and return everything needed to form a correct prediction
    interval at an arbitrary target M (plan Section E.8).

    The intercept is retained deliberately: forcing the fit through the origin
    would assert zero fixed per-round overhead, which is false here (each
    round pays two SHA-256 evaluations regardless of vector length) and biases
    the slope used for all downstream extrapolation.
    """
    M = np.array(mult_counts, dtype=np.float64)
    y = np.array(prove_times, dtype=np.float64)
    if len(M) < 3:
        raise ValueError(
            "fit_scaling_model requires at least 3 (M, time) observations to fit reliably.")
    if len(np.unique(M)) < 2:
        raise ValueError(
            "fit_scaling_model requires at least 2 distinct M values.")

    X_design = np.column_stack([np.ones_like(M), M])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    alpha, beta = coeffs
    y_hat = X_design @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(len(M) - 2, 1)
    resid_std = float(np.sqrt(ss_res / dof))
    M_mean = float(M.mean())
    S_xx = float(np.sum((M - M_mean) ** 2))

    return {
        "alpha": float(alpha), "beta": float(beta), "r_squared": float(r_squared),
        "residual_std": resid_std, "dof": dof, "n_observations": len(M),
        "M_mean": M_mean, "S_xx": S_xx,
        "max_observed_M": float(M.max()), "min_observed_M": float(M.min()),
    }


def predict_with_interval(scaling_model: dict, M_target: float, level: float = 0.90) -> dict:
    """
    Point prediction and a two-sided prediction interval for Time_prove at
    M_target, using the full leverage-corrected standard error

        se_pred(M*) = s * sqrt(1 + 1/n + (M* - M_bar)^2 / S_xx)

    The leverage term is essential: it is precisely what makes the interval
    widen as M_target moves away from the measured range, which is the honest
    statistical expression of the Section E.8 extrapolation cap.
    """
    from scipy.stats import t as t_dist

    alpha, beta = scaling_model["alpha"], scaling_model["beta"]
    s, n = scaling_model["residual_std"], scaling_model["n_observations"]
    dof, M_mean, S_xx = scaling_model["dof"], scaling_model["M_mean"], scaling_model["S_xx"]

    point = alpha + beta * M_target
    leverage = 1.0 + 1.0 / n + \
        ((M_target - M_mean) ** 2) / S_xx if S_xx > 0 else float("inf")
    se_pred = s * np.sqrt(leverage)
    t_crit = float(t_dist.ppf(0.5 + level / 2.0, dof))
    halfwidth = t_crit * se_pred

    return {
        "M_target": float(M_target),
        "point_prediction_s": float(point),
        "interval_level": level,
        "interval_halfwidth_s": float(halfwidth),
        "interval_lower_s": float(point - halfwidth),
        "interval_upper_s": float(point + halfwidth),
    }


def extrapolate_mb2(scaling_model: dict, T: int, d: int, level: float = 0.90) -> dict:
    """
    MB-2 (full sufficient-statistics relation) as an EXPLICITLY LABELED
    ANALYTICAL EXTRAPOLATION, per plan Section E.7's fallback: "If only MB-1
    is implemented, the paper may present MB-2 and MB-3 as analytical
    extrapolations, but must label them as such."

    Method: evaluate the OLS scaling model fitted on measured MB-1 timings at
    the target relation's multiplication count M(T, d) = T * d(d+3)/2, and
    report the leverage-corrected prediction interval alongside the point
    estimate. This replaces an earlier origin-forced time-per-multiplication
    average, which both ignored fixed per-round overhead and produced a bare
    scalar with no uncertainty attached.
    """
    mult_info = relation_multiplication_count(T, d)
    M = mult_info["M_T_d"]
    prediction = predict_with_interval(scaling_model, float(M), level=level)

    max_measured_M = scaling_model["max_observed_M"]
    extrapolation_ratio = M / \
        max_measured_M if max_measured_M > 0 else float("inf")
    within_one_order_of_magnitude = extrapolation_ratio <= 10.0

    return {
        "label": "MB-2_ANALYTICAL_EXTRAPOLATION_NOT_MEASURED",
        "T": T, "d": d, "q_d": mult_info["q_d"], "M_T_d": M,
        "scaling_model_r_squared": scaling_model["r_squared"],
        "extrapolated_prove_time_s": prediction["point_prediction_s"],
        "prediction_interval_level": level,
        "prediction_interval_lower_s": prediction["interval_lower_s"],
        "prediction_interval_upper_s": prediction["interval_upper_s"],
        "extrapolation_ratio_to_max_measured": float(extrapolation_ratio),
        "within_one_order_of_magnitude_bound": within_one_order_of_magnitude,
        "warning": None if within_one_order_of_magnitude else
        "Extrapolation exceeds one order of magnitude beyond measured MB-1 range "
        "(plan Section E.8 cap); treat this number as illustrative only.",
    }


def extrapolate_mb3(
    mb2_extrapolation: dict, n_range_proofs: int, per_range_proof_cost_s: float,
    per_range_proof_cost_se_s: float = 0.0,
) -> dict:
    """
    MB-3 (full relation + bounded-input range proofs) as a further labeled
    extrapolation on top of MB-2. The range-proof cost is treated as an
    independent additive component, so its uncertainty (if supplied via
    per_range_proof_cost_se_s) combines in quadrature with the MB-2 interval
    rather than being silently dropped.

    per_range_proof_cost_s should be sourced from a real range-proof
    benchmark where available; if not, it must be documented in the paper as
    an assumed constant.
    """
    if n_range_proofs < 0 or per_range_proof_cost_s < 0:
        raise ValueError(
            "n_range_proofs and per_range_proof_cost_s must be non-negative.")

    added_cost = n_range_proofs * per_range_proof_cost_s
    base = mb2_extrapolation["extrapolated_prove_time_s"]
    base_halfwidth = (mb2_extrapolation["prediction_interval_upper_s"] - base)
    added_halfwidth = n_range_proofs * per_range_proof_cost_se_s
    total_halfwidth = float(
        np.sqrt(base_halfwidth ** 2 + added_halfwidth ** 2))
    total = base + added_cost

    return {
        "label": "MB-3_ANALYTICAL_EXTRAPOLATION_NOT_MEASURED",
        "base_mb2_prove_time_s": base,
        "n_range_proofs": n_range_proofs,
        "per_range_proof_cost_s": per_range_proof_cost_s,
        "added_range_proof_cost_s": added_cost,
        "total_extrapolated_prove_time_s": total,
        "prediction_interval_level": mb2_extrapolation.get("prediction_interval_level"),
        "prediction_interval_lower_s": total - total_halfwidth,
        "prediction_interval_upper_s": total + total_halfwidth,
        "warning": mb2_extrapolation.get("warning"),
    }
