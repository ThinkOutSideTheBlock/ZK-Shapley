"""
Cryptographic primitives for the ZK-Shapley protocol.

This module implements two distinct things, and is explicit about the
boundary between them:

(1) A REAL, fully implemented and tested cryptographic primitive: a
    hash-based commit-reveal scheme (binding + computationally hiding under
    the random oracle model). This is what the protocol actually uses to
    prevent equivocation -- a firm cannot submit sufficient statistics
    (A_i, b_i), observe what other firms submitted, and then change its own
    submission, because doing so would require finding a SHA-256 preimage
    collision. This is a standard, well-understood primitive (not novel
    cryptography) and is implemented in full, with no placeholders.

(2) An ANALYTICAL COST MODEL for the zk-SNARK proof of honest computation
    (i.e., a proof that A_i = X_i^T X_i and b_i = X_i^T y_i were correctly
    derived from data satisfying agreed-upon bounds, without revealing X_i,
    y_i). We do NOT implement a SNARK circuit (e.g. in circom or halo2) here
    -- that is out of scope for a mechanism-design contribution and would
    require months of dedicated cryptographic-engineering work to do
    correctly, which is exactly the kind of overclaiming this paper avoids.
    Instead, the proving/verification cost is estimated from a published
    benchmark, with the calibration assumption stated explicitly:

      Chen et al., "ZKML: An Optimizing System for ML Inference in
      Zero-Knowledge Proofs," EuroSys 2024 report ~6.65s proof generation for
      ResNet-101 inference (~44.5M multiply-accumulate parameters) using an
      optimized halo2-based circuit, with verification reported as orders of
      magnitude cheaper than proving.

    Our circuit (proving A_i = X_i^T X_i, b_i = X_i^T y_i for a ridge
    regression sufficient-statistics computation with d=6 features and
    n_i training rows) requires O(d^2 * n_i) multiply-accumulate operations,
    several orders of magnitude smaller than a CNN inference circuit. We
    therefore model proving time as scaling linearly with circuit
    multiplication-gate count, calibrated against the ZKML benchmark as a
    reference point, and report the result as an ESTIMATE, not a measurement.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# (1) Hash-based commit-reveal scheme
# ---------------------------------------------------------------------------

NONCE_BYTES = 32  # 256-bit nonce: standard, gives computational hiding under SHA-256/random-oracle model


def _serialize(*arrays: np.ndarray) -> bytes:
    """Deterministic byte serialization of one or more arrays for hashing."""
    parts = []
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        parts.append(a.shape.__repr__().encode("utf-8"))
        parts.append(a.tobytes())
    return b"||".join(parts)


@dataclass(frozen=True)
class Commitment:
    digest: bytes      # the public commitment, c = H(data || nonce)
    nonce: bytes        # kept secret by the committer until reveal


def commit(*arrays: np.ndarray, rng: np.random.Generator | None = None) -> Commitment:
    """
    Produce a binding, hiding commitment to one or more numpy arrays
    (typically a firm's sufficient statistics A_i, b_i).

    Binding: changing the committed data after the fact requires a SHA-256
    preimage collision (computationally infeasible).
    Hiding: a 256-bit uniformly random nonce is concatenated before hashing,
    so the commitment reveals no information about the data prior to reveal
    (standard random-oracle-model hiding argument for hash commitments).
    """
    if rng is None:
        nonce = os.urandom(NONCE_BYTES)
    else:
        nonce = rng.bytes(NONCE_BYTES)
    payload = _serialize(*arrays) + b"||" + nonce
    digest = hashlib.sha256(payload).digest()
    return Commitment(digest=digest, nonce=nonce)


def verify(commitment: Commitment, *arrays: np.ndarray) -> bool:
    """
    Verify that `arrays` are exactly the data that produced `commitment`,
    given the (now-revealed) nonce. Returns False on any mismatch, including
    a tampered/incorrect nonce.
    """
    payload = _serialize(*arrays) + b"||" + commitment.nonce
    recomputed = hashlib.sha256(payload).digest()
    return recomputed == commitment.digest


def verify_batch(commitments: dict[str, Commitment], revealed: dict[str, tuple[np.ndarray, ...]]) -> dict[str, bool]:
    """Verify a round of commit-reveals from multiple firms; returns per-firm pass/fail."""
    return {name: verify(commitments[name], *revealed[name]) for name in commitments}


# ---------------------------------------------------------------------------
# (2) Analytical ZK proof cost model (literature-calibrated estimate)
# ---------------------------------------------------------------------------

# Calibration point: Chen et al., EuroSys 2024 (ZKML), ResNet-101 proof generation.
_RESNET101_PARAMS = 44.5e6          # approx. multiply-accumulate operations, order of magnitude
_RESNET101_PROVE_SEC = 6.65          # reported proof generation time (optimized halo2 circuit)
_VERIFY_TO_PROVE_RATIO = 1e-3        # "orders of magnitude cheaper" per source; conservative 1000x used here
_GATES_PER_MAC = 1.0                 # one multiplication gate per multiply-accumulate, standard R1CS costing


def estimate_circuit_gates(n_train_rows: int, n_features: int) -> int:
    """
    Multiplication-gate count for proving A = X^T X, b = X^T y correctly
    computed: O(n_features^2 * n_train_rows) for A, O(n_features * n_train_rows)
    for b. Dominant term is the A computation.
    """
    gates_A = n_features * n_features * n_train_rows
    gates_b = n_features * n_train_rows
    return int(gates_A + gates_b)


@dataclass
class ZKCostEstimate:
    n_gates: int
    proving_time_sec: float
    verification_time_sec: float
    calibration_source: str = "Chen et al., ZKML, EuroSys 2024 (ResNet-101 benchmark)"


def estimate_zk_cost(n_train_rows: int, n_features: int) -> ZKCostEstimate:
    """
    Linear gate-count scaling against the ZKML EuroSys 2024 ResNet-101
    reference point. This is an explicit ESTIMATE for planning/discussion
    purposes (reported in the paper's cost-overhead table), not a measured
    benchmark of an implemented circuit.
    """
    n_gates = estimate_circuit_gates(n_train_rows, n_features)
    scale = n_gates / (_RESNET101_PARAMS * _GATES_PER_MAC)
    proving_time = _RESNET101_PROVE_SEC * scale
    # Floor proving time at a small constant to reflect fixed proof-system
    # overhead (trusted-setup-independent constant factors) regardless of how
    # small the circuit is -- avoids a misleadingly near-zero estimate.
    proving_time = max(proving_time, 0.05)
    verification_time = proving_time * _VERIFY_TO_PROVE_RATIO
    return ZKCostEstimate(
        n_gates=n_gates,
        proving_time_sec=proving_time,
        verification_time_sec=verification_time,
    )
