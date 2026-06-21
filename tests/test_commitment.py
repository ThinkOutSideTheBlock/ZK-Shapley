"""
Tests for the commit-reveal scheme. Verifies the two properties that make it
a valid cryptographic commitment: binding (tampering is detected) and hiding
(commitments to different/random nonces look unrelated even for identical data).
"""
import numpy as np
import pytest

from src.commitment import commit, verify, verify_batch, estimate_zk_cost, estimate_circuit_gates


def test_honest_reveal_verifies():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    rng = np.random.default_rng(0)
    c = commit(A, b, rng=rng)
    assert verify(c, A, b) is True


def test_tampered_data_fails_verification():
    """Binding property: changing even one element after commitment must be detected."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    rng = np.random.default_rng(0)
    c = commit(A, b, rng=rng)

    A_tampered = A.copy()
    A_tampered[0, 0] = 999.0
    assert verify(c, A_tampered, b) is False


def test_tampered_nonce_fails_verification():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    rng = np.random.default_rng(0)
    c = commit(A, b, rng=rng)

    from src.commitment import Commitment
    forged = Commitment(digest=c.digest, nonce=b"\x00" * 32)
    assert verify(forged, A, b) is False


def test_same_data_different_nonce_gives_different_commitment():
    """Hiding property check: identical data committed twice (independent nonces)
    must produce different public commitments (otherwise an observer could link
    repeated submissions of the same value, leaking information)."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    rng = np.random.default_rng(0)
    c1 = commit(A, b, rng=rng)
    c2 = commit(A, b, rng=rng)
    assert c1.digest != c2.digest
    assert c1.nonce != c2.nonce


def test_verify_batch_detects_single_cheater():
    rng = np.random.default_rng(1)
    A1, b1 = np.eye(2), np.array([1.0, 2.0])
    A2, b2 = np.eye(2) * 2, np.array([3.0, 4.0])
    c1 = commit(A1, b1, rng=rng)
    c2 = commit(A2, b2, rng=rng)

    commitments = {"firm_a": c1, "firm_b": c2}
    # firm_b tries to reveal different data than it committed to
    revealed = {"firm_a": (A1, b1), "firm_b": (np.eye(2) * 999, b2)}
    results = verify_batch(commitments, revealed)
    assert results["firm_a"] is True
    assert results["firm_b"] is False


def test_zk_cost_estimate_positive_and_scales_with_data():
    small = estimate_zk_cost(n_train_rows=100, n_features=6)
    large = estimate_zk_cost(n_train_rows=10000, n_features=6)
    assert small.proving_time_sec > 0
    assert large.proving_time_sec > small.proving_time_sec
    assert small.verification_time_sec < small.proving_time_sec, (
        "Verification must be strictly cheaper than proving (the core zkML "
        "asymmetry the cost model is calibrated against)."
    )


def test_circuit_gate_count_scaling():
    gates_small = estimate_circuit_gates(n_train_rows=100, n_features=6)
    gates_large = estimate_circuit_gates(n_train_rows=200, n_features=6)
    # Gate count for the dominant A = X^T X term scales linearly in n_train_rows.
    assert gates_large == pytest.approx(2 * gates_small, rel=0.01)
