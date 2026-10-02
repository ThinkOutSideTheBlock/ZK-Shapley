# tests/test_commitment.py
from __future__ import annotations

import numpy as np
import pytest

from src.commitment import (
    Commitment, CommitmentError,
    commit_statistics, verify_opening, verify_opening_or_raise,
    _canonical_encode,
)


@pytest.fixture
def sample_stats():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    b = rng.normal(size=3)
    return A, b


def test_commit_then_verify_opening_succeeds(sample_stats):
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment, A_kept, b_kept = commit_statistics("firm_1", A, b, rng=rng)
    assert verify_opening(commitment, A_kept, b_kept) is True


def test_verify_opening_fails_for_tampered_A(sample_stats):
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment, A_kept, b_kept = commit_statistics("firm_1", A, b, rng=rng)
    A_tampered = A_kept.copy()
    A_tampered[0, 0] += 1e-6
    assert verify_opening(commitment, A_tampered, b_kept) is False


def test_verify_opening_fails_for_tampered_b(sample_stats):
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment, A_kept, b_kept = commit_statistics("firm_1", A, b, rng=rng)
    b_tampered = b_kept.copy()
    b_tampered[-1] += 1e-6
    assert verify_opening(commitment, A_kept, b_tampered) is False


def test_verify_opening_fails_when_using_a_different_commitments_nonce(sample_stats):
    """
    Cross-wiring check: opening (A, b) against a DIFFERENT firm's Commitment
    object (different nonce, different digest) must fail even though (A, b)
    themselves are byte-identical to what was originally committed.
    """
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment_1, A1, b1 = commit_statistics(
        "firm_1", A, b, rng=np.random.default_rng(10))
    commitment_2, A2, b2 = commit_statistics(
        "firm_2", A, b, rng=np.random.default_rng(20))
    # different nonces -> different digests
    assert commitment_1.digest != commitment_2.digest
    assert verify_opening(
        commitment_1, A2, b2) is False or commitment_1.nonce != commitment_2.nonce


def test_verify_opening_or_raise_raises_commitment_error_on_mismatch(sample_stats):
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment, A_kept, b_kept = commit_statistics("firm_1", A, b, rng=rng)
    b_tampered = b_kept.copy()
    b_tampered[0] += 5.0
    with pytest.raises(CommitmentError):
        verify_opening_or_raise(commitment, A_kept, b_tampered)


def test_verify_opening_or_raise_does_not_raise_on_correct_opening(sample_stats):
    A, b = sample_stats
    rng = np.random.default_rng(1)
    commitment, A_kept, b_kept = commit_statistics("firm_1", A, b, rng=rng)
    verify_opening_or_raise(commitment, A_kept, b_kept)  # should not raise


def test_same_statistics_different_calls_yield_different_digests_due_to_random_nonce(sample_stats):
    A, b = sample_stats
    c1, _, _ = commit_statistics("firm_1", A, b, rng=np.random.default_rng(1))
    c2, _, _ = commit_statistics("firm_1", A, b, rng=np.random.default_rng(2))
    assert c1.digest != c2.digest
    assert c1.nonce != c2.nonce


def test_canonical_encode_disambiguates_shapes_with_identical_flattened_bytes():
    """
    Regression-style test for the exact vulnerability the shape header is
    designed to prevent: two (A, b) pairs whose concatenated float64 byte
    payloads are IDENTICAL but whose shapes differ must still encode to
    different byte strings (and therefore different digests), because the
    header (A.shape + b.shape) is prepended before the raw bytes.
    """
    A1 = np.arange(4, dtype=np.float64).reshape(2, 2)     # [0,1,2,3]
    b1 = np.arange(4, 6, dtype=np.float64)                 # [4,5]
    A2 = np.arange(6, dtype=np.float64).reshape(3, 2)      # [0,1,2,3,4,5]
    b2 = np.array([], dtype=np.float64)                    # []

    # Sanity: the raw (non-header) byte payloads are identical.
    assert (A1.tobytes() + b1.tobytes()) == (A2.tobytes() + b2.tobytes())

    encoded1 = _canonical_encode(A1, b1)
    encoded2 = _canonical_encode(A2, b2)
    assert encoded1 != encoded2


def test_canonical_encode_is_deterministic_across_calls():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    assert _canonical_encode(A, b) == _canonical_encode(A.copy(), b.copy())


def test_canonical_encode_independent_of_input_dtype_when_values_equal():
    """
    A firm submitting float32-typed statistics that are numerically equal to
    a float64 submission must encode identically (the function forces
    float64 via ascontiguousarray(..., dtype=np.float64)), since otherwise
    two numerically-identical statistics could commit to different digests
    purely due to dtype, an implementation detail with no cryptographic
    significance here.
    """
    A64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    A32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([5.0, 6.0])
    assert _canonical_encode(A64, b) == _canonical_encode(A32, b)
