"""
Hash commit-reveal protocol: anti-equivocation only.

Guarantee established (plan Section H.2, verbatim):

    "The implemented hash commitment proves that a participant opens the same
    sufficient statistics it previously committed to, subject to standard
    binding assumptions. It does not establish that those statistics were
    correctly calculated, that the underlying records are genuine or
    complete, or that observations were not duplicated."

This module implements EXACTLY that guarantee and nothing more. It does NOT
attempt to prove correctness of the ridge sufficient-statistic computation --
that is the job of the (separately costed, not implemented) proof layer
specified in proof_cost.py. Conflating the two is the single most damaging
overclaim this paper could make, so the boundary is enforced here at the
type level: `verify_opening` returns a binding-check boolean and nothing that
could be mistaken for a correctness certificate.
"""
from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass


class CommitmentError(Exception):
    """Raised when a commitment opening fails to bind, or on malformed input."""


@dataclass
class Commitment:
    """
    c_i = H(enc(A_i, b_i), r_i)  per plan Definition 1.

    `digest` is the public commitment value; `nonce` and the committed
    (A, b) pair must be retained privately by the committing firm and
    revealed later for `verify_opening`.
    """
    firm_name: str
    digest: bytes
    nonce: bytes


def _canonical_encode(A: np.ndarray, b: np.ndarray) -> bytes:
    """
    Deterministic, architecture-independent canonical encoding: fixed
    little-endian float64 byte layout plus explicit shape header, so that
    encoding order/padding ambiguity cannot be exploited to produce two
    distinct byte strings for numerically-equal (A, b).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    header = np.array(A.shape + b.shape, dtype=np.int64)
    return header.tobytes() + A.tobytes() + b.tobytes()


def commit_statistics(
    firm_name: str, A: np.ndarray, b: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[Commitment, np.ndarray, np.ndarray]:
    """
    Produce a binding commitment c_i = H(enc(A_i, b_i), r_i). Returns the
    public Commitment object plus the (A, b) pair the firm must retain to
    open later (returned for simulation convenience -- in a real deployment
    the firm already holds these).
    """
    if rng is None:
        rng = np.random.default_rng()
    nonce = rng.bytes(32)
    payload = _canonical_encode(A, b) + nonce
    digest = hashlib.sha256(payload).digest()
    return Commitment(firm_name=firm_name, digest=digest, nonce=nonce), A.copy(), b.copy()


def verify_opening(commitment: Commitment, A_opened: np.ndarray, b_opened: np.ndarray) -> bool:
    """
    Binding check: recompute H(enc(A_opened, b_opened), nonce) and compare
    against the previously published digest. Returns True iff the opening is
    consistent with the original commitment (anti-equivocation). This is
    NOT a correctness proof -- a firm that commits to a fabricated or
    incorrectly-computed (A, b) from the outset will pass this check, because
    binding only prevents CHANGING the value after commitment, not fabricating
    it before commitment (plan Section H.2 / H.4).
    """
    payload = _canonical_encode(A_opened, b_opened) + commitment.nonce
    recomputed = hashlib.sha256(payload).digest()
    return recomputed == commitment.digest


def verify_opening_or_raise(commitment: Commitment, A_opened: np.ndarray, b_opened: np.ndarray) -> None:
    """Convenience wrapper: raises CommitmentError on binding failure instead of returning False."""
    if not verify_opening(commitment, A_opened, b_opened):
        raise CommitmentError(
            f"Commitment binding check failed for firm '{commitment.firm_name}': "
            f"opened (A, b) does not match the previously published digest."
        )