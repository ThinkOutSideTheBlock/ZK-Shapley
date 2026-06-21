"""
Shapley value engine for the coalitional data-sharing game.

The coalitional game is (N, v) where N is the set of firms and v: 2^N -> R is
the characteristic function. In this codebase v(S) is constructed (see
mechanism.py) as the realized out-of-sample forecast-utility gain from pooling
the sufficient statistics of firms in S via federated ridge regression,
normalized so v(emptyset) = 0.

The Shapley value of player i is:

    phi_i(v) = sum_{S subseteq N \\ {i}} [ |S|! (n-|S|-1)! / n! ] * (v(S u {i}) - v(S))

This is the *unique* payoff division satisfying four axioms (Shapley, 1953):
  1. Efficiency:    sum_i phi_i(v) = v(N)                    (full surplus is distributed, no burn)
  2. Symmetry:       v(S u {i}) = v(S u {j}) for all S => phi_i = phi_j   (identical contributors paid identically)
  3. Null player:   v(S u {i}) = v(S) for all S => phi_i = 0  (a firm contributing nothing is paid nothing)
  4. Additivity:     phi_i(v + w) = phi_i(v) + phi_i(w)        (linearity across independent games)

These are mathematically guaranteed by construction of the Shapley formula --
not something to "test" in the sense of an empirical hypothesis -- but we
verify axioms 1-3 numerically on every experiment run as a correctness check
on the *implementation* (a real and common source of bugs in permutation-based
Monte Carlo estimators), and additivity is verified on synthetic toy games in
the test suite where closed-form values are known analytically.

IMPORTANT SCOPING NOTE (stated explicitly per the paper's honesty requirements):
Shapley-value payment mechanisms are fair in the above sense, but are NOT in
general strategy-proof against arbitrary manipulation of reported value (this
is a known result in cooperative game theory -- e.g. players can profit by
"splitting" into multiple reporting identities, or by collusive misreporting,
under generic characteristic functions; see Lehrer, 1988 and subsequent
literature on manipulability of Shapley-based mechanisms). What we *do* prove
and verify in mechanism.py is a narrower, real claim: in THIS specific
construction, where v(S) is computed by the mechanism from REALIZED validation
performance rather than from self-reported metadata (e.g. claimed sample
counts), a firm cannot profit by lying about its data quantity, because the
payment is a function of what the firm's data actually contributed when
verified, not what it claims. Robustness to gradient-level manipulation
(submitting corrupted but real-looking statistics) is assessed empirically via
deviation simulation, not claimed as a clean theorem.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Hashable

import numpy as np


CharacteristicFunction = Callable[[frozenset], float]


class CachedGame:
    """
    Wraps a (typically expensive -- involves fitting a ridge model and
    evaluating RMSE) characteristic function with memoization, since exact and
    Monte Carlo Shapley computation both repeatedly query the same coalitions.
    """

    def __init__(self, players: list[Hashable], v: CharacteristicFunction):
        self.players = list(players)
        self._v = v
        self._cache: dict[frozenset, float] = {}
        self.n_evaluations = 0  # tracks actual (non-cached) calls, for cost reporting

    def value(self, coalition: frozenset) -> float:
        if coalition not in self._cache:
            self.n_evaluations += 1
            self._cache[coalition] = self._v(coalition)
        return self._cache[coalition]


def exact_shapley(game: CachedGame) -> dict[Hashable, float]:
    """
    Exact Shapley value via the direct weighted-marginal-contribution formula,
    O(2^n) characteristic function evaluations (memoized, so O(2^n) total
    regardless of how many players' values are requested). Tractable for
    n <= ~12 with a cheap v(); used here for n=5 as ground truth to validate
    the Monte Carlo estimator (see tests/test_shapley.py).
    """
    n = len(game.players)
    phi = {p: 0.0 for p in game.players}
    all_others = {p: [q for q in game.players if q != p] for p in game.players}

    for p in game.players:
        others = all_others[p]
        for r in range(len(others) + 1):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                marginal = game.value(S | {p}) - game.value(S)
                phi[p] += weight * marginal
    return phi


def monte_carlo_shapley(
    game: CachedGame,
    n_permutations: int = 2000,
    seed: int = 0,
    convergence_check_every: int = 200,
    convergence_tol: float = 1e-3,
) -> tuple[dict[Hashable, float], dict]:
    """
    Permutation-sampling Monte Carlo Shapley estimator (Castro et al., 2009;
    used at scale for data valuation by Ghorbani & Zou, "Data Shapley", ICML
    2019). For a uniformly random permutation pi of N, each player's marginal
    contribution when added to the set of players preceding it in pi is an
    unbiased sample of a term in the exact Shapley sum; averaging over many
    permutations converges to phi_i by the law of large numbers.

    Includes a running-mean convergence diagnostic (checked every
    `convergence_check_every` permutations): the L1 change in the payoff
    vector between checkpoints. This diagnostic is reported in the paper's
    Monte Carlo accuracy figure (estimated vs exact Shapley, n=5 case) rather
    than used to silently truncate -- we run the full `n_permutations` budget
    for determinism and report convergence behavior as a result, not as a
    stopping rule baked into the estimator.
    """
    rng = np.random.default_rng(seed)
    players = game.players
    n = len(players)
    running_sum = {p: 0.0 for p in players}
    count = 0
    convergence_trace = []  # (n_perm_so_far, max_abs_change_in_running_mean)
    prev_mean = {p: 0.0 for p in players}

    for k in range(1, n_permutations + 1):
        perm = rng.permutation(n)
        coalition = frozenset()
        prev_value = game.value(coalition)
        for idx in perm:
            p = players[idx]
            coalition = coalition | {p}
            new_value = game.value(coalition)
            running_sum[p] += new_value - prev_value
            prev_value = new_value
        count = k

        if k % convergence_check_every == 0:
            cur_mean = {p: running_sum[p] / count for p in players}
            max_change = max(abs(cur_mean[p] - prev_mean[p]) for p in players)
            convergence_trace.append((k, max_change))
            prev_mean = cur_mean

    phi = {p: running_sum[p] / count for p in players}
    diagnostics = {
        "n_permutations": n_permutations,
        "n_unique_coalition_evaluations": game.n_evaluations,
        "convergence_trace": convergence_trace,
    }
    return phi, diagnostics


def check_efficiency(phi: dict[Hashable, float], game: CachedGame, atol: float = 1e-6) -> tuple[bool, float]:
    """Axiom 1: sum_i phi_i == v(N). Returns (passes, absolute_gap)."""
    total = sum(phi.values())
    v_N = game.value(frozenset(game.players))
    gap = abs(total - v_N)
    return gap <= atol, gap


def check_null_player(
    phi: dict[Hashable, float], game: CachedGame, player: Hashable, atol: float = 1e-6
) -> tuple[bool, float]:
    """
    Axiom 3: if player contributes zero marginal value to every coalition,
    phi[player] should be (numerically) zero.
    """
    others = [p for p in game.players if p != player]
    max_marginal = 0.0
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            S = frozenset(subset)
            marginal = abs(game.value(S | {player}) - game.value(S))
            max_marginal = max(max_marginal, marginal)
    is_null_player = max_marginal <= atol
    if not is_null_player:
        return True, 0.0  # axiom not applicable; vacuously "passes"
    return abs(phi[player]) <= atol, abs(phi[player])


def check_symmetry(
    phi: dict[Hashable, float], game: CachedGame, p1: Hashable, p2: Hashable, atol: float = 1e-6
) -> tuple[bool, float]:
    """
    Axiom 2: if p1 and p2 are interchangeable (identical marginal contribution
    to every coalition not containing either), their Shapley payoffs should be equal.
    """
    others = [p for p in game.players if p not in (p1, p2)]
    max_diff = 0.0
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            S = frozenset(subset)
            m1 = game.value(S | {p1}) - game.value(S)
            m2 = game.value(S | {p2}) - game.value(S)
            max_diff = max(max_diff, abs(m1 - m2))
    is_symmetric = max_diff <= atol
    if not is_symmetric:
        return True, 0.0  # axiom not applicable; vacuously "passes"
    gap = abs(phi[p1] - phi[p2])
    return gap <= atol, gap
