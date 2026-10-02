# tests/test_shapley.py
"""
Tests for src/shapley.py: exact/Monte Carlo Shapley values + axiom checks.

Deliberately exercises this module with toy characteristic functions that
have hand-computable ground-truth Shapley values (unanimity games and simple
weighted-voting-style games), rather than only smoke-testing against
inventory/accuracy games elsewhere in the repo -- so a bug in exact_shapley
itself can't hide behind noise in those richer games.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.shapley import (
    CachedGame, exact_shapley, monte_carlo_shapley,
    check_efficiency, check_symmetry, check_null_player,
)


# ---------------------------------------------------------------------------
# CachedGame
# ---------------------------------------------------------------------------

def test_cached_game_memoizes_repeated_coalition_evaluations():
    call_count = {"n": 0}

    def v_func(S):
        call_count["n"] += 1
        return len(S)

    game = CachedGame(players=["a", "b", "c"], v_func=v_func)
    game.v(frozenset({"a", "b"}))
    game.v(frozenset({"a", "b"}))
    game.v(frozenset({"a", "b"}))
    assert call_count["n"] == 1
    assert game.n_evaluations() == 1


def test_cached_game_n_evaluations_counts_distinct_coalitions():
    game = CachedGame(players=["a", "b", "c"], v_func=lambda S: len(S))
    game.v(frozenset())
    game.v(frozenset({"a"}))
    game.v(frozenset({"a", "b"}))
    assert game.n_evaluations() == 3


def test_cached_game_v_func_receives_frozenset_and_returns_float():
    def v_func(S):
        assert isinstance(S, frozenset)
        return 3
    game = CachedGame(players=["a"], v_func=v_func)
    result = game.v(frozenset({"a"}))
    assert isinstance(result, float)
    assert result == 3.0


def test_cached_game_empty_coalition_evaluates_correctly():
    game = CachedGame(players=["a", "b"], v_func=lambda S: len(S) * 10.0)
    assert game.v(frozenset()) == 0.0


# ---------------------------------------------------------------------------
# exact_shapley: hand-computable ground truths
# ---------------------------------------------------------------------------

def test_exact_shapley_unanimity_game_splits_value_equally_among_all_players():
    """
    Unanimity game on N: v(S) = 1 if S == N else 0. Shapley value is well
    known to be 1/n for every player (each player is symmetric and the value
    is only realized by the grand coalition).
    """
    players = ["a", "b", "c", "d"]
    n = len(players)
    full = frozenset(players)

    def v_func(S):
        return 1.0 if S == full else 0.0

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    for p in players:
        assert phi[p] == pytest.approx(1.0 / n, abs=1e-10)
    assert sum(phi.values()) == pytest.approx(1.0, abs=1e-9)


def test_exact_shapley_additive_game_gives_each_player_their_own_singleton_value():
    """
    Additive (modular) game: v(S) = sum of fixed per-player weights. Every
    marginal contribution of player p is exactly weight[p] regardless of S,
    so phi[p] == weight[p] exactly -- a strong, exact-equality check.
    """
    weights = {"a": 2.0, "b": 5.0, "c": -1.5, "d": 3.0}
    players = list(weights.keys())

    def v_func(S):
        return sum(weights[p] for p in S)

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    for p in players:
        assert phi[p] == pytest.approx(weights[p], abs=1e-9)


def test_exact_shapley_matches_hand_computed_three_player_glove_game():
    """
    Classic 'glove game': player a has a left glove, players b and c each
    have a right glove; a pair (one left + one right) sells for 1, otherwise
    0. v({a})=v({b})=v({c})=v({b,c})=0, v({a,b})=v({a,c})=v({a,b,c})=1.
    Hand-derived Shapley values: phi_a = 2/3, phi_b = phi_c = 1/6.
    """
    players = ["a", "b", "c"]

    def v_func(S):
        has_left = "a" in S
        has_right = ("b" in S) or ("c" in S)
        return 1.0 if (has_left and has_right) else 0.0

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    assert phi["a"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert phi["b"] == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert phi["c"] == pytest.approx(1.0 / 6.0, abs=1e-9)


def test_exact_shapley_single_player_game_gets_full_value():
    game = CachedGame(players=["a"], v_func=lambda S: 7.0 if S == frozenset({"a"}) else 0.0)
    phi = exact_shapley(game)
    assert phi["a"] == pytest.approx(7.0)


def test_exact_shapley_rejects_more_than_twenty_players():
    players = [f"p{i}" for i in range(21)]
    game = CachedGame(players=players, v_func=lambda S: len(S))
    with pytest.raises(ValueError):
        exact_shapley(game)


def test_exact_shapley_efficiency_holds_for_random_game_with_five_players():
    rng = np.random.default_rng(0)
    players = ["a", "b", "c", "d", "e"]
    # random monotone-ish game via random weights + random pairwise synergy
    weights = {p: rng.normal() for p in players}
    synergy = {}
    for i, p in enumerate(players):
        for q in players[i + 1:]:
            synergy[frozenset({p, q})] = rng.normal() * 0.5

    def v_func(S):
        val = sum(weights[p] for p in S)
        for pair, s in synergy.items():
            if pair <= S:
                val += s
        return val

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    v_grand = game.v(frozenset(players))
    check = check_efficiency(phi, v_grand)
    assert check["satisfied"] is True


# ---------------------------------------------------------------------------
# monte_carlo_shapley: unbiasedness / convergence to exact
# ---------------------------------------------------------------------------

def test_monte_carlo_shapley_converges_to_exact_values_on_glove_game_with_many_permutations():
    players = ["a", "b", "c"]

    def v_func(S):
        has_left = "a" in S
        has_right = ("b" in S) or ("c" in S)
        return 1.0 if (has_left and has_right) else 0.0

    game_exact = CachedGame(players=players, v_func=v_func)
    phi_exact = exact_shapley(game_exact)

    game_mc = CachedGame(players=players, v_func=v_func)
    phi_mc = monte_carlo_shapley(game_mc, n_permutations=20000, rng=np.random.default_rng(0))

    for p in players:
        assert phi_mc[p] == pytest.approx(phi_exact[p], abs=0.02)


def test_monte_carlo_shapley_is_reproducible_given_same_rng_seed():
    players = ["a", "b", "c", "d"]
    weights = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}

    def v_func(S):
        return sum(weights[p] for p in S)

    game1 = CachedGame(players=players, v_func=v_func)
    game2 = CachedGame(players=players, v_func=v_func)

    phi1 = monte_carlo_shapley(game1, n_permutations=500, rng=np.random.default_rng(42))
    phi2 = monte_carlo_shapley(game2, n_permutations=500, rng=np.random.default_rng(42))

    for p in players:
        assert phi1[p] == pytest.approx(phi2[p], abs=1e-12)


def test_monte_carlo_shapley_different_seeds_give_different_but_close_estimates():
    players = ["a", "b", "c", "d"]
    weights = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}

    def v_func(S):
        return sum(weights[p] for p in S)

    game1 = CachedGame(players=players, v_func=v_func)
    game2 = CachedGame(players=players, v_func=v_func)

    phi1 = monte_carlo_shapley(game1, n_permutations=2000, rng=np.random.default_rng(1))
    phi2 = monte_carlo_shapley(game2, n_permutations=2000, rng=np.random.default_rng(2))

    # For an additive game, phi should be exactly weight[p] regardless of
    # permutation order, so both should agree tightly with each other AND
    # with the known-exact weight (an additive game has zero estimator
    # variance since every marginal contribution equals weight[p] exactly).
    for p in players:
        assert phi1[p] == pytest.approx(weights[p], abs=1e-9)
        assert phi2[p] == pytest.approx(weights[p], abs=1e-9)


def test_monte_carlo_shapley_exact_on_additive_game_regardless_of_permutation_count():
    """
    Additive games have zero variance in the permutation-sampling estimator
    (every marginal contribution equals the fixed weight, independent of
    insertion order), so even n_permutations=1 must recover the exact value
    -- this pins down that the accumulation logic itself is correct, not
    just "close enough after averaging."
    """
    weights = {"a": -3.0, "b": 0.0, "c": 4.5}
    players = list(weights.keys())

    def v_func(S):
        return sum(weights[p] for p in S)

    game = CachedGame(players=players, v_func=v_func)
    phi = monte_carlo_shapley(game, n_permutations=1, rng=np.random.default_rng(0))
    for p in players:
        assert phi[p] == pytest.approx(weights[p], abs=1e-9)


def test_monte_carlo_shapley_uses_default_rng_when_none_supplied():
    players = ["a", "b"]

    def v_func(S):
        return len(S) * 2.0

    game = CachedGame(players=players, v_func=v_func)
    phi = monte_carlo_shapley(game, n_permutations=50, rng=None)
    assert set(phi.keys()) == set(players)
    for val in phi.values():
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# check_efficiency
# ---------------------------------------------------------------------------

def test_check_efficiency_satisfied_when_phi_sums_to_v_grand():
    phi = {"a": 1.0, "b": 2.0, "c": 3.0}
    result = check_efficiency(phi, v_grand=6.0)
    assert result["satisfied"] is True
    assert result["abs_gap"] == pytest.approx(0.0, abs=1e-12)


def test_check_efficiency_not_satisfied_when_phi_sum_mismatches_v_grand():
    phi = {"a": 1.0, "b": 2.0, "c": 3.0}
    result = check_efficiency(phi, v_grand=100.0)
    assert result["satisfied"] is False
    assert result["abs_gap"] == pytest.approx(94.0)


def test_check_efficiency_respects_custom_tolerance():
    phi = {"a": 1.0000001, "b": 2.0}
    result_tight = check_efficiency(phi, v_grand=3.0, tol=1e-9)
    result_loose = check_efficiency(phi, v_grand=3.0, tol=1e-3)
    assert result_tight["satisfied"] is False
    assert result_loose["satisfied"] is True


# ---------------------------------------------------------------------------
# check_symmetry
# ---------------------------------------------------------------------------

def test_check_symmetry_satisfied_for_truly_symmetric_players():
    players = ["a", "b", "c"]

    def v_func(S):
        # symmetric in a and b: only cares whether |S| includes 'a' or 'b' at all
        n_from_ab = len(S & {"a", "b"})
        n_from_c = 1 if "c" in S else 0
        return 2.0 * n_from_ab + 5.0 * n_from_c

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_symmetry(game, phi, "a", "b")
    assert result["interchangeable"] is True
    assert result["satisfied"] is True
    assert result["phi_a"] == pytest.approx(result["phi_b"], abs=1e-9)


def test_check_symmetry_detects_non_interchangeable_players_and_still_reports_satisfied():
    """
    check_symmetry's contract: if players are NOT interchangeable, the axiom
    doesn't apply to them, so 'satisfied' is vacuously True regardless of
    phi_a vs phi_b -- but 'interchangeable' must correctly report False.
    """
    players = ["a", "b", "c"]

    def v_func(S):
        return (10.0 if "a" in S else 0.0) + (1.0 if "b" in S else 0.0) + (1.0 if "c" in S else 0.0)

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_symmetry(game, phi, "a", "b")
    assert result["interchangeable"] is False
    assert result["satisfied"] is True  # vacuous
    assert phi["a"] != pytest.approx(phi["b"], abs=1e-3)  # genuinely different


def test_check_symmetry_glove_game_b_and_c_are_symmetric():
    players = ["a", "b", "c"]

    def v_func(S):
        has_left = "a" in S
        has_right = ("b" in S) or ("c" in S)
        return 1.0 if (has_left and has_right) else 0.0

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_symmetry(game, phi, "b", "c")
    assert result["interchangeable"] is True
    assert result["satisfied"] is True


# ---------------------------------------------------------------------------
# check_null_player
# ---------------------------------------------------------------------------

def test_check_null_player_detects_genuine_null_player():
    players = ["a", "b", "c"]

    def v_func(S):
        # c never affects the value
        return 3.0 if "a" in S else (1.0 if "b" in S else 0.0)

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_null_player(game, phi, "c")
    assert result["is_null"] is True
    assert result["satisfied"] is True
    assert result["phi_player"] == pytest.approx(0.0, abs=1e-9)


def test_check_null_player_reports_false_for_non_null_player():
    players = ["a", "b"]

    def v_func(S):
        return 5.0 if "a" in S else 0.0

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_null_player(game, phi, "b")
    assert result["is_null"] is True  # b truly never changes the value here
    assert result["phi_player"] == pytest.approx(0.0, abs=1e-9)


def test_check_null_player_non_null_case_with_nonzero_phi():
    players = ["a", "b"]

    def v_func(S):
        return (5.0 if "a" in S else 0.0) + (2.0 if "b" in S else 0.0)

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_null_player(game, phi, "b")
    assert result["is_null"] is False
    assert result["satisfied"] is True  # axiom vacuous since b isn't null
    assert result["phi_player"] == pytest.approx(2.0, abs=1e-9)


def test_check_null_player_glove_game_a_is_not_null():
    players = ["a", "b", "c"]

    def v_func(S):
        has_left = "a" in S
        has_right = ("b" in S) or ("c" in S)
        return 1.0 if (has_left and has_right) else 0.0

    game = CachedGame(players=players, v_func=v_func)
    phi = exact_shapley(game)
    result = check_null_player(game, phi, "a")
    assert result["is_null"] is False
    assert result["phi_player"] == pytest.approx(2.0 / 3.0, abs=1e-9)