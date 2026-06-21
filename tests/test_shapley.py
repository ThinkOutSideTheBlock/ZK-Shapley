"""
Shapley engine tests. Strategy: validate against TOY GAMES with known
closed-form Shapley values (hand-computable / well-known results), not just
the demand-forecasting game, so a bug in the Shapley math itself cannot hide
behind the complexity of the federated learning pipeline.
"""
import numpy as np
import pytest

from src.shapley import CachedGame, exact_shapley, monte_carlo_shapley, check_efficiency, check_null_player, check_symmetry


def test_exact_shapley_three_player_unanimity_game():
    """
    Unanimity game on {1,2,3}: v(S) = 1 if S == {1,2,3} else 0.
    Closed-form Shapley value for the unanimity game on the grand coalition is
    a well-known result: each player gets exactly 1/n. (Shapley, 1953.)
    """
    players = [1, 2, 3]

    def v(S):
        return 1.0 if S == frozenset(players) else 0.0

    game = CachedGame(players, v)
    phi = exact_shapley(game)
    for p in players:
        assert phi[p] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_exact_shapley_additive_game():
    """
    Additive (purely individual-value) game: v(S) = sum of fixed weights
    w_i for i in S. Closed-form Shapley value: phi_i = w_i exactly (each
    player's value is independent of coalition structure -- the textbook
    "additive game" sanity check).
    """
    weights = {1: 3.0, 2: 7.0, 3: 1.5, 4: -2.0}
    players = list(weights.keys())

    def v(S):
        return sum(weights[p] for p in S)

    game = CachedGame(players, v)
    phi = exact_shapley(game)
    for p in players:
        assert phi[p] == pytest.approx(weights[p], abs=1e-9)


def test_exact_shapley_glove_game():
    """
    Classic "glove game": players 1, 2 each hold a left glove; player 3 holds
    a right glove. v(S) = 1 if S contains at least one left AND one right
    glove, else 0. Known closed-form solution (standard cooperative game
    theory textbook example, e.g. Osborne & Rubinstein): phi_1 = phi_2 = 1/6,
    phi_3 = 2/3.
    """
    left = {1, 2}
    right = {3}

    def v(S):
        has_left = len(S & left) > 0
        has_right = len(S & right) > 0
        return 1.0 if (has_left and has_right) else 0.0

    players = [1, 2, 3]
    game = CachedGame(players, v)
    phi = exact_shapley(game)
    assert phi[1] == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert phi[2] == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert phi[3] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_efficiency_axiom_holds_numerically():
    weights = {1: 3.0, 2: 7.0, 3: 1.5, 4: -2.0, 5: 4.2}
    players = list(weights.keys())

    def v(S):
        return sum(weights[p] for p in S)

    game = CachedGame(players, v)
    phi = exact_shapley(game)
    passed, gap = check_efficiency(phi, game)
    assert passed, f"Efficiency axiom violated, gap={gap}"


def test_null_player_axiom():
    """Player 4 contributes nothing to any coalition -> must receive 0."""
    def v(S):
        return float(len(S & {1, 2, 3}))  # player 4 never affects v

    players = [1, 2, 3, 4]
    game = CachedGame(players, v)
    phi = exact_shapley(game)
    passed, gap = check_null_player(phi, game, player=4)
    assert passed, f"Null player axiom violated for player 4, |phi_4|={gap}"
    assert phi[4] == pytest.approx(0.0, abs=1e-9)


def test_symmetry_axiom():
    """Players 1 and 2 are interchangeable in this game -> equal Shapley payoff."""
    def v(S):
        score = 0.0
        if 1 in S or 2 in S:
            score += 5.0
        if 3 in S:
            score += 2.0
        return score

    players = [1, 2, 3]
    game = CachedGame(players, v)
    phi = exact_shapley(game)
    passed, gap = check_symmetry(phi, game, p1=1, p2=2)
    assert passed, f"Symmetry axiom violated, gap={gap}"


def test_monte_carlo_converges_to_exact():
    """
    On the glove game (closed-form ground truth available), Monte Carlo
    Shapley with a generous permutation budget should be within a tight
    tolerance of the exact value.
    """
    left, right = {1, 2}, {3}

    def v(S):
        return 1.0 if (len(S & left) > 0 and len(S & right) > 0) else 0.0

    players = [1, 2, 3]
    game_exact = CachedGame(players, v)
    phi_exact = exact_shapley(game_exact)

    game_mc = CachedGame(players, v)
    phi_mc, diag = monte_carlo_shapley(game_mc, n_permutations=20000, seed=123)

    for p in players:
        assert phi_mc[p] == pytest.approx(phi_exact[p], abs=0.02), (
            f"MC Shapley for player {p} = {phi_mc[p]:.4f}, exact = {phi_exact[p]:.4f}"
        )


def test_monte_carlo_satisfies_efficiency_approximately():
    weights = {1: 3.0, 2: 7.0, 3: 1.5, 4: -2.0, 5: 4.2}
    players = list(weights.keys())

    def v(S):
        return sum(weights[p] for p in S)

    game = CachedGame(players, v)
    phi, _ = monte_carlo_shapley(game, n_permutations=5000, seed=0)
    total = sum(phi.values())
    v_N = v(frozenset(players))
    assert total == pytest.approx(v_N, abs=1e-9), (
        "Monte Carlo permutation-sampling Shapley is exactly budget-balanced "
        "by construction (telescoping sum over each permutation), this is not "
        "an approximation -- any violation indicates an implementation bug."
    )
