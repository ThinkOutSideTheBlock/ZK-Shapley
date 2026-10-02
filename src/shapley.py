
"""
Exact and Monte Carlo Shapley value computation, plus axiom verification.

Used identically by BOTH the operational value function (inventory-cost
savings) and the accuracy value function (loss reduction) -- per plan Section
F.2's design requirement that "only the characteristic function changes"
between the two games being compared. This module is agnostic to what v(S)
represents.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CachedGame:
    """
    A coalitional game with memoized characteristic-function evaluations.
    v_func must accept a frozenset[str] (the coalition, possibly empty) and
    return a float.
    """
    players: list[str]
    v_func: Callable[[frozenset[str]], float]
    _cache: dict[frozenset[str], float] = field(default_factory=dict)

    def v(self, coalition: frozenset[str]) -> float:
        if coalition not in self._cache:
            self._cache[coalition] = float(self.v_func(coalition))
        return self._cache[coalition]

    def n_evaluations(self) -> int:
        return len(self._cache)


def exact_shapley(game: CachedGame) -> dict[str, float]:
    """
    Exact Shapley value via the marginal-contribution definition, summing over
    all 2^n coalitions. Intended for n in {5, 6, 8} per plan Section F.1
    ("compute exact Shapley values over all 2^n coalitions... use exact
    values as ground truth when assessing Monte Carlo approximation").
    """
    players = game.players
    n = len(players)
    if n > 20:
        raise ValueError(
            f"exact_shapley called with n={n} players; exact computation is "
            f"O(2^n) and intractable beyond ~20 players. Use monte_carlo_shapley."
        )

    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                S_with_p = S | {p}
                weight = math.factorial(
                    r) * math.factorial(n - r - 1) / math.factorial(n)
                marginal = game.v(S_with_p) - game.v(S)
                phi[p] += weight * marginal
    return phi


def monte_carlo_shapley(
    game: CachedGame, n_permutations: int, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """
    Permutation-sampling Monte Carlo Shapley estimator: for each of
    n_permutations random player orderings, accumulate each player's marginal
    contribution when inserted at its position in that ordering, then average.
    Unbiased estimator of the exact Shapley value.
    """
    if rng is None:
        rng = np.random.default_rng()
    players = list(game.players)
    n = len(players)
    totals = {p: 0.0 for p in players}

    for _ in range(n_permutations):
        order = players.copy()
        rng.shuffle(order)
        prefix: list[str] = []
        prev_value = game.v(frozenset())
        for p in order:
            prefix.append(p)
            new_value = game.v(frozenset(prefix))
            totals[p] += new_value - prev_value
            prev_value = new_value

    return {p: totals[p] / n_permutations for p in players}


def check_efficiency(phi: dict[str, float], v_grand: float, tol: float = 1e-6) -> dict[str, float]:
    """Efficiency axiom: sum_i phi_i == v(N). Returns diagnostic dict, does not raise."""
    total = sum(phi.values())
    return {"sum_phi": total, "v_grand": v_grand, "abs_gap": abs(total - v_grand), "satisfied": abs(total - v_grand) <= tol}


def check_symmetry(
    game: CachedGame, phi: dict[str, float], player_a: str, player_b: str, tol: float = 1e-6
) -> dict:
    """
    Symmetry axiom check: if a and b are interchangeable (v(S U {a}) ==
    v(S U {b}) for every S not containing either), they must receive equal
    Shapley values. Verifies interchangeability empirically over all
    coalitions before checking the payoff equality.
    """
    others = [p for p in game.players if p not in (player_a, player_b)]
    interchangeable = True
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            S = frozenset(subset)
            if abs(game.v(S | {player_a}) - game.v(S | {player_b})) > tol:
                interchangeable = False
                break
        if not interchangeable:
            break
    diff = abs(phi[player_a] - phi[player_b])
    return {"interchangeable": interchangeable, "phi_a": phi[player_a], "phi_b": phi[player_b],
            "abs_diff": diff, "satisfied": (not interchangeable) or diff <= tol}


def check_null_player(game: CachedGame, phi: dict[str, float], player: str, tol: float = 1e-6) -> dict:
    """Null-player axiom: if player never changes any coalition's value, phi_player must be 0."""
    others = [p for p in game.players if p != player]
    is_null = True
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            S = frozenset(subset)
            if abs(game.v(S | {player}) - game.v(S)) > tol:
                is_null = False
                break
        if not is_null:
            break
    return {"is_null": is_null, "phi_player": phi[player],
            "satisfied": (not is_null) or abs(phi[player]) <= tol}
