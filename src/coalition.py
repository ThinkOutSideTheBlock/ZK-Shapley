"""
CoalitionEvaluator — the keystone of the ZK-Shapley research instrument.

Phase-1 updates (Gate 1):
  - firm-specific predictive residual (validation residual of β_S on firm i)
  - size-scaled regularization λ_S = λ0 * n_S (default)
  - legacy "fixed" λ and in-sample residual retained only behind explicit flags
    for ablation / regression tests

Maps a coalition S to:
  - fitted ridge model          fit(S)
  - inventory cost breakdown    cost(S)
  - operational value v_op(S)   value(S)
  - accuracy value v_acc(S)     value_accuracy(S)

Design decisions locked by docs/model_spec.md:
  - standalone / member-relative benchmark  →  v({i}) = 0
  - members-only evaluation scope
  - common-random-number discipline via stable seeds
  - identical hold-out windows for every coalition
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .federated import (
    FirmDataset,
    aggregate_coalition_statistics,
    fit_ridge_from_statistics,
    evaluate_holdout_rmse,
)
from .inventory import EchelonConfig, simulate_serial_supply_chain

# Phase-1 helpers
try:
    from .federated_phase1_patch import (
        coalition_train_size,
        effective_ridge_lambda,
        fit_ridge_for_coalition,
        firm_predictive_residual_std,
        firm_holdout_rmse,
    )
except ImportError:
    def coalition_train_size(datasets, coalition):
        return int(sum(datasets[n].n_train for n in coalition))

    def effective_ridge_lambda(lam0, n_S, mode="size_scaled"):
        if mode == "fixed":
            return float(lam0)
        return float(lam0) * max(int(n_S), 1)

    def fit_ridge_for_coalition(datasets, coalition, lam0=1.0, lambda_mode="size_scaled"):
        A, b = aggregate_coalition_statistics(datasets, coalition)
        n_S = coalition_train_size(datasets, coalition)
        lam = effective_ridge_lambda(lam0, n_S, mode=lambda_mode)
        return fit_ridge_from_statistics(A, b, lam=lam)

    def firm_predictive_residual_std(beta, ds, min_val_rows=10):
        if ds.n_val < min_val_rows:
            raise ValueError(
                f"Firm {ds.name!r}: n_val={ds.n_val} < min_val_rows={min_val_rows}. "
                "Validation residual is required; refusing silent train-residual fallback."
            )
        resid = ds.y_val - ds.X_val @ beta
        rms = float(np.sqrt(np.mean(resid ** 2)))
        return max(rms, 1e-6)

    def firm_holdout_rmse(beta, ds):
        if ds.n_holdout == 0:
            return float("nan")
        pred = ds.X_holdout @ beta
        return float(np.sqrt(np.mean((pred - ds.y_holdout) ** 2)))


@dataclass(frozen=True)
class CoalitionConfig:
    """
    All parameters that affect the numerical value of v(S).
    Changing any field produces a different config_hash and therefore a
    different cache namespace.
    """
    firms: tuple[str, ...]
    benchmark: Literal["standalone"] = "standalone"
    evaluation_scope: Literal["members_only"] = "members_only"
    lead_times: tuple[int, ...] = (1, 2, 3)
    holding_costs: tuple[float, ...] = (1.0, 0.6, 0.3)
    shortage_costs: tuple[float, ...] = (9.0, 0.0, 0.0)
    service_levels: tuple[float, ...] = (0.95, 0.90, 0.90)
    ridge_lambda: float = 1.0
    lambda_mode: Literal["size_scaled", "fixed"] = "size_scaled"
    residual_mode: Literal["predictive", "insample"] = "predictive"
    seed: int = 0

    def to_echelons(self) -> list[EchelonConfig]:
        return [
            EchelonConfig(
                holding_cost=h,
                shortage_cost=s,
                lead_time=lt,
                service_level=sl,
            )
            for h, s, lt, sl in zip(
                self.holding_costs,
                self.shortage_costs,
                self.lead_times,
                self.service_levels,
            )
        ]

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "firms": self.firms,
                "benchmark": self.benchmark,
                "evaluation_scope": self.evaluation_scope,
                "lead_times": self.lead_times,
                "holding_costs": self.holding_costs,
                "shortage_costs": self.shortage_costs,
                "service_levels": self.service_levels,
                "ridge_lambda": self.ridge_lambda,
                "lambda_mode": self.lambda_mode,
                "residual_mode": self.residual_mode,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class CostBreakdown:
    total: float
    holding: float
    shortage: float
    fill_rate: float
    per_firm: dict[str, float] = field(default_factory=dict)


class CoalitionEvaluator:
    """
    Maps coalition S → operational / accuracy value with multi-level caching.

    Phase-1 defaults:
      lambda_mode="size_scaled", residual_mode="predictive"
    """

    def __init__(
        self,
        datasets: dict[str, FirmDataset],
        demand_paths: dict[str, np.ndarray],
        config: CoalitionConfig,
        cache_dir: Path | str | None = None,
    ):
        self.datasets = datasets
        self.demand_paths = demand_paths
        self.config = config
        self.echelons = config.to_echelons()
        self._lam0 = config.ridge_lambda
        self._lambda_mode = config.lambda_mode
        self._residual_mode = config.residual_mode
        self._seed = config.seed

        self._model_cache: dict[frozenset[str], np.ndarray] = {}
        self._cost_cache: dict[frozenset[str], CostBreakdown] = {}
        self._value_op_cache: dict[frozenset[str], float] = {}
        self._value_acc_cache: dict[frozenset[str], float] = {}
        self._autarky_cost: dict[str, float] = {}
        self._autarky_rmse: dict[str, float] = {}

        self._cache_dir: Path | None = None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir) / config.config_hash()
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _stable_seed(self, *parts) -> int:
        h = hashlib.sha256()
        for p in parts:
            if isinstance(p, (frozenset, set)):
                h.update(b"|".join(sorted(str(s).encode() for s in p)))
            else:
                h.update(str(p).encode())
            h.update(b"\x00")
        return int.from_bytes(h.digest()[:8], "big")

    def _disk_key(self, kind: str, S: frozenset[str]) -> Path:
        assert self._cache_dir is not None
        key = hashlib.sha256(
            (kind + "|" + "|".join(sorted(S))).encode()
        ).hexdigest()[:20]
        return self._cache_dir / f"{kind}_{key}.pkl"

    def _load_disk(self, kind: str, S: frozenset[str]) -> Any | None:
        if self._cache_dir is None:
            return None
        path = self._disk_key(kind, S)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_disk(self, kind: str, S: frozenset[str], obj: Any) -> None:
        if self._cache_dir is None:
            return
        path = self._disk_key(kind, S)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _residual_std(self, beta: np.ndarray, name: str) -> float:
        """Firm-specific residual scale under the configured residual_mode."""
        ds = self.datasets[name]
        if self._residual_mode == "predictive":
            return firm_predictive_residual_std(beta, ds)
        # insample ablation: still use RMS about zero (retain bias)
        if ds.n_train < 2:
            return 1.0
        resid = ds.y_train - ds.X_train @ beta
        return max(float(np.sqrt(np.mean(resid ** 2))), 1e-6)

    def fit(self, S: frozenset[str]) -> np.ndarray:
        """Return the ridge coefficient vector for coalition S."""
        if not S:
            raise ValueError("Cannot fit the empty coalition.")
        if S in self._model_cache:
            return self._model_cache[S]

        cached = self._load_disk("model", S)
        if cached is not None:
            self._model_cache[S] = cached
            return cached

        beta = fit_ridge_for_coalition(
            self.datasets, S, lam0=self._lam0, lambda_mode=self._lambda_mode
        )
        self._model_cache[S] = beta
        self._save_disk("model", S, beta)
        return beta

    def _autarky_cost_for(self, name: str) -> float:
        if name in self._autarky_cost:
            return self._autarky_cost[name]
        ds = self.datasets[name]
        if ds.X_train.shape[0] == 0 or ds.X_holdout.shape[0] == 0:
            self._autarky_cost[name] = 0.0
            return 0.0
        beta = self.fit(frozenset({name}))
        forecast = ds.X_holdout @ beta
        residual_std = self._residual_std(beta, name)
        demand = self.demand_paths[name][-ds.X_holdout.shape[0]:]
        rng = np.random.default_rng(
            [self._seed, self._stable_seed("autarky", name)])
        out = simulate_serial_supply_chain(
            demand, forecast, residual_std, self.echelons, rng=rng
        )
        self._autarky_cost[name] = out.total_cost
        return out.total_cost

    def cost(self, S: frozenset[str]) -> CostBreakdown:
        if S in self._cost_cache:
            return self._cost_cache[S]

        cached = self._load_disk("cost", S)
        if cached is not None:
            self._cost_cache[S] = cached
            return cached

        if not S:
            bd = CostBreakdown(total=0.0, holding=0.0,
                               shortage=0.0, fill_rate=1.0)
            self._cost_cache[S] = bd
            return bd

        beta = self.fit(S)
        total_h = total_s = total_c = 0.0
        fill_num = fill_den = 0.0
        per_firm: dict[str, float] = {}

        for name in S:
            ds = self.datasets[name]
            if ds.X_holdout.shape[0] == 0:
                per_firm[name] = 0.0
                continue
            forecast = ds.X_holdout @ beta
            residual_std = self._residual_std(beta, name)
            demand = self.demand_paths[name][-ds.X_holdout.shape[0]:]
            rng = np.random.default_rng(
                [self._seed, self._stable_seed(S, name)]
            )
            out = simulate_serial_supply_chain(
                demand, forecast, residual_std, self.echelons, rng=rng
            )
            per_firm[name] = out.total_cost
            total_c += out.total_cost
            total_h += out.holding_cost
            total_s += out.shortage_cost
            fill_num += out.fill_rate * len(demand)
            fill_den += len(demand)

        bd = CostBreakdown(
            total=total_c,
            holding=total_h,
            shortage=total_s,
            fill_rate=(fill_num / fill_den) if fill_den > 0 else 1.0,
            per_firm=per_firm,
        )
        self._cost_cache[S] = bd
        self._save_disk("cost", S, bd)
        return bd

    def value(self, S: frozenset[str]) -> float:
        if S in self._value_op_cache:
            return self._value_op_cache[S]

        cached = self._load_disk("value_op", S)
        if cached is not None:
            self._value_op_cache[S] = cached
            return cached

        if not S:
            self._value_op_cache[S] = 0.0
            return 0.0

        coal_cost = self.cost(S)
        total_savings = 0.0
        for name in S:
            auto = self._autarky_cost_for(name)
            total_savings += auto - coal_cost.per_firm.get(name, 0.0)

        self._value_op_cache[S] = total_savings
        self._save_disk("value_op", S, total_savings)
        return total_savings

    def _autarky_rmse_for(self, name: str) -> float:
        if name in self._autarky_rmse:
            return self._autarky_rmse[name]
        beta = self.fit(frozenset({name}))
        rmse = firm_holdout_rmse(beta, self.datasets[name])
        self._autarky_rmse[name] = rmse
        return rmse

    def value_accuracy(self, S: frozenset[str]) -> float:
        if S in self._value_acc_cache:
            return self._value_acc_cache[S]

        cached = self._load_disk("value_acc", S)
        if cached is not None:
            self._value_acc_cache[S] = cached
            return cached

        if not S:
            self._value_acc_cache[S] = 0.0
            return 0.0

        beta = self.fit(S)
        total = 0.0
        for name in S:
            auto = self._autarky_rmse_for(name)
            coal = firm_holdout_rmse(beta, self.datasets[name])
            if np.isnan(auto) or np.isnan(coal):
                continue
            total += auto - coal

        self._value_acc_cache[S] = total
        self._save_disk("value_acc", S, total)
        return total

    def operational_v_func(self):
        return lambda S: self.value(S)

    def accuracy_v_func(self):
        return lambda S: self.value_accuracy(S)

    def cache_info(self) -> dict[str, int]:
        return {
            "models": len(self._model_cache),
            "costs": len(self._cost_cache),
            "value_op": len(self._value_op_cache),
            "value_acc": len(self._value_acc_cache),
        }
