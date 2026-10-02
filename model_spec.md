# Model Specification — ZK-Shapley (Route R2)

Any change to these definitions requires an update of this file and `docs/claims_registry.md`.

## 1. Primary characteristic functions

### 1.1 Residual-weighted operational value (PRIMARY)

\[
v^{\mathrm{res}}(S)
=
\sum_{i\in S}
w_i\bigl(\sigma_i(\{i\})-\sigma_i(S)\bigr),
\qquad
v^{\mathrm{res}}(\emptyset)=v^{\mathrm{res}}(\{i\})=0.
\]

- \(\sigma_i(T)\): Phase-1 firm predictive residual scale = validation RMS of \(\hat\beta_T\) on firm \(i\) (no mean-centering; min validation rows enforced).
- \(w_i=(c_{h,i}+c_{b,i})\,\varphi(z_{\tau_i})\,\sqrt{L_i+1}\), \(\tau_i=c_{b,i}/(c_{h,i}+c_{b,i})\).
- Weights are **inventory-motivated**. They are **not** proven to satisfy \(\Delta C_i=w_i\Delta\sigma_i\) under the multi-echelon simulator.

### 1.2 Unweighted residual / accuracy comparator

\[
v^{\mathrm{acc}}(S)
=
\sum_{i\in S}
\bigl(\sigma_i(\{i\})-\sigma_i(S)\bigr)
\quad\text{or hold-out RMSE reduction, as specified per experiment.}
\]

\(v^{\mathrm{acc}}(\{i\})=0\).

### 1.3 Path-cost value (SECONDARY / LIMITATION ONLY)

\[
v^{\mathrm{path}}(S)
=
\sum_{i\in S}
\bigl(C_i(\{i\})-C_i(S)\bigr).
\]

**Empirical status (R1):** residual-weighted Shapley ranks do **not** match path-cost Shapley ranks
(`results/r1_residual_vs_path/`, k1 mean Spearman \(\rho(\phi^{\mathrm{res}},\phi^{\mathrm{path}})\approx -0.17\)).
Path-cost is **not** the primary paper object under Route R2.

## 2. Estimation

- Exact federated ridge on training sufficient statistics \((A_i,b_i)\).
- Default \(\lambda\) mode: size-scaled \(\lambda= \lambda_0 n_S\) (Phase-1).
- Hold-out evaluation window identical across coalitions within a replication.

## 3. Allocation

- Exact Shapley for \(n\le 8\): `CachedGame(players=..., v_func=...)` then `exact_shapley(game)`.
- Efficiency: \(\sum_i\phi_i=v(N)\) (numerical gaps \(\sim 10^{-15}\)).
- IR \(\phi_i\ge 0\) is **empirical**, not guaranteed; if \(v(N)<0\), all-IR is impossible.

## 4. Integrity layer

- Hash commit–reveal: anti-equivocation only (binding of opened \((A,b)\) to prior commitment).
- Does **not** prove correctness of statistics or authenticity of underlying records.
- SNARK/Bulletproofs: literature-calibrated cost model only (`proof_cost.py`).

## 5. Non-claims

- No identity between residual-weighted value and path inventory savings.
- No universal strategy-proofness.
- No implemented zero-knowledge proof of correct computation.
- “ZK” in historical project name does **not** mean an implemented ZK valuation system.