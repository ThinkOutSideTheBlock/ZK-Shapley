```markdown
# AGENT.md — ZK-Shapley / Operational Federated Valuation

> Context for coding agents (VS Code / Cursor / Copilot / Claude Code).
> Read before editing `src/` or `experiments/`. Do not expand scope beyond this file.

---

## 1. Project identity (LOCKED — Route **R2**)

**Route R1 is closed.**  
Experiment `results/r1_residual_vs_path/` showed residual-weighted Shapley does **not** track path-cost Shapley ranks (k1 mean Spearman ρ(φ_res, φ_path) ≈ −0.17).  
We do **not** claim residual value = inventory cost savings.

**One-sentence goal (R2):**  
Publish an OR/analytics paper + reference implementation where competing firms pool data via **exact federated ridge**, coalition value is **inventory-weighted residual reduction** \(v^{\mathrm{res}}(S)=\sum_i w_i\Delta\sigma_i(S)\), payments are **Shapley**, reporting is **commit–reveal** (not a zk-SNARK), and ranks diverge from unweighted residual/accuracy games when inventory primitives make weights unequal (\(\kappa_{\mathrm{eff}}>1\)).

**Working title options (R2):**
- Inventory-Weighted Residual Shapley for Federated Demand Forecasting among Competing Firms  
- Operational Weighting of Federated Forecast Contributions: A Residual-Scale Shapley Mechanism  

**Not the goal:**
- Path-cost as primary characteristic function  
- Proving \(\Delta C_i = w_i\Delta\sigma_i\)  
- Production zk-SNARK / ZK-Value competitor  
- Universal strategy-proofness  
- Deep FL models  

---

## 2. Primary scientific objects (R2)

| Object | Role |
|--------|------|
| \(\sigma_i(S)\) | Phase-1 firm predictive residual (validation RMS of \(\hat\beta_S\)) |
| \(w_i\) | Inventory-**motivated** weight \((c_h+c_b)\varphi(z_\tau)\sqrt{L+1}\) — **motivated, not proven cost identity** |
| \(v^{\mathrm{res}}(S)\) | \(\sum_{i\in S} w_i(\sigma_i(\{i\})-\sigma_i(S))\), \(v(\{i\})=0\) — **PRIMARY** |
| \(v^{\mathrm{acc}}(S)\) | Unweighted residual/RMSE reduction — comparator |
| \(\phi^{\mathrm{res}},\phi^{\mathrm{acc}}\) | Exact Shapley |
| Path-cost \(v^{\mathrm{path}}\) | **Limitation / negative result only** (archived R1 table) |

**Algebra (keep):** equal \(w\) ⇒ same ranks as unweighted residual game (linearity of Shapley).  
**Empirical (keep):** cost-channel \(\kappa_{\mathrm{eff}}\) sweeps show rank divergence when weights differ.  
**Honest limit:** path-cost ranks ≠ residual ranks under current simulator.

---

## 3. Implementation scope (unchanged core)

```text
local data
  → sufficient statistics (exact federated ridge)
  → σ_i(S) (Phase-1 predictive residual)
  → v_res / v_acc  (+ path only for diagnostics)
  → exact Shapley
  → commit–reveal (anti-equivocation only)
```

| Path | Role |
|------|------|
| `src/demand.py` | DGP; fixed ladders when profiles=`fixed` |
| `src/federated.py` / `federated_phase1_patch.py` | Stats, ridge, residual RMS |
| `src/coalition.py` | Evaluator caches |
| `src/inventory.py` | Simulator (secondary / probes only) |
| `src/shapley.py` | `CachedGame(players=..., v_func=...)`, `exact_shapley(game)` |
| `src/mechanism.py` | Runners, divergence metrics |
| `src/commitment.py` | Hash commit–reveal only |
| `src/proof_cost.py` | Literature-calibrated cost model, not full SNARK |
| `docs/model_spec.md` | Frozen definitions — **must say residual ≠ path cost** |
| `docs/claims_registry.md` | R1 savings claims → Withdrawn / Limitation |

**API facts agents must respect:**
```python
CachedGame(players=firm_names, v_func=v_func)
phi = exact_shapley(game)   # dict[str, float]
```

---

## 4. What is done vs what remains

### Done (do not re-litigate without evidence)
- Exact federated ridge + tests  
- Residual-scale games, efficiency  
- Cost-channel rank divergence vs \(\kappa_{\mathrm{eff}}\) (\(n\in\{4,6,8\}\))  
- Nested residual long-series reference  
- IR / \(P(v(N)<0)\) diagnostics  
- \(C^{\mathrm{unc}}\) homogeneity probes  
- **R1 residual vs path: FAIL** → archived under `results/r1_residual_vs_path/`

### Active R2 work (paper + light code)
1. Rename claims: “inventory-weighted residual,” not “inventory savings.”  
2. Document R1 negative result as limitation.  
3. Fix trend-test reporting (seed-level / paired) on **existing** cost-channel CSVs.  
4. Related work: Raghunathan (2003), Leng–Parlar (2009), Ali et al. (2012), etc.  
5. Reproducibility: public archive + run commands.  
6. Optional small code: export weight vectors \((c_h,c_b,w)\) into experiment summaries.

### Out of scope
- Re-tuning path cost to force ρ(φ_res, φ_path) high  
- Full SNARK  
- Universal IC  
- Deep models  
- \(n\gg 8\) as headline  

---

## 5. Coding rules for agents

1. Primary value function in new experiments = **residual-weighted** or accuracy — not path-cost.  
2. Path-cost code may stay for probes; do not promote it to main tables.  
3. Do not change fixed DGP ladders without explicit user approval.  
4. Preserve `v({i})=0` for residual and accuracy games.  
5. Commit–reveal = binding openings only; never “proves true private data.”  
6. Prefer exact Shapley for \(n\le 8\).  
7. Update `docs/model_spec.md` / claims registry when renaming scientific objects.  
8. If asked to “fix R1 / make residual = cost,” refuse and point to this file + R1 results.

---

## 6. Success criteria (R2 submission)

| Criterion | Target |
|-----------|--------|
| Residual primary, path as limitation | Explicit in abstract + model_spec |
| Equal-weight rank invariance | Stated + tested |
| Rank divergence under unequal \(w\) | Existing cost-channel tables |
| No “inventory savings = residual” claim | Global wording pass |
| No “ZK implemented” claim | Title/abstract match commitment.py |
| IR / negative \(v(N)\) honest | Rates, not axioms |
| Trend tests audit-safe | Seed-level unit documented |
| Artifacts + commit hash | Deposit ready |

**Venues:** EJOR / Omega / IJPE primary after polish; POM only with strong framing; not M&SOM until theory+application deepen beyond current scope.

---

## 7. One-paragraph identity (paste into paper)

> We allocate the value of federated ridge forecasts among competing firms using **inventory-weighted residual reductions**. Equal inventory weights recover unweighted residual rankings by linearity of the Shapley value; heterogeneous cost and lead-time primitives induce material rank divergence. Under the same base-stock simulator, residual-weighted Shapley ranks do **not** match path-cost inventory-savings ranks, so residual weighting is an operationally *motivated* valuation rule rather than a proven cost identity. Reporting uses hash commit–reveal for anti-equivocation; zero-knowledge proving cost is literature-calibrated, not implemented.

---

## 8. If the agent must choose

| Situation | Action |
|-----------|--------|
| User asks to restore “inventory savings” as primary | Cite R1 gate fail; stay R2 |
| User asks for path-cost as main game | Requires new design + paper rewrite; out of current scope |
| User asks for ZK circuit | Out of scope |
| Ambiguous “operational value” | Means \(v^{\mathrm{res}}\) with weights \(w_i\), not path cost |

---

*Scope lock: **R2**. R1 residual↔path bridge closed (negative). Coding = wording/docs/stats hygiene + optional weight export; no new primary simulator campaign.*
```

---

## What coding is needed for R2

**Almost none for science.** R2 is mostly claims, docs, and analysis of artifacts you already have.

| Priority | Task | Code? |
|----------|------|--------|
| **P0** | Update `docs/model_spec.md` + `docs/claims_registry.md` (residual primary; R1 negative; withdraw savings identity) | Docs only |
| **P0** | Rename experiment/section labels in any **new** prose generators if they say “inventory savings” for \(v^{\mathrm{res}}\) | Light string fixes |
| **P1** | Trend-test script on existing `results/stage2_cost_n_sweep/` (seed-level paired, document \(N\)) | Small analysis script |
| **P1** | Optional: one helper that dumps \((c_h,c_b,L,\tau,w_i)\) next to each \(\kappa\) cell in summaries | Tiny |
| **P2** | Ensure `AGENT.md` / README point to `results/r1_residual_vs_path/` as limitation | Docs |
| **Skip** | Re-run path-cost Shapley, retune inventory, force ρ↑ | Do **not** |

### Do **not** code for R2

- New path-cost primary value function campaign  
- Homogeneity “fixes” aimed at making residual = cost  
- SNARK / deep models / large-\(n\) Shapley  

### Optional single code artifact (only if useful)

`experiments/analyze_cost_channel_trend.py` — reload existing cost-channel seed-level ρ, report paired trend + CI, write a small table for the paper. No new Monte Carlo of the full pipeline required.

---

## What you should do next (order)

1. Commit/save revised `AGENT.md` (above).  
2. Edit `model_spec.md` / claims registry: R2 language + R1 limitation row.  
3. Paper wording: title, abstract, contributions (no savings identity).  
4. Related work: SC cooperative allocation papers.  
5. Stats: fix trend-test description from existing CSVs.  
6. Reproducibility deposit.  

**Coding default answer for agents:** *no new experiment suite; documentation and analysis only unless the user explicitly opens a new design for path-cost.*