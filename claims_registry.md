# Claims Registry — Route R2

| ID | Claim | Type | Evidence | Status |
|----|--------|------|----------|--------|
| C2 | Shapley allocation is efficient | Theory+test | `check_efficiency`, mechanism tests | **Done** |
| C3 | Null player receives ~0 when applicable | Theory+test | `check_null_player` | **Done** |
| C4 | No standalone sample-size argument in payment rule | Structural | `theory.size_declaration_is_unused` | **Done** |
| C19 | Equal inventory weights ⇒ residual-weighted and unweighted residual ranks coincide | Theory | Linearity of Shapley | **Done** |
| C20 | Unequal inventory-motivated weights can change ranks vs unweighted residual game | Empirical | Cost-channel \(n\in\{4,6,8\}\) tables | **Done** |
| C21 | \(C^{\mathrm{unc}}\) scales ~linearly in \(\sigma\) under Stage-0 probe assumptions | Diagnostic | Homogeneity probes M=1,2,3 | **Done** (diagnostic) |
| C-IR | IR is not guaranteed; rates reported | Empirical | `results/ir_core_diagnostics/` | **Done** |
| C-R1 | Residual-weighted Shapley tracks path-cost Shapley | Empirical | `results/r1_residual_vs_path/` | **Withdrawn** (mean ρ≈−0.17 at k1) |
| C-ZK | Full ZK proof of statistic correctness implemented | Systems | — | **Non-claim** |
| C-IC | Universal strategy-proofness | Theory | — | **Non-claim** |

### Withdrawn / limitation

- “Inventory savings” as primary characteristic function identity with path cost: **withdrawn** after R1 gate failure.
- Use wording: **inventory-weighted residual value**.