# Paper 2 (NC submission) — Pre-submission Plan

**Target venue:** Nature Communications. **Framing:** "Gauge-Transformed Spectral Cascades for Unified Nonlinear PDE Integration."

## Five tasks + dependencies

```
Task 1 (proof tightening) ────┐
Task 2 (regularization thm) ──┼──> Task 5 (manuscript)
Task 3 (2D validation) ───────┤
Task 4 (code polish) ─────────┘
```

Tasks 1–4 can be worked in parallel (Task 2 depends partially on Task 1's notation). Task 5 (manuscript) depends on all four.

## Task 1 — Theorem 1 proof tightening

**Location:** [reports/theorem1_unified_formal.md](theorem1_unified_formal.md) §3.2, §3.3.

**What's missing:**
- §3.2 (Picard residual bound) has a claim without rigorous bilinear-form analysis. Need explicit Sobolev-space argument.
- §3.3 (2-iter Picard reduction) has a hand-wavy "each iteration reduces by Δt·L_Picard" — needs a Banach fixed-point argument with explicit contraction constant.

**Deliverable:** [reports/theorem1_proof_v2.md](theorem1_proof_v2.md) with two tightened lemmas:

- **Lemma B' (tight Picard residual bound):** $\|\tilde N[w_0^* + h, \cdot]\|_{L^2} \leq \tfrac{1}{2}\|\tilde N''\|_\infty\,\|h\|_{H^1}^2 + O(\|h\|_{H^1}^3)$, with $\|\tilde N''\|_\infty$ explicitly computed per PDE (table of values).

- **Lemma E' (tight 2-iter reduction):** let $P_W(v) = v_\text{lin} + \omega\int S^W\tilde N[v]$ be the Picard operator. Show it is a contraction on a ball $B_R(v_\text{lin})$ of radius $R = O(\Delta t)$ with contraction constant $L_P \leq C\Delta t$. Then two iterations give $\|P_W^2(v_\text{lin}) - v^\text{true}\| \leq L_P^2 R \leq C^2\Delta t^3$. Aggregate: $O(T^3/K^2)$.

**Estimated effort:** 3–4 pages of math, 1 week.

## Task 2 — Regularization theorem

**Location:** [reports/theorem1_unified_formal.md](theorem1_unified_formal.md) §5.

**What's stated:** Theorem 1$^\varepsilon$ with bound $\|u-\hat u\| \leq C T^3/K^2 + K\varepsilon_N + O(\varepsilon_\text{reg})$.

**What's missing:** The regularized gauge $\Phi_\varepsilon$ defines a *different* PDE in $w$-space than the exact gauge. Need:
- Prove $\|\Phi_\varepsilon^{-1}(w_\varepsilon) - \Phi^{-1}(w)\|_{L^2} \leq C\varepsilon_\text{reg}$ for matching initial data.
- Prove the cascade on the regularized problem preserves Theorem 1's hypothesis (G).
- Assemble: $\|u - \hat u\| \leq \|u - u_\varepsilon\| + \|u_\varepsilon - \hat u_\varepsilon\| = O(\varepsilon_\text{reg}) + \text{Theorem 1 bound}$.

**Deliverable:** [reports/theorem1_eps_proof.md](theorem1_eps_proof.md) with the three-step argument.

**Estimated effort:** 2 pages of math, 1 week.

## Task 3 — 2D empirical validation

**Critical for NC.** The paper needs at least one 2D result to avoid "limited to 1D" reviewer objections. NC reviewers expect broad applicability.

**Options:**
- **2D Burgers** $\partial_t u + \vec{u}\cdot\nabla u = \nu\nabla^2 u$ (vector-valued, harder).
- **2D Fisher-KPP** $\partial_t u = D\nabla^2 u + ru(1-u)$ (scalar, simpler).
- **2D Allen-Cahn** $\partial_t u = \varepsilon^2\nabla^2 u + u - u^3$.

**Recommendation: 2D Fisher-KPP via logit gauge.** Simplest extension, single-component, demonstrates the gauge principle in multi-dimensional setting. Use a radially-symmetric IC plus an asymmetric perturbation.

**Deliverable:**
- [scalpel/nonlinear/unified_cascade_2d.py](../scalpel/nonlinear/unified_cascade_2d.py) — 2D cascade with 2D FFT in the linear substep and 2D gradient operators in the residual.
- [reports/theorem1_2d_fkpp_validation.py](theorem1_2d_fkpp_validation.py) — slope-2 confirmation for 2D.

**Prediction:** Should give slope 2.00 identically to 1D (theorem is dimension-agnostic). If the 2D test fails, we need to investigate — possibly indicates a subtle dimension-dependent issue.

**Estimated effort:** ~2 weeks (code + validation + possible debug).

## Task 4 — Code polish for publication

**Location:** [scalpel/nonlinear/unified_cascade.py](../scalpel/nonlinear/unified_cascade.py) currently has `@dataclass` specs and is structurally clean, but lacks:
- Complete docstrings with math formulas for each function.
- Type hints throughout.
- Regression tests in `tests/test_unified_cascade.py`.
- Example notebooks (Burgers, Fisher-KPP, Allen-Cahn, KS).
- Performance benchmarks.

**Deliverable:** Final polished module + 4 example notebooks + regression tests + benchmark table.

**Estimated effort:** 1 week.

## Task 5 — NC manuscript

**Structure (NC format, ~5000 words main text + extended data figures):**

1. **Introduction (~800 words).** The unification narrative: diverse nonlinear PDEs share a hidden structural commonality. Gauge transforms expose it. Spectral scalpel framework (Paper 1) extends cleanly via this commonality.

2. **Results (~2500 words).** Four-PDE empirical demonstration with slope-2 figure as the centerpiece. 2D validation to show it extends beyond 1D. Comparison to Strang splitting, ETDRK4, and IMEX-BDF showing 2–3× better accuracy at equivalent cost.

3. **Theoretical framework (~1000 words, condensed).** Hypothesis (G), Theorem 1 statement, cubic-balance tuner. Full proof relegated to Methods and Supplementary.

4. **Discussion (~500 words).** What PDE classes fit under (G) and which don't. Limitations (singular-limit regularization). Open problems (extending to systems, 3D, non-autonomous).

5. **Methods.** Full proof of Theorem 1, Theorem 1$^\varepsilon$, commutator constant computation per PDE, algorithmic details.

6. **Supplementary.** Additional empirical results (13-parameter-grid-sweep if time), code availability, reproducibility scripts.

**Extended Data figures (NC allows 10):**
- Fig. 1: Unification diagram — four PDEs, four gauges, one algorithm.
- Fig. 2: Slope-2 convergence plot across all 4 PDEs.
- Fig. 3: Error at fixed K (K=64) vs PDE parameter.
- Fig. 4: 2D validation — snapshot of Fisher-KPP front propagation.
- Fig. 5: Comparison with Strang / ETDRK / IMEX baselines.
- Fig. 6: Algorithm flowchart (pseudocode).

**Estimated effort:** 4–6 weeks for first draft.

## Timeline

**Week 1:** Task 1 (theorem proof tightening).
**Week 2:** Task 2 (regularization theorem).
**Week 3–4:** Task 3 (2D validation).
**Week 5:** Task 4 (code polish).
**Week 6–10:** Task 5 (manuscript, first draft + revisions).
**Week 11–12:** Internal red-team review (same methodology as Paper 1's Phase 0).
**Week 13:** Submit to NC.

**Total: ~13 weeks = 3 months** to submission.

## Order of execution

**Start with Task 1 (theorem tightening).** It's the foundation the others depend on (Task 2 uses its notation, Task 5 cites its theorems). Complete it fully before moving to Task 2.

Task 3 (2D) can be started in parallel with Task 1 since it doesn't depend on proof details — it just needs the algorithm to work.

Task 4 (code polish) can be deferred until end; needs all theoretical details locked in.

Task 5 (manuscript) is last; needs everything else in place.

## Why NC and not SISC?

SISC is the safer, more natural venue for this work technically. NC is the bigger swing. The user's call to aim for NC is reasonable given:

1. The four-PDE unification narrative has the kind of breadth NC likes.
2. The gauge-transform principle is conceptually clean — easy to explain to a non-specialist.
3. The companion-paper structure (Paper 1 at NCS) provides editorial continuity.
4. Lower risk of obscurity — NC visibility is significantly higher than SISC.

Risk: NC is pickier about "methodological novelty" and may reject for "just another spectral scheme." Mitigation: lead with the gauge-unification angle, not with the algorithm.

If NC rejects, fallback to SISC or JCP is straightforward (~1 month reformat).

---

Beginning Task 1 now.
