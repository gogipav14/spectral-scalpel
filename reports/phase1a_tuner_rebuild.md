# Phase 1a — Tuner Rebuild Report

**Date:** 2026-04-18
**Scope:** Replace profile-blind `u_t_est = u_rms/tau_cross` with a PDE-aware estimator, and integrate the angular-CFL ω formula into the main tuner.
**Deliverable target (from [phase0_tuner_validation.md](phase0_tuner_validation.md)):** K_ratio ∈ [0.8, 1.2] across the 15 Phase 0 cases.
**Outcome:** partial success — robustness restored, Re-scaling still wrong.

## Summary table (K_ratio = K_tuner / K_bruteforce_optimum)

| Profile family | n cases | v1 (Phase 0) | v2 (L∞) | v2 (L²)  |
|---|---:|---:|---:|---:|
| Gaussian | 9 | median **1.00**, range [0.75, 1.50] | 2.50 [0.80, 18.3] | 0.83 [0.40, 5.00] |
| tanh step | 3 | median **3.17**, range [2.33, 4.38] | 3.00 [2.38, 7.33] | **0.83** [0.62, 2.00] |
| sine × Gaussian | 3 | median **0.17**, range [0.10, 0.25] | 0.92 [0.50, 1.50] | **0.50** [0.30, 0.75] |

## What changed

1. **PDE-aware `rhs_evaluator`** ([tuner.py:`algorithm2_tune_v2`](../scalpel/nonlinear/tuner.py)): the caller now supplies `rhs(u, x) -> ∂_t u` computed from the actual PDE at t=0. The tuner uses `‖∂_t u‖_{L²}` directly instead of the profile-blind `u_rms/τ_cross`.

2. **Angular-CFL ω integration**: `algorithm2_tune_v2` now calls `angular_cfl.angular_cfl_analysis` (via lazy import to avoid circular dependency) and uses its `omega_angular = cos²(Δθ_eff)` output as ω*. This replaces the v1 stability bound that always saturated at the `omega_safety = 0.7` default.

3. **PDE-specific wrappers** added: `tune_burgers_v2`, `tune_fisher_kpp_v2`, `tune_allen_cahn_v2`. Fisher-KPP automatically forces `omega_max = 0` since its Picard residual is anti-aligned.

4. **Exposed RHS builders**: `burgers_rhs`, `fisher_kpp_rhs`, `allen_cahn_rhs`, `ks_rhs` for reuse in tests and demos.

## The two fixes that worked

- **L² norm on the nonlinear drift**, not L∞. L∞ was over-weighting sharp features (Gaussian at σ=0.25, tanh step) and blowing up K by 5-18×. L² averages over the domain and brought K_ratios back into the same order as K_opt.

- **`angular_cfl_analysis` for ω**. The v1 stability bound `ω < ‖u‖/(M‖u_t‖²Δt³)` always saturated at 1.0 (so `ω* = 0.7` constant). The angular formula `ω = cos²(Δθ_eff)` actually varies with Γ_NL: at Re=2 it emits ω=0.46, at Re=100 it emits ω=0.90.

## The one thing that is still wrong: Re-scaling

| Source | K(Re) scaling |
|---|---|
| Brute force (ground truth) | K_opt = 1.62·Re^{**0.52**} |
| v1 (cubic-balance with τ_cross estimator) | K_v1 = 1.16·Re^{+0.65} |
| v2 L² (this report) | K_v2 = 10.06·Re^{**−0.14**} |

v2 gives approximately **constant K across Re** (weak negative slope). v1 gave the right sign with the wrong exponent (+0.65 vs +0.52). Brute force wants K to grow substantially with Re (2 → 16 as Re goes 2 → 100).

**Why v2 has the wrong sign.** The nonlinear drift `N[u] = u·u_x` in Burgers is ν-independent, so ‖N[u]‖_{L²} does not grow with Re. The only Re-dependent input to the cubic balance is the operating-state `u_eff` (barely changes), and the angular-CFL ω actually *rises* with Re (making the correction more effective, reducing required K). So v2 predicts K drops slightly with Re — opposite to observation.

**What the data is telling us.** K_opt grows with Re for a reason not in the current cubic-balance model. Candidates:
1. **NILT tolerance ε_N is Re-dependent.** At high Re the per-mode stiffness σ_NILT grows, forcing smaller achievable ε_N even at fixed Bromwich size. If ε_N ∝ Re^{-1}, then K* ∝ (1/ε_N)^{1/3} ∝ Re^{+0.33}, which combined with angular ω-growth could land closer to +0.52.
2. **Picard effectiveness drops as shocks sharpen.** The angular-CFL formula predicts ω rises with Re (because δ_cross shrinks), but the empirical Picard effectiveness (reduction of linearization error per window) may drop at high Re because `alignment` drops (shocks break the advective alignment). The current code *ignores* the alignment sign in `ω_geo = cos²(Δθ)`; incorporating it would reduce ω at high Re, forcing K to rise.
3. **Linearization error accumulates super-linearly when residual aligns.** The claim "Picard makes ε_lin ∝ 1/K²" is an asymptotic statement; at finite K the actual reduction factor depends on how many Picard iterations per window (currently 1) and on alignment.

## Implication for Theorem 1

The cubic-balance **structural form** K³ ∝ M·‖u_t‖²·T³/ε_N is consistent with v1, v2, and the Re-scaling data — the issue is the **inputs** (what ‖u_t‖ to use, what ε_N to use). The formula is not *wrong*; it is *incomplete* without a Re-aware ε_N.

The theorem as drafted in the Phase 1 plan claims a clean cubic balance with a fixed ε_N. Before writing it:
- **Phase 1b must characterize ε_N(Re)** empirically across all 4 target PDEs.
- The theorem then takes the form K³ ∝ M·‖u_t‖²·T³/ε_N(Re), with ε_N(Re) as a measured quantity or derived from the per-mode feasibility bound of the linear paper ([spectral_factorization.tex eq. 3](../paper/spectral_factorization.tex)).
- If ε_N(Re) is not clean (e.g., system-dependent), the theorem demotes to: "K* is the unique integer minimizer of the empirical cost function ε_lin(K) + ε_acc(K); closed-form approximation is K ≈ (2M‖u_t‖²T³/ε_N)^{1/3} up to Re-dependent corrections of O(1)."

## Honest assessment

v2 is a strict improvement over v1 on robustness (eliminates the catastrophic sine-profile underperforming) but not a clean win: it retreats on Gaussian scaling and gets the Re-dependence wrong. We do not yet have a tuner accurate enough to serve as the basis of a rigorously stated theorem. **The cubic-balance law as published would survive review only as an empirical approximation, not a first-principles result.**

This is what Phase 0 was designed to catch. Fixing it properly requires Phase 1b (below), which adds about 1 week to the Track B schedule but avoids a theorem that reviewers would shred.

## Phase 1b adjustment (proposed)

Instead of moving straight to the theorem, Phase 1b (1 week) becomes:

1. **Measure ε_N(Re)** empirically for each of the 4 target PDEs (Burgers, Fisher-KPP, Allen-Cahn, KS). Protocol: run the linear NILT substep with known forcing, measure per-window relative error, report vs Re/stiffness.
2. **Measure ε_lin(K) at fixed Re** (rather than K_opt) to determine the actual Picard order-reduction. Does it go as 1/K^α with α=2 (current claim) or α=1.5 (weaker Picard)?
3. **Incorporate alignment sign into ω_angular.** The code path currently computes alignment but does not feed it into `omega_geo = cos²(Δθ)`. Including it as `ω = alignment × cos²(Δθ)` will zero Picard for anti-aligned residuals (Fisher-KPP) automatically.
4. **Re-run the validation harness** with the alignment-corrected ω and measured ε_N(Re). Target: K_ratio ∈ [0.8, 1.2] on all 15 Phase 0 cases AND matching Re-scaling exponent to within 10%.

Only after these four steps pass should Theorem 1 be drafted (Phase 1c).

## Files produced/modified

- [scalpel/nonlinear/tuner.py](../scalpel/nonlinear/tuner.py) — added `algorithm2_tune_v2`, `burgers_rhs/fisher_kpp_rhs/allen_cahn_rhs/ks_rhs`, `*_residual`, `tune_burgers_v2/tune_fisher_kpp_v2/tune_allen_cahn_v2`
- [reports/phase1a_validate_v2.py](phase1a_validate_v2.py) — validation harness
- [reports/phase1a_tuner_rebuild.md](phase1a_tuner_rebuild.md) — this report

## Action items into Phase 1b

- [ ] Instrument the per-window NILT to report ε_N as a function of (Re, stiffness, ε_tail tolerance).
- [ ] Fit a Re-scaling to ε_N(Re) per PDE.
- [ ] Modify `angular_cfl.angular_cfl_analysis` to multiply `omega_geo` by `alignment` (clipped at 0 from below).
- [ ] Re-run `phase1a_validate_v2.py` with these fixes and update this report.
- [ ] Decide: if K_ratio still misses [0.8, 1.2], demote cubic balance to empirical approximation in the theorem wording.
