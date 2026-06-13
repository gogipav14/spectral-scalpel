# Phase 0 — Tuner Validation Report

**Date:** 2026-04-17
**Scope:** Audit the Track B nonlinear tuner ([scalpel/nonlinear/tuner.py](../scalpel/nonlinear/tuner.py)) against brute-force optima *before* building Theorem 1 on top of it.
**Deliverable for:** Track B Phase 0 (foundation audit for the nonlinear NCS companion paper).

## Executive summary

The production `algorithm2_tune` function claims a **first-principles cubic-balance law** K* = (2·M·‖u_t‖²·t³/ε_N)^{1/3} (derivation in [tuner.py:265-296](../scalpel/nonlinear/tuner.py#L265)). Empirically, this law:

- **holds for Gaussian initial conditions** (K_ratio = 1.00 median across 9 cases spanning Re 2–100 and width 0.25–1.0);
- **fails by 3×** for sharp step profiles (tanh, K_ratio median 3.17); overshoots K by a factor that grows with Re;
- **fails by 6×** for oscillatory profiles (sin×Gaussian, K_ratio median 0.17); undershoots K so severely the error is 3–4× the brute-force optimum.

**Implication for Phase 1.** The theorem as currently stated will not generalize across PDE classes or profile shapes. The issue is not the balance law itself but the `u_t_est = u_rms/tau_cross` estimator in [tuner.py:313](../scalpel/nonlinear/tuner.py#L313), which is profile-blind. Before writing Theorem 1, the `u_t` estimator must be replaced with a PDE-aware norm derived directly from the nonlinear operator.

## Checks run

All checks run on NVIDIA RTX 5060, CUDA 13.1, JAX 0.9.1 (float64), python 3.12.3.

### 1. Baseline: existing test suite passes

```
pytest nonlinear/tests/ -v
  9 passed in 0.04s
```
Tests cover: gradient-weighted operating-state estimator, Langmuir re-linearizer (pole-shift direction), GRM transfer function, 1D NL-NILT at dilute and moderate loadings. No xfail, no skip. All pass.

### 2. Existing empirical tuner (v2) reproduces: [nonlinear/scripts/burgers_tuner_v2.py](../nonlinear/scripts/burgers_tuner_v2.py)

This is the **fitted** law `K* = 2.0 · Γ^0.55` (Γ = Re_eff · t / t_shock), calibrated against a brute-force sweep on Gaussian Burgers ICs.

Result: K_tuner matches K_opt within ±2 across Re ∈ {2, 5, 10, 20, 50, 100}; L2 error is within 20% of brute-force optimum on 4 widths and 4 amplitudes. The fitted law works for Burgers Gaussians but is not derived from first principles and would not transfer to other PDEs.

### 3. Production cubic-balance tuner vs brute force on Gaussian Burgers

Script: [reports/phase0_tune_compare.py](phase0_tune_compare.py)

| ν | Re | K_cubic | ω_cubic | L2_cubic | K_opt | ω_opt | L2_opt | K_ratio | L2_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 2 | 2 | 0.70 | 3.9% | 2 | 0.90 | 2.7% | 1.00 | 1.42 |
| 0.200 | 5 | 3 | 0.70 | 4.9% | 4 | 0.90 | 3.9% | 0.75 | 1.26 |
| 0.100 | 10 | 5 | 0.70 | 5.9% | 6 | 0.80 | 5.2% | 0.83 | 1.13 |
| 0.050 | 20 | 8 | 0.70 | 7.8% | 8 | 0.80 | 7.1% | 1.00 | 1.10 |
| 0.020 | 50 | 15 | 0.70 | 12.5% | 12 | 0.80 | 11.2% | 1.25 | 1.12 |
| 0.010 | 100 | 24 | 0.70 | 21.0% | 16 | 0.70 | 15.9% | 1.50 | 1.32 |

Log-log regression of K vs Re:
- Empirical: K_opt ~ 1.62 · Re^**0.516**
- Cubic-balance: K_cubic ~ 1.16 · Re^**0.651**

The cubic-balance exponent (~0.65, close to the 2/3 derivable from the model) disagrees with the empirical exponent (~0.5, close to ½). For this problem class the tuner overshoots K, which is conservative (safer) but wastes work.

### 4. Profile-family generalization: Gaussian vs tanh vs sine

Script: [reports/phase0_profile_sweep.py](phase0_profile_sweep.py), 15 cases.

| Profile | K_ratio median | L2_ratio median | Cases |
|---|---:|---:|---:|
| Gaussian (3 widths × 3 viscosities) | **1.00** | 1.17 | 9 |
| tanh step (width 0.3) | **3.17** | 1.12 | 3 |
| sin × Gaussian | **0.17** | 3.48 | 3 |

For step profiles, `K_cubic` overshoots K_opt by 3× (e.g., K_cubic=35 vs K_opt=8 at ν=0.02) — large constant wasted. For sine profiles, `K_cubic` undershoots by 6× and the resulting L2 error is 3.5× larger than brute-force optimum — the tuner is **unsafe**, not just inefficient, on this class.

### 5. ω* is constant; the "adaptive Picard relaxation" is not active

Every case returns ω_cubic = 0.700 — the `omega_safety` default from [tuner.py:232](../scalpel/nonlinear/tuner.py#L232), saturating `min(omega_max, 1.0)`. Brute force finds ω_opt varying in {0.70, 0.80, 0.90, 0.50} across cases. The stability bound `ω < u_rms/(M·‖u_t‖²·Δt³)` that was supposed to produce Γ_NL-dependent relaxation yields ω_max >> 1 in every case tested, so the safety factor alone determines the output.

### 6. Angular-CFL alignment diagnostic runs, not yet validated against K-optimum

Script: [nonlinear/scripts/burgers_angle_precession.py](../nonlinear/scripts/burgers_angle_precession.py) runs and produces:
- Energy-weighted mean |Δθ_eff| = 55.6° for the Gaussian Burgers problem at Re≈20
- A correlation between per-mode angular deviation and per-mode error, shown in the generated figure

What is missing: a direct check that `ω_geo = cos²(Δθ_eff) × alignment` (from [angular_cfl.py](../scalpel/nonlinear/angular_cfl.py)) matches the ω_opt seen in the brute force. The diagnostic exists, is tested on Burgers, but its predictive value for Picard gating is not yet quantified.

## Gaps for Theorem 1

### Gap A — Profile-blind `u_t` estimator

**Location:** [scalpel/nonlinear/tuner.py:308-313](../scalpel/nonlinear/tuner.py#L308)

```python
u_t_est = u_rms / cross.tau_cross if cross.tau_cross < 1e30 else u_eff * u_x_max
```

This estimator treats every profile the same shape as its RMS. It's right for Gaussians (where u_rms and u_max are within a factor of 2) but catastrophically wrong for:
- step profiles, where `u_rms` is large (half the domain is near 1) but the local ‖u_t‖ is concentrated in a thin front;
- sine profiles, where u_rms underestimates the amplitude because positive and negative lobes average.

**Fix for Phase 1.** Derive ‖u_t‖ from the PDE residual at t=0 rather than from dispersion-relation time scales:
- Burgers: ‖u_t‖_∞ = ‖N(u₀) - L[u₀]‖_∞ = ‖ν·u_xx - u·u_x‖_∞
- Fisher-KPP: ‖u_t‖_∞ = ‖D·u_xx + r·u(1-u)‖_∞
- Chromatography: ‖u_t‖ = ‖binding source term‖

This is what the `nonlinearity_hessian_norm` argument originally conceptualized; the generic path needs a `rhs_evaluator` callback rather than a scalar.

### Gap B — Exponent off: predicted 2/3 vs empirical ≈ 1/2

Even on Gaussian Burgers where the ‖u_t‖ estimator is reasonable, the cubic-balance model predicts K ∝ Re^0.65 while brute force gives K ∝ Re^0.52.

Two possibilities:
1. The accumulation term ε_acc ∝ K·ε_N is wrong for the Picard-corrected cascade — it may scale as K^{1/2}·ε_N (random walk of round-off), which gives a quadratic balance K² ∝ ‖u_t‖²·t²/ε_N → K ∝ Re (too steep) OR K² ∝ ‖u_t‖·t²/ε_N → K ∝ Re^{1/2} (matches).
2. The linearization term ε_lin is not O(Δt³) per window; it may be O(Δt²) per window (one order lower), in which case the balance is quadratic, not cubic: K² ∝ ‖u_t‖²·t²/ε_N.

The *quadratic* balance law gives K ∝ ‖u_t‖·t/√ε_N, which with ‖u_t‖ ∝ Re would predict K ∝ Re — too steep. But with the correct ‖u_t‖ norm (H^0 or L^∞ directly, not dispersion-tau_cross scaled), the exponent might come out right.

**Fix for Phase 1.** Derive both ε_lin and ε_acc from per-window Taylor expansions with explicit tracking of the Picard correction's order-reduction effect. The claim "Picard removes the 1/K term leaving 1/K²" in [tuner.py:288-290](../scalpel/nonlinear/tuner.py#L288) is asserted but not proven; the proof must say under what assumptions on alignment it is true.

### Gap C — ω* is not adaptive in practice

The omega_max bound saturates above 1 for all tested cases, so `omega_safety = 0.7` alone sets the output. This means:
- The theorem cannot claim that ω* is derived from the problem — as written, it is constant.
- The *angular-CFL* formulation in [angular_cfl.py](../scalpel/nonlinear/angular_cfl.py) (ω_geo = cos²(Δθ_eff) × alignment) is separate from the tuner.py code path and has NOT been integrated into `algorithm2_tune`.

**Fix for Phase 1.** The paper's ω* claim should use the angular-CFL formula, not the stability bound in `algorithm2_tune`. Before Theorem 1 is drafted:
1. Integrate `angular_cfl.picard_gate()` into `algorithm2_tune` so ω = cos²(Δθ) · alignment from a single call.
2. Validate empirically that this integrated ω matches ω_opt from brute force on ≥3 PDE families.

### Gap D — no non-Gaussian Burgers, no non-Burgers demos have been K-validated

Current K-validation data exists only for Gaussian Burgers (Phase 0 checks 3, 4). No brute-force K-sweep has been run for Fisher-KPP, Allen-Cahn, or KS. Before Theorem 1 is stated as covering these, at minimum one-PDE-one-profile brute force is needed for each.

## Recommended Phase 1 adjustment

The Phase 1 plan in [~/.claude/plans/track-b-nonlinear-ncs-paper.md](../../../.claude/plans/track-b-nonlinear-ncs-paper.md) called for "prove Theorem 1 and derive cubic-balance as a corollary." Based on Phase 0 findings, that is premature. Revised ordering:

1. **Phase 1a — Rebuild the tuner with PDE-aware `u_t` estimator and angular-CFL ω** (2 weeks).
   Replace the profile-blind `u_rms/tau_cross` with a `rhs_evaluator` callback that computes ‖∂_t u‖ at t=0 directly from the PDE. Integrate `angular_cfl.picard_gate()` into `algorithm2_tune`. Target: K_ratio within [0.8, 1.2] on all 15 Phase 0 cases.
2. **Phase 1b — Re-run brute-force validation on the rebuilt tuner** (1 week).
   Expected exponent agreement to 5% on Burgers, tanh, sine; same for Fisher-KPP and Allen-Cahn (2 additional PDEs, 4 profiles each).
3. **Phase 1c — Derive Theorem 1 against the rebuilt tuner** (3 weeks).
   Only once the tuner demonstrably matches brute force do we write the theorem. The cubic-vs-quadratic balance question will answer itself from the empirics — the theorem should match observation, not the other way around.

This inserts one extra week into the Track B timeline but substantially de-risks Phase 5 (manuscript): reviewers will ask "does your law predict K from first principles?" and right now the answer is **no** for 2 of 3 profile families tested.

## Chromatography: out-of-scope for Track B

Track A ([nl_nilt_paper.tex](../nl_nilt_paper.tex)) owns the 1D chromatography demonstration. I did not run [nonlinear/scripts/plot_nl_chrom_1d.py](../nonlinear/scripts/plot_nl_chrom_1d.py) in Phase 0 because its outputs belong to Track A's validation, not Track B's. The `tune_chromatography` wrapper in [tuner.py:383](../scalpel/nonlinear/tuner.py#L383) also uses the same `u_t = u_rms/tau_cross` estimator and inherits Gap A; the Track A paper does not rely on this tuner's K prediction (it uses single-step re-linearization), so the gap doesn't affect Track A's claims.

## Files produced

- [reports/phase0_tune_compare.py](phase0_tune_compare.py) — Re scan, cubic-vs-empirical scaling
- [reports/phase0_profile_sweep.py](phase0_profile_sweep.py) — profile-family generalization check
- [reports/phase0_tuner_validation.md](phase0_tuner_validation.md) — this report

## Action items flowing into Phase 1

- [ ] Decide: rebuild tuner (Phase 1a) or demote cubic-balance to an empirical law in the paper.
- [ ] Implement `rhs_evaluator` callback for Burgers, Fisher-KPP, Allen-Cahn, KS.
- [ ] Integrate `angular_cfl.picard_gate()` into the main tuner entry point.
- [ ] Write a PDE-agnostic brute-force harness (currently Burgers-specific) so Phase 1b can cover all four PDEs.
- [ ] Only then attempt Theorem 1.
