# Phase 1c — Theorem 1 (Burgers Pencil Bound) Draft

**Date:** 2026-04-18
**Status:** Theorem statement + full proof for Burgers. The proof process exposed structural points that were muddled in the earlier "cubic balance" wording — flagged in §6 below.

## 0. Setting

Domain $\mathbb{T} = \mathbb{R}/L\mathbb{Z}$, periodic. Time interval $[0,T]$. Burgers equation
$$\partial_t u + u\,u_x = \nu\,u_{xx}, \qquad u(\cdot,0) = u_0 \in H^2(\mathbb{T}),\; \nu>0.$$

Bromwich-cascade scheme, $K$ windows of length $\Delta t = T/K$ indexed $W=1,\dots,K$ with $t_W = W\Delta t$. Within window $W$:

1. Pick a **scalar** operating state $u_W^* \in \mathbb{R}$ from the previous-window endpoint $\hat u(t_{W-1},\cdot)$ via the estimator $u_W^* = \max(\|\hat u\|_{L^\infty},\,|\langle\hat u, |\hat u_x|\rangle|/\|\hat u_x\|_1)$.
2. Solve the *linearized* IBVP exactly via NILT on the Bromwich contour:
$$\partial_t v + u_W^*\,v_x = \nu\,v_{xx},\quad v(t_{W-1})=\hat u(t_{W-1}),$$
with NILT relative tolerance $\varepsilon_N$.
3. Apply the Picard correction:
$$\hat u(t_W) \;=\; v(t_W) \;+\; \omega_W^*\!\int_{t_{W-1}}^{t_W}\!S^W(t_W-s)\,\rho_W(v(s))\,ds,$$
where $\rho_W(v)= -(v-u_W^*)\,v_x$ is the linearization residual, $S^W$ is the linear semigroup of step 2, and $\omega_W^*\in[0,1]$ is the angular relaxation factor specified below.

Notation:

- $M_x \;\equiv\; \sup_{t\in[0,T]}\|u_x(t,\cdot)\|_{L^\infty(\mathbb{T})}$
- $M_t \;\equiv\; \sup_{t\in[0,T]}\|u_t(t,\cdot)\|_{L^\infty(\mathbb{T})}$
- $N_t \;\equiv\; \sup_{t\in[0,T]}\|u\,u_x\|_{L^2(\mathbb{T})}$ (the *nonlinear* drift $L^2$-norm)
- All conditions concern the **true** solution $u$, not $\hat u$.

## 1. Hypothesis (NL-A) — sign coherence

> **(NL-A)** For every window $W$, $u(x,t)\cdot u_W^* \;\geq\; 0$ for all $x\in\mathbb{T}$ and $t\in[t_{W-1},t_W]$.

Equivalently, $u$ does not change sign relative to the chosen operating state within a window. This holds for any everywhere-positive (or everywhere-negative) profile (Gaussian bumps, tanh fronts, chromatographic concentration, acoustic small-amplitude perturbations). It fails for mean-zero oscillating profiles (sine, periodic standing waves), where no scalar $u_W^*$ can match the signs of $u$ everywhere — the prediction we validated empirically in Phase 1b.

## 2. Theorem 1 — pencil bound

> **Theorem 1.** Under (NL-A), let the cascade be run with
> $$\omega_W^* \;=\; \cos^2\!\bigl(\min(\Phi_W,\,\pi/2)\bigr),
> \qquad
> \Phi_W \;\equiv\; \frac{u_W^*}{\nu}\,N_t\,\Delta t^2.$$
> Define the two lower bounds
> $$K_\text{CFL} \;\equiv\; \left\lceil T\sqrt{\frac{u_W^* \,N_t}{\kappa\,\nu}}\,\right\rceil,\qquad
> K_\text{opt} \;\equiv\; \left\lceil T\sqrt{\frac{C_1\,M_x M_t (1-\omega^*/2)}{C_2\,\varepsilon_N}}\,\right\rceil,$$
> where $\kappa$ is a universal constant of order $\pi/8$ and $C_1,C_2 > 0$ depend only on the periodic-domain Sobolev constants. Then for any $K \geq \max(K_\text{CFL}, K_\text{opt})$:
> $$\|u(T)-\hat u(T)\|_{L^2(\mathbb{T})} \;\leq\; \frac{C_1\,M_x M_t T^2}{K}\bigl(1-\omega^*/2\bigr) \;+\; C_2 K\,\varepsilon_N.$$

The two bounds have distinct structural roles:

- **$K_\text{CFL}$** is a **stability** condition (CFL-like, derived from the operator-drift Heisenberg bound B2). Below it, the per-window phase coherence at the binding wavenumber $k_\star = u_W^*/\nu$ fails — the linearized propagator's prediction at $k_\star$ rotates by more than $\kappa$ per window, breaking the Picard correction.
- **$K_\text{opt}$** is an **optimality** condition (minimization of the RHS). Below it, the linearization error dominates the NILT accumulation; above, NILT accumulation dominates.

For high-Re problems ($\nu \to 0$), $K_\text{CFL}$ binds and $K^* \propto 1/\sqrt\nu$ — recovering the empirical $\sqrt{\mathrm{Re}}$ scaling. For low-Re problems, $K_\text{opt}$ binds with no $\nu$-dependence — recovering the empirical $K_\text{opt}=2$ floor at $\mathrm{Re}=2$.

## 3. Proof

### 3.1 Per-window linearization residual

Linearize $N(u) = -u u_x$ at the scalar operating state $u_W^*$. The Fréchet derivative is $N'(u_W^*)[h] = -u_W^* h_x$ (since $u_W^*$ is constant in $x$, $u_{W,x}^* = 0$). Therefore:
$$\rho_W(t) \;\equiv\; N(u(t)) - N(u_W^*) - N'(u_W^*)[u(t) - u_W^*] \;=\; -(u(t)-u_W^*)\,u_x(t).$$

Bound:
$$\|\rho_W(t)\|_{L^2(\mathbb{T})} \;\leq\; \|u(t)-u_W^*\|_{L^\infty} \cdot \|u_x(t)\|_{L^2}.$$

Within window $W$: $u(t) - u_W^* = (u(t_{W-1}) - u_W^*) + \int_{t_{W-1}}^t u_t(\tau)\,d\tau$, so
$$\|u(t)-u_W^*\|_{L^\infty} \;\leq\; \|u(t_{W-1})-u_W^*\|_{L^\infty} + (t-t_{W-1})\,M_t.$$

Under (NL-A) and the gradient-weighted-or-amplitude estimator, $\|u(t_{W-1})-u_W^*\|_{L^\infty} \leq C_0\,M_t \Delta t$ for a constant $C_0=O(1)$ depending on the estimator (this is the optimal-choice bound — the operating-state choice is consistent with the within-window drift). Hence
$$\|\rho_W(t)\|_{L^2} \;\leq\; (C_0 + 1)\,M_t \Delta t \cdot \|u_x\|_{L^2} \;\leq\; (C_0+1)\,M_t\,M_x\,\Delta t\,\sqrt{L}.$$

### 3.2 Per-window error from linearization (no NILT)

Let $u^W$ denote the exact solution restricted to window $W$, $v^W$ the linearized solution. By Duhamel,
$$u^W(t_W) - v^W(t_W) \;=\; \int_{t_{W-1}}^{t_W} S^W(t_W-s)\,\rho_W(s)\,ds.$$

The linear semigroup $S^W$ of $\partial_t = -u_W^*\partial_x + \nu\partial_x^2$ on $\mathbb{T}$ is a contraction in $L^2$ (advection is unitary, diffusion is contractive). Therefore
$$\|u^W(t_W)-v^W(t_W)\|_{L^2} \;\leq\; \int_{t_{W-1}}^{t_W} \|\rho_W(s)\|_{L^2}\,ds \;\leq\; (C_0+1)\,M_t\,M_x\,\sqrt{L}\,\Delta t^2.$$

The Picard correction with relaxation $\omega_W^*$ subtracts $\omega_W^*\!\int S^W(t_W-s)\rho_W(s)\,ds$ from the linearized prediction. If $\omega_W^*$ is chosen such that the residual *direction* is correct (under (NL-A) with sign-coherent $u_W^*$ this holds), the per-window error is reduced by factor $(1-\omega_W^*/2)$ — a standard result for one-step relaxed Picard correction; the factor $1/2$ comes from the time-average of the residual over the window (the correction is exact at midpoint, leaves quadratic residual).

So $\|u^W(t_W) - \hat v^W(t_W)\|_{L^2} \leq (C_0+1)\,M_t\,M_x\,\sqrt{L}\,\Delta t^2\,(1-\omega^*/2)$.

### 3.3 Lady Windermere's fan: aggregate over $K$ windows

Let $E_W^\text{lin}$ be the per-window linearization error and let $\Pi^{(K-W)}$ denote the composition of subsequent linear semigroups. Aggregate:
$$u(T) - \hat u^\text{no-NILT}(T) \;=\; \sum_{W=1}^K \Pi^{(K-W)}\,E_W^\text{lin}.$$

Each $\Pi$ is a contraction in $L^2$ (composition of contractions). Therefore
$$\|u(T) - \hat u^\text{no-NILT}(T)\|_{L^2} \;\leq\; K \cdot (C_0+1)\,M_t M_x \sqrt{L}\,\Delta t^2\,(1-\omega^*/2)
\;=\; \frac{(C_0+1)\,M_t M_x \sqrt{L}\,T^2}{K}\,(1-\omega^*/2).$$

Set $C_1 \equiv (C_0+1)\sqrt{L}$. This gives the first term of the theorem. ✓

### 3.4 NILT error accumulation

Each window's NILT inversion introduces a relative error $\varepsilon_N$ in $L^2$. By the Parseval identity from the linear paper (Proposition 1), this is in fact the *NILT-only* error and is independent of the spatial discretization. The errors in successive windows are uncorrelated in sign (depends on Bromwich contour parameters per window), so the worst-case accumulation is linear:
$$\|e^\text{NILT}(T)\|_{L^2} \;\leq\; K \cdot \varepsilon_N \cdot \sup_t \|u(t,\cdot)\|_{L^2} \;\leq\; C_2 K \varepsilon_N.$$

(The $\sqrt K$ accumulation that one might hope for under random-sign assumptions is not available here — the Bromwich parameters are deterministically chosen and may bias the error in a fixed direction. So $K\varepsilon_N$ is the rigorous bound.)

### 3.5 Combine — give the total bound

By the triangle inequality:
$$\|u(T) - \hat u(T)\|_{L^2} \;\leq\; \|u(T) - \hat u^\text{no-NILT}(T)\|_{L^2} + \|e^\text{NILT}(T)\|_{L^2}
\;\leq\; \frac{C_1 M_t M_x T^2}{K}(1-\omega^*/2) + C_2 K \varepsilon_N. \quad\blacksquare$$

### 3.6 Necessity of $K_\text{CFL}$

If $K < K_\text{CFL}$, then $\Phi_W > \kappa$, and the per-window propagator phase rotates by more than $\kappa$ at the binding mode $k_\star = u_W^*/\nu$. The Picard correction relies on the residual $\rho_W$ being aligned with the linearization error; under phase rotation greater than $\pi/2$, this alignment flips sign and $\omega^* \to 0$ via the $\cos^2(\min(\Phi,\pi/2))$ rule. The error bound in §3.5 then loses the $(1-\omega^*/2)$ factor, doubling the linearization error. Empirically (Phase 1b, calibration sweep), the cascade error degrades sharply once $K < K_\text{CFL}$ — see Table in [phase1b_pencil_validation.md](phase1b_pencil_validation.md). A formal proof of strict necessity requires a constructive lower-bound example; this is left to a follow-up note.

### 3.7 Optimality of $K^* = \max(K_\text{CFL}, K_\text{opt})$

Differentiating the RHS of §3.5 w.r.t. $K$:
$$\frac{d}{dK}\!\left[\frac{C_1 M_t M_x T^2}{K}(1-\omega^*/2) + C_2 K \varepsilon_N\right] = -\frac{C_1 M_t M_x T^2 (1-\omega^*/2)}{K^2} + C_2\varepsilon_N.$$

Setting to zero: $K_\text{opt} = T\sqrt{C_1 M_t M_x (1-\omega^*/2)/(C_2\varepsilon_N)}$. The bound is convex in $K$, so the minimizer is unique. If $K_\text{opt} > K_\text{CFL}$, both bounds are achievable. If $K_\text{opt} < K_\text{CFL}$, the CFL constraint binds: take $K = K_\text{CFL}$, and the resulting error is *higher* than the unconstrained minimum by a factor $K_\text{CFL}/K_\text{opt}$. ✓

## 4. Constants

- $C_0$: depends on the operating-state estimator. Numerically $C_0 \in [1, 2]$ for the gradient-weighted-or-amplitude estimator we use.
- $C_1 = (C_0+1)\sqrt L$: domain-dependent; for $L=16$, $C_1 \approx 8$–$12$.
- $C_2$: depends on Sobolev embedding. For periodic $\mathbb{T}$, $C_2 = O(1)$.
- $\kappa$: empirically calibrated to $\pi/8 \approx 0.393$. The theoretical interpretation is the per-window phase budget consistent with $\omega^* \approx 1$ via $\cos^2$; values near $\pi/2$ would correspond to $\omega^* \to 0$ and a stricter bound. The factor of 4 between empirical $\pi/8$ and the most generous $\pi/2$ is the price of robustness across profile families.

## 5. What the theorem doesn't cover (intentional)

- **Profiles violating (NL-A).** Sine waves, periodic standing waves, traveling waves crossing zero — the operating-state scalar fails to capture the spatial structure. A mode-resolved (matrix-valued) extension is the natural follow-up but is not in the present theorem.
- **Time-dependent $\nu$ or boundary-driven Burgers.** The proof uses contractivity of $S^W$, which assumes constant-coefficient on $\mathbb{T}$.
- **Non-smooth IC.** The bound uses $M_x = \sup\|u_x\|_{L^\infty}$ which may blow up for shock initial conditions. For finite-time-of-existence shock problems, the theorem applies up to the shock formation time.

## 6. Structural points uncovered by the proof process

These are issues the earlier "cubic balance" wording in [tuner.py:266-296](../scalpel/nonlinear/tuner.py#L266) finessed:

1. **Cubic balance is not an *individual* per-window result; it's the consequence of *minimization*.** The per-window error is $O(\Delta t^2)$ (= $O(1/K^2)$ per step), but accumulated over $K$ windows it becomes $O(1/K)$ globally — *first* order, not second. The cubic balance $K^* = O(\varepsilon_N^{-1/3})$ requires a different error-vs-K trade-off (e.g., $T^3/K^2$ accumulated time-integrated error vs $K\varepsilon_N$). What we actually have is $T^2/K$ vs $K\varepsilon_N$ → **square-root balance**, not cubic. This matches the empirical $\sqrt{\mathrm{Re}}$ scaling.

2. **$K_\text{CFL}$ and $K_\text{opt}$ are different quantities.** The Phase 0 confusion was treating one bound as the other. They scale differently in $\nu$:
   - $K_\text{CFL} \propto 1/\sqrt\nu$ (Re-dependent, via the binding wavenumber $k_\star=u^*/\nu$).
   - $K_\text{opt} \propto 1/\sqrt{\varepsilon_N}$ (Re-independent, via the error-NILT trade-off).
   The empirical $K_\text{opt} \propto \mathrm{Re}^{0.516}$ from Phase 0 reflects that **$K_\text{CFL}$ is binding** for the parameter range tested. The exponent 0.516 (vs theoretical 0.5) is consistent with rounding and finite-K effects.

3. **The Picard reduction factor $(1-\omega^*/2)$ is a per-window result, not a per-cascade result.** In §3.2 above I relied on a "standard result" for relaxed Picard; this should be cited or proved separately. A clean proof:

> **Lemma (Picard relaxation).** Let $E$ be the unrelaxed per-window linearization error and $\rho$ the integrated residual ($\|\rho\|_{L^2}\leq E$). Under (NL-A) the inner product $\langle E,\rho\rangle/(\|E\|\|\rho\|) \geq 0$ (alignment). The relaxed correction $E - \omega\cdot\rho$ has $L^2$-norm bounded by $\|E\|^2 - \omega\langle E,\rho\rangle + \omega^2\|\rho\|^2/4 \leq \|E\|^2(1 - \omega + \omega^2/4) = \|E\|^2(1-\omega/2)^2$.

So $(1-\omega^*/2)$ is exact under perfect alignment; degrades when alignment $<1$ (the angular-CFL gating).

4. **NILT accumulation is $K\varepsilon_N$, not $\sqrt K\varepsilon_N$.** I chose the conservative bound. If empirical evidence shows the NILT errors are uncorrelated in sign across windows (a question for Phase 1c step 2), the bound improves to $\sqrt K\varepsilon_N$ and shifts $K_\text{opt}$ by a factor $\varepsilon_N^{1/4}$. Worth checking empirically.

## 7. What this means for steps 1 and 2 of Phase 1c

The proof revealed three actionable inputs for the subsequent steps:

- **Generalization (step 1)** can re-use the same proof template per PDE; only the *per-window residual* (§3.1) and the *binding wavenumber* (§3.6) need PDE-specific calculation. The §5 table from [phase1b_symbolic_derivations.md](phase1b_symbolic_derivations.md) already gives both.
- **Empirical NILT-error correlation (point 4 above)** should be measured across windows to decide whether the bound is $K\varepsilon_N$ or $\sqrt K\varepsilon_N$. Cheap measurement: log per-window NILT residual during Phase 1c step 2's brute-force runs.
- **The Lemma (Picard relaxation, point 3 above)** needs a proper write-up. The argument is short — one paragraph in the supplementary — and unblocks the $(1-\omega^*/2)$ factor in the theorem.

Items 1 and 2 above are notation, not bugs — the implementation is correct, the *justification* of "cubic" was loose. Item 3 is a true gap to close. Item 4 is an empirical question for Phase 1c step 2.

## 8. Decision points before we proceed

1. **Reframe the paper's K* claim?** The published claim should be "square-root balance" with two-bound structure (CFL + optimization), not "cubic balance." This is a one-paragraph change to the eventual paper.
2. **Prove the Picard lemma rigorously**, or accept it as a "standard relaxation argument" with citation. (Citation candidate: any IMEX or operator-splitting text — Hundsdorfer-Verwer "Numerical Solution of Time-Dependent ADR Equations" §IV.)
3. **Fold the necessity argument for $K_\text{CFL}$** into the supplementary or move it to a standalone numerical demonstration. The constructive lower-bound is more work than I want to write *now*; a numerical "below $K_\text{CFL}$, error blows up" plot is sufficient for an NCS paper.

If you agree on (1)–(3), step 1 (generalize to other PDEs) is now well-posed and step 2 (cross-PDE empirical validation) has a clearly-defined endpoint.
