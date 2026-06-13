# Phase 1c step 2 — Fisher-KPP validation finding: the scalar-linearization cascade has a fundamental accuracy floor

## Empirical result

Run of [reports/phase1c_validate_fkpp.py](phase1c_validate_fkpp.py) on a 0.3-amplitude Gaussian IC with $D\in\{0.01, 0.05, 0.1\}$ and $r\in\{0.5, 1, 2, 5, 10\}$, $T=2$:

| $D$ | $r$ | $K_\text{tuner}$ | $L^2_\text{tuner}$ | $K_\text{opt}$ | $L^2_\text{opt}$ | $K_\text{ratio}$ | $L^2_\text{ratio}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.5 | 2 | 14% | 32 | 1.7% | 0.06 | 8.31 |
| 0.01 | 1.0 | 4 | 49% | 32 | 13% | 0.12 | 3.77 |
| 0.01 | 2.0 | 7 | 133% | 32 | 91% | 0.22 | 1.45 |
| 0.01 | 5.0 | 16 | 118% | 3 | 117% | 5.33 | 1.01 |
| 0.01 | 10.0 | 31 | 77% | 6 | 77% | 5.17 | 1.00 |

Three separate failure modes:

- **Low r ($r \leq 1$):** tuner under-predicts K by ~10×; L² error 4–9× worse than brute-force optimum.
- **Moderate r ($r=2$):** brute force *also* can't achieve low error — best is 91% L² error.
- **High r ($r\geq 5$):** error is catastrophic (1.0–1.2 relative L²) independent of K. Brute force prefers small K because the linearized evolution diverges less.

Scaling law check:
- Theory predicts $K \propto \sqrt r$, $D$-independent.
- Tuner emits $K \propto r^{0.9}$, $D$-independent. ✓ (D-independence holds)
- Brute force emits $K \propto r^{-0.78}$. **Wrong sign of $r$-scaling.** $K_\text{opt}$ DECREASES with $r$.

ω-sweep at $(D=0.05, r=2, K=8)$ shows monotonic decrease of $L^2$ error as $\omega$ increases from 0 to 1.2, with $\omega=1.2$ giving the best result. This is evidence that the linearized scheme is **systematically under-correcting**.

## Diagnosis — the bound is correctly predicted but the bound is just large

The Fisher-KPP per-window linearization residual is
$$\rho_W^F = -r\,(u - u_W^*)^2.$$

For the non-tracking estimator $u_W^* = \|u\|_{L^\infty}$ applied to a non-uniform profile:
$$u_\text{rel} = \|u - u_W^*\|_{L^\infty} \approx \|u\|_{L^\infty} - \min u = O(u_\text{max}).$$

The per-window error (Duhamel) is
$$\|E_W\|_{L^2} \leq \int_0^{\Delta t}\|\rho_W^\text{true}\|_{L^2}\,d\tau \leq r\,u_\text{rel}^2\sqrt L\,\Delta t.$$

**Linear in $\Delta t$**, not $\Delta t^2$. Aggregating $K$ windows:
$$\|u(T) - \hat u(T)\|_{L^2}^\text{lin} \leq e^{rT}\,K\,r\,u_\text{rel}^2\sqrt L\,\Delta t = e^{rT}\,r\,u_\text{rel}^2\sqrt L\,T.$$

**This is constant in $K$** (does not improve as $K$ grows). The cascade does not converge in $K$ for Fisher-KPP with a non-uniform profile and a scalar operating state.

For the $r=0.5$ case: $e^{rT} = e^1 = 2.72$, $u_\text{rel}\approx 0.3$, $\sqrt L\approx 4.5$: bound is $2.72 \cdot 0.5 \cdot 0.09 \cdot 4.5 \cdot 2 \approx 1.1$ — 110% error. The actual error at $K_\text{opt}=32$ is 1.7%, so the bound is loose, but it is correctly *non-vanishing in $K$*.

For $r=10$: $e^{rT}=e^{20} \approx 5\times 10^8$. Bound is meaningless.

## Why Burgers doesn't have this pathology

Burgers' per-window residual has the structure
$$\rho_W^\text{Burg} = -(u - u_0^*)\cdot u_x,$$
a product of a "spatial deviation" and a "spatial derivative." When integrated against the contractive semigroup via Duhamel and bounded, the *per-window* error becomes $O(u_\text{rel}\cdot M_x\cdot\Delta t^2)$ — **second order in $\Delta t$** under a mild continuity assumption on the Picard correction's handling of the $(u - u_0^*)\cdot u_x$ bilinear structure.

Fisher-KPP's residual is $-r(u-u_0^*)^2$, a **pure power of the deviation with no spatial derivative**. There is no bilinear structure to exploit, no dt-factor to extract from the spatial variation. Per-window error stays first-order in $\Delta t$.

**This is a real structural distinction**, not a bug: the Burgers template does not transfer to reactive PDEs as simply as the generalization document suggested. The claim in [phase1c_theorem_generalization.md](phase1c_theorem_generalization.md) §1.5 — "Picard with $\omega = 1$ gives the same $O(\Delta t^2)$ reduction" — is **wrong** for Fisher-KPP unless an additional hypothesis is added.

## Corrected Fisher-KPP theorem

The theorem as written is still correct on its face:
$$\|u(T) - \hat u(T)\|_{L^2} \leq (1-\eta)\,e^{rT}\,\sqrt L\,r\,u_\text{rel}^2\,T + K\varepsilon_N\|u_0\|_{L^2}.$$

But the first term does not vanish in $K$ — the cascade cannot reduce it past $(1-\eta)\cdot$ a constant. The useful regime is therefore:

> **Useful regime for Fisher-KPP cascade (scalar operating state):** $u_\text{rel}\ll 1$ and $rT = O(1)$. In this regime, both the amplification $e^{rT}$ and the residual $r u_\text{rel}^2$ are small, the bound is tight, and the cascade converges.

For profiles with $u_\text{rel} = O(1)$ (our Gaussian bump tests), the bound is not useful and the cascade does not converge in $K$.

## Three paths forward

### Option A — Restrict theorem statement to weakly nonlinear regime

Add hypothesis **(NL-B):** $u_\text{rel} = \|u - u_0^*\|_{L^\infty}/\|u_0^*\|_{L^\infty} \ll 1$. This is the "small perturbation" regime where Fisher-KPP dynamics are essentially linear.

Cost: theorem applies only to boring near-uniform profiles. The interesting Fisher-KPP phenomenology (front propagation, invasion dynamics, traveling waves) is *excluded*. Not acceptable for a convincing paper.

### Option B — Spatially-varying operating state (tracking estimator)

Use $u_0^*(x) = \hat u(t_{W-1}, x)$ as a spatially-varying operating state per window. The linearization becomes
$$\partial_t v = D v_{xx} + r(1 - 2u_0^*(x))\,v - r u_0^*(x)^2 + r u_0^*(x),$$
a non-constant-coefficient linear PDE. Fourier diagonalization fails; need a non-spectral linear solver per window (sparse linear systems, or spectral with convolutions).

The per-window residual becomes
$$\rho_W = -r[u(t,x) - u_0^*(x)]^2 \approx -r[u_t(t_{W-1})\cdot(t - t_{W-1})]^2,$$
which is $O((\Delta t)^2)$ in time — $\Delta t^2$-small per window, giving cubic balance and good convergence.

**Cost:** the scheme no longer decomposes mode-by-mode. The spectral-scalpel structure (batched IFFT over modes) is broken. This makes Fisher-KPP a **qualitatively different scheme** from Burgers.

### Option C — Operator splitting: linear diffusion via NILT, reactive via pointwise ODE

Split Fisher-KPP as
$$\partial_t u = \underbrace{D u_{xx}}_{L[u]} + \underbrace{r u(1-u)}_{N[u]}.$$

Linear part $L[u]$: handle via NILT (standard spectral scalpel, modewise).
Reactive part $N[u]$: **this is a pointwise logistic ODE** $\partial_t u = r u(1-u)$ with closed-form solution
$$u(t) = \frac{u_0}{u_0 + (1-u_0) e^{-rt}},$$
applied pointwise at each grid point.

**Strang splitting** gives second-order accuracy: $u_{W+1} = \mathcal{N}(\Delta t/2)\,\mathcal{L}(\Delta t)\,\mathcal{N}(\Delta t/2)\,u_W$ where $\mathcal{L}$ is the NILT step and $\mathcal{N}$ is the pointwise logistic flow.

**Cost:** departs from the "single theorem, one tuner" template. The tuner for this scheme is a standard splitting-error analysis, not the pencil framework.

## Recommendation

**Option C is the honest path.** Fisher-KPP is fundamentally a different regime than Burgers because the nonlinearity is pointwise-local (no spatial-derivative coupling), and the natural scheme exploits this locality (pointwise ODE) rather than forcing it into the spectral-scalpel mold.

**For the eventual NCS paper:**
- Burgers is the poster child for the pencil cascade (wave-like, tracking estimator, clean Re-scaling).
- Kuramoto-Sivashinsky is a second poster child (fourth-order dispersion + same Burgers nonlinearity).
- Fisher-KPP and Allen-Cahn should be treated **separately** via splitting. Either include them as "how the framework combines with operator splitting for reactive PDEs" (a complementary section) or exclude them entirely from the theorem coverage.

The [phase1c_theorem_generalization.md](phase1c_theorem_generalization.md) §1 and §2 claims for FKPP and AC are **incorrect as written**. They need to be withdrawn and replaced with:

> **Claim for Fisher-KPP and Allen-Cahn (revised):** The scalar-linearization cascade does not converge in K for non-uniform profiles. Use operator splitting (Option C above) instead.

## Next steps

1. Mark [phase1c_theorem_generalization.md](phase1c_theorem_generalization.md) §1 and §2 as "superseded — see [phase1c_fkpp_failure_diagnosis.md](phase1c_fkpp_failure_diagnosis.md)."
2. Check whether Allen-Cahn has the same structural issue (it does — residual is $-(u-u_0^*)^2(u+2u_0^*)$, the quadratic factor dominates and has the same pathology). Document briefly.
3. Validate Burgers + KS (the two wave-like PDEs that do fit the cascade). KS on a single-sign regime if possible.
4. Acknowledge in the eventual paper that the unified theorem covers **wave-like PDEs with spatial-derivative nonlinearity**. Reactive PDEs are a different regime.

## What this does not mean

The theoretical framework is still correct for Burgers and KS. The CFL bound, the angular-CFL formula, the square-root vs cubic balance regimes, the tracking-vs-non-tracking distinction — all of these hold.

What changes: the claim that **the same theorem generalizes to all four PDEs** is wrong. Fisher-KPP and Allen-Cahn need a different scheme (operator splitting) to be handled correctly. The original pencil cascade is a **wave-like-PDE** framework.

This is actually **consistent with the symbolic derivations**: Fisher-KPP and Allen-Cahn have *real* Fourier symbols (no wave structure); they are qualitatively different from Burgers and KS.

The empirical validation **saved us from a wrong theorem claim**. The generalization-by-analogy was too aggressive; the careful proof (Lemma C, Lemma E bounds for FKPP) would have caught this if I had been stricter, but it was easier to see by running the numbers and observing that the cascade doesn't converge in K.
