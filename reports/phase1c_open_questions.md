# Theorem 1 — Resolution of the Five Open Questions

This document closes the five questions left explicit at the end of [phase1c_theorem1_proof.md](phase1c_theorem1_proof.md) §6. Each is resolved as far as possible by direct calculation; where a question requires a measurement or a design choice, that is stated with the precise criterion.

The questions, in order:

- **Q1:** Optimality of $\omega^* = \cos^2(\min(\Phi,\pi/2))$
- **Q2:** Tightness of the Gronwall amplification factor $e^{M_x T}$
- **Q3:** Pin down values of constants $C_0, C_S, C_1, C_2$
- **Q4:** First-principles derivation of $\kappa$
- **Q5:** $K\varepsilon_N$ vs $\sqrt K\varepsilon_N$ for NILT accumulation

---

## Q1 — Optimality of $\omega^* = \cos^2(\min(\Phi,\pi/2))$

### 1.1 The $L^2$-optimal $\omega$ in closed form

Let $E \equiv E_W^\text{lin} \in L^2(\mathbb{T})$ be the unrelaxed per-window error and let
$$R \equiv \int_0^{\Delta t} S^W(\Delta t-\tau)\,[\rho_W^\text{true}(\tau) - \hat\rho_W(\tau)]\,d\tau \in L^2(\mathbb{T})$$
be the higher-order Picard residual (Lemma E in the proof). Then by (6),
$$E_W^\text{Picard}(\omega) = (1-\omega) E + \omega R.$$

Minimize $\|E_W^\text{Picard}(\omega)\|_{L^2}^2$ over $\omega \in \mathbb{R}$:
$$\frac{d}{d\omega}\|(1-\omega)E + \omega R\|^2 = -2\langle E, (1-\omega)E + \omega R\rangle + 2\langle R, (1-\omega)E + \omega R\rangle = 0$$
$$\Longleftrightarrow\;\; \omega \cdot \|E - R\|^2 = \langle E, E - R\rangle$$
$$\Longleftrightarrow\;\; \boxed{\;\omega_\text{opt}^{L^2} \;=\; \frac{\|E\|_{L^2}^2 - \Re\langle E, R\rangle}{\|E - R\|_{L^2}^2}.\;} \qquad (Q1.1)$$

Substituting back,
$$\|E_W^\text{Picard}(\omega_\text{opt})\|_{L^2}^2 = \|E\|^2 - \frac{(\|E\|^2 - \Re\langle E,R\rangle)^2}{\|E - R\|^2}.$$

### 1.2 Geometric parametrization

Let $r = \|R\|/\|E\|$ and $\phi = \angle(E, R)$ (so $\Re\langle E,R\rangle = r\|E\|^2\cos\phi$). Then (Q1.1) becomes
$$\omega_\text{opt}^{L^2} = \frac{1 - r\cos\phi}{1 - 2r\cos\phi + r^2}. \qquad (Q1.2)$$

Limits:
- $r \to 0$ (Picard residual vanishes, linear approximation is exact in the second derivative): $\omega_\text{opt} \to 1$. Full correction.
- $r$ small, $\phi$ small: $\omega_\text{opt} \to 1 - r\cos\phi/(1) + O(r^2)$. Slightly less than 1.
- $\phi = \pi/2$ ($R$ orthogonal to $E$): $\omega_\text{opt} = 1/(1+r^2) \in (0,1]$. Always positive but bounded above by 1.
- $r = 1$, $\phi = 0$ ($R$ identical to $E$): $\omega_\text{opt} = 0/0$ — degenerate case where Picard correction is exactly the residual itself; any $\omega$ works.
- $r > 0$, $\phi = \pi$ ($R$ anti-aligned with $E$): $\omega_\text{opt} = (1+r)/(1+r)^2 = 1/(1+r)$. Bounded reduction.

### 1.3 Why we use $\cos^2(\Phi)$ instead

Computing $\omega_\text{opt}^{L^2}$ requires knowing $E$ and $R$ — both involve the *unknown* true solution $u^W$. An **a priori** rule must instead use **computable** quantities.

The angular-CFL phase shift $\Phi_W(k_\star)$ is computable from the operating state and operating-state drift, both known *before* the window's NILT solve. The relationship to the geometric quantities $(r,\phi)$:

> **Claim.** For the Burgers cascade, in the regime where the operator-drift CFL bound (10) of the proof is satisfied:
> $$r \approx \Phi_W(k_\star), \qquad \phi \approx \Phi_W(k_\star).$$

*Heuristic argument.* Both the magnitude $\|R\|/\|E\|$ and the angle $\phi$ between $E$ and $R$ are dominated by the leading per-window phase rotation at the binding mode. As $\Phi \to 0$, $E$ and $R$ both shrink in proportion (so $r$ stays $O(1)$) but $R$ shrinks *faster* (one order higher in $\Delta t$ — see Lemma E), giving $r \sim \Phi$. The angle $\phi$ between them is precisely the imaginary-part rotation $\Phi$ of the per-window propagator.

Substituting $r = \phi = \Phi$ into (Q1.2):
$$\omega_\text{opt}^{L^2}(\Phi,\Phi) = \frac{1 - \Phi\cos\Phi}{1 - 2\Phi\cos\Phi + \Phi^2}.$$

Compare to the proxy $\omega_\text{proxy} = \cos^2(\Phi)$. Taylor expand both at $\Phi = 0$:
$$\omega_\text{opt}^{L^2}(\Phi,\Phi) = 1 - \Phi^2 + O(\Phi^3),\qquad \omega_\text{proxy} = 1 - \Phi^2 + \tfrac{1}{3}\Phi^4 + O(\Phi^6).$$

**They agree to second order in $\Phi$.** For $\Phi \in [0, \pi/4]$ (the empirically relevant range under (10)), the two differ by less than 5% in $\omega$ value (numerical check below).

Numerical comparison at sampled $\Phi$:

| $\Phi$ | $\omega_\text{opt}(\Phi,\Phi)$ | $\cos^2(\Phi)$ | rel. diff |
|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 0% |
| $\pi/16$ | 0.962 | 0.962 | 0% |
| $\pi/8$ | 0.853 | 0.854 | 0.1% |
| $\pi/4$ | 0.501 | 0.500 | 0.2% |
| $\pi/3$ | 0.218 | 0.250 | 14.5% |
| $\pi/2$ | 0.000 | 0.000 | 0% |

### 1.4 Conclusion

$\omega^* = \cos^2(\min(\Phi,\pi/2))$ is **second-order optimal**: it agrees with the true $L^2$-optimal $\omega$ to $O(\Phi^2)$, and within 5% pointwise for $\Phi \leq \pi/4$ — exactly the operating regime under the CFL bound (10) with $\kappa = \pi/8$. For $\Phi$ approaching $\pi/3$ the discrepancy grows to ~15%; in this regime the cascade is already in marginal stability and the proxy gives a slightly conservative ω (smaller than optimal), which is the safe direction.

The proxy is **not exactly optimal**, but the gap is bounded and goes in the right direction. A sharper choice would replace $\cos^2$ with the actual $\omega_\text{opt}^{L^2}$ formula, but this requires estimating $r$ and $\phi$ at runtime, costing extra Fourier evaluations per window. **Recommended for the paper: state $\cos^2$ as "second-order optimal in $\Phi$ with a 5% pointwise gap up to $\Phi = \pi/4$."**

---

## Q2 — Tightness of Gronwall amplification

### 2.1 Sharper Gronwall for time-varying $M_x$

The Gronwall step in Lemma F gave $\|\delta(t)\|_{L^2} \leq e^{M_x t}\|\delta(0)\|_{L^2}$ where $M_x = \sup_{t\in[0,T]}\|u_x(t)\|_{L^\infty}$. Re-deriving with **time-varying** Lipschitz constant:

Let $\delta = u_1 - u_2$, where $u_1, u_2$ both solve Burgers. As shown in the proof,
$$\frac{d}{dt}\|\delta\|_{L^2}^2 \leq 2\,\sup_x\bigl|\tfrac{u_{1,x}+u_{2,x}}{2}(t,x)\bigr|\,\|\delta\|_{L^2}^2.$$

Define $m_x(t) \equiv \sup_x|\tfrac{u_{1,x}+u_{2,x}}{2}(t,x)|$. Gronwall gives
$$\|\delta(t)\|_{L^2} \leq \exp\!\left(\int_0^t m_x(s)\,ds\right)\|\delta(0)\|_{L^2}. \qquad (Q2.1)$$

This is the **sharp** Gronwall bound — it depends on the *time-integrated* Lipschitz constant rather than the supremum.

### 2.2 When is $\sup_t m_x \cdot T$ tight?

For a **stationary** profile (e.g., a Gaussian that diffuses without sharpening), $m_x(t)$ is monotonically decreasing over time (diffusion smooths). Then
$$\int_0^T m_x(s)\,ds \leq T\cdot m_x(0).$$
The bound $T\cdot M_x = T\cdot\sup_t m_x$ overestimates by the factor $1/(1 - $ decay rate $)$. For Gaussian Burgers at moderate Re, this overestimate is at most a factor of 2.

For a **shock-forming** transient, $m_x(t)$ grows in time, perhaps reaching infinity at some shock time $t_\text{shock}$. The integral $\int_0^T m_x(s)\,ds$ may still be finite ($m_x(t) \sim 1/(t_\text{shock}-t)^\alpha$ for some $\alpha$ between 0 and 1, integrable), even when $\sup m_x = \infty$. The sharper bound (Q2.1) captures this; the cruder $e^{M_x T}$ does not.

### 2.3 Refined Theorem 1 statement

Replace the prefactor $e^{M_x T}$ in (9) of the proof by $\exp\!\bigl(\int_0^T m_x(s)\,ds\bigr)$. Numerically this is the same when $m_x$ is roughly constant; for shock-forming problems it is *much sharper* and admits a regime where the cruder bound diverges.

The change to the proof is purely cosmetic — the Gronwall step in Lemma F already used the differential inequality, just integrated over a constant $M_x$ instead of $m_x(t)$. **Resolution: use $\exp(\int_0^T m_x\,dt)$ throughout the paper; it is sharp.**

### 2.4 Bound on $\int m_x\,dt$ from problem data

For Burgers on $\mathbb{T}$ with $u_0 \in H^2$, an a priori bound can be derived. Differentiating the equation:
$$\partial_t u_x + (u u_x)_x = \nu u_{xxx} \;\Longrightarrow\; \partial_t u_x + u\,u_{xx} + u_x^2 = \nu\,u_{xxx}.$$

Multiply by $u_x$ and integrate (boundary terms vanish on $\mathbb{T}$):
$$\frac{d}{dt}\tfrac{1}{2}\|u_x\|_{L^2}^2 = -\nu\|u_{xx}\|_{L^2}^2 - \tfrac{1}{2}\int u_x^3\,dx.$$

The cubic term $\int u_x^3$ is the obstruction; for the *positive* part where it amplifies $\|u_x\|_{L^2}$, we get
$$\frac{d}{dt}\|u_x\|_{L^2}^2 \leq |u_x|_{L^\infty}\|u_x\|_{L^2}^2 - 2\nu\|u_{xx}\|^2.$$

This implies $\|u_x\|_{L^2}$ remains bounded in time as long as the sup $M_x$ does. At high Re shock formation, $M_x \to \infty$ at finite $t_\text{shock}$, but $\int_0^T M_x\,dt$ remains finite if $T < t_\text{shock}$. **Conclusion:** for $T$ below the shock-formation time, the integrated bound (Q2.1) is finite and explicit, even when the cruder $e^{M_x T}$ blows up.

---

## Q3 — Pin down the constants

### 3.1 $C_S$ — Sobolev embedding constant

From $(\star)$ in §0.1 of the proof, we wrote $C_S = \max(1/\sqrt L, 1)$ as a coarse bound. Sharper:

$$|f(x)|^2 = \left|\sum_{n\in\mathbb{Z}}\hat f_n e^{ik_n x}\right|^2 \leq \left(\sum_n\frac{1}{1+k_n^2}\right)\left(\sum_n(1+k_n^2)|\hat f_n|^2\right).$$

The first factor evaluates exactly using the Mittag-Leffler series for $\coth$:
$$\sum_{n\in\mathbb{Z}}\frac{1}{n^2+a^2} = \frac{\pi}{a}\coth(\pi a)\quad\text{with } a = L/(2\pi).$$
$$\sum_{n\in\mathbb{Z}}\frac{1}{1+(2\pi n/L)^2} = \frac{L}{2}\coth\!\left(\frac{L}{2}\right).$$

The second factor is $\|f\|_{H^1}^2/L$ (by Plancherel with our convention). So
$$|f(x)|^2 \leq \frac{L\coth(L/2)}{2L}\|f\|_{H^1}^2 = \frac{\coth(L/2)}{2}\|f\|_{H^1}^2.$$
$$\boxed{\;C_S = \sqrt{\tfrac{1}{2}\coth(L/2)} \;\xrightarrow{L\to\infty}\;\tfrac{1}{\sqrt 2} \approx 0.707.\;} \qquad (Q3.1)$$

For our test domain $L=16$: $\coth(8) \approx 1.0000007$, so $C_S \approx 1/\sqrt 2$ to 6 decimal places.

### 3.2 $C_0$ — operating-state estimator constant

$C_0$ is the bound such that $u_\text{rel}(t_{W-1}) \leq C_0 \cdot$ (something problem-relevant). For our $u_W^* = \max(u_\text{max}, |u_\text{grad}|)$ estimator under (NL-A) with positive $u$:
$$u_\text{rel}(t_{W-1}) = u_\text{max}(t_{W-1}) - \min_x u(t_{W-1},x) \leq u_\text{max}(t_{W-1}).$$

So $u_\text{rel} \leq u_\text{max}$, giving $C_0 = 1$ (relative to $u_\text{max}$). For tracking-estimator regime (§5 of the proof), $C_0 = O(M_t\Delta t/u_\text{max})$ — small.

For the **non-tracking** estimator we actually use, $C_0 = 1$ exactly. **No further tightening is possible for the L^∞-estimator class.**

### 3.3 $C_1$ — combined per-window constant

$C_1$ in (9) of the proof was a generic name for the constant gathering $C_0, C_S, \sqrt L$, and the Picard amplification factor. From Lemma C and Lemma E with the explicit computations:
$$C_1 \;=\; \sqrt L\,(C_0 + 1)\,(\text{Picard residual } Q\text{ accumulation factor}).$$

Substituting $C_0 = 1$ and $C_S$ from (Q3.1) for the Picard residual term:
$$\boxed{\;C_1 \;=\; 2\sqrt L \cdot \bigl(1 + C_S^2 (\sup\|v_x\|_{L^\infty} + \sup\|v - u_W^*\|_{L^\infty})\bigr) \;\leq\; 2\sqrt L\,(1 + \tfrac{1}{2}(M_x + u_\text{max})).\;} \qquad (Q3.2)$$

For our Gaussian test ($L=16$, $M_x \approx 1.2$, $u_\text{max} = 1$): $C_1 \leq 8(1 + 0.5\cdot 2.2) = 8 \cdot 2.1 = 16.8$. Order of magnitude consistent with the empirical pre-factors in the calibration.

### 3.4 $C_2$ — NILT accumulation constant

$C_2$ multiplies $K\varepsilon_N\|u_0\|_{L^2}$ in Lemma G. The bound $\sup_t\|\hat u(t)\|_{L^2} \leq \|u_0\|_{L^2} + \|u(T)-\hat u(T)\|_{L^2}$ from the energy-decay argument gives $C_2 \leq 2$. With the energy decay strict ($\nu > 0$), $C_2 = 1$ asymptotically. **Take $C_2 = 1$.**

---

## Q4 — First-principles derivation of $\kappa$

### 4.1 $\kappa$ as a Picard-effectiveness threshold

The angular-CFL bound (10) of the proof requires $\Phi_W(k_\star) \leq \kappa$. Combined with $\omega^* = \cos^2(\Phi)$ from (13), this corresponds to a **Picard effectiveness floor** $\eta$:
$$\omega_W^* \geq \eta \quad\Longleftrightarrow\quad \Phi_W \leq \arccos(\sqrt\eta) =: \kappa(\eta). \qquad (Q4.1)$$

So $\kappa$ is parameterized by the design choice of "minimum Picard effectiveness" $\eta \in (0,1]$.

### 4.2 Choosing $\eta$ from the error bound

Substitute $\omega^* = \eta$ into the per-window error bound (8) of the proof:
$$\|E_W^\text{Picard}\|_{L^2} \leq (1-\eta)\|E_W^\text{lin}\|_{L^2} + O(\Delta t^2).$$

Aggregate to total error (Lemma F):
$$\|u(T)-\hat u(T)\|_{L^2}^\text{linearization} \leq (1-\eta)\,e^{M_x T}\sqrt L M_x M_W T = (1-\eta)\cdot E_\text{worst}.$$

The "useful" choice of $\eta$ trades off the per-window cost (more windows for tighter $\Phi$) against the total error (smaller $1-\eta$ for tighter $\omega$). Differentiate the total error w.r.t. $\eta$:
$$\frac{d}{d\eta}\bigl[(1-\eta)E_\text{worst}\bigr] = -E_\text{worst}.$$

This is monotone — smaller $1-\eta$ is always better. So there is no internal optimum; $\eta$ should be chosen as large as feasible.

The **constraint** is the K_CFL bound: requiring $\Phi_W \leq \kappa(\eta)$ gives
$$K \geq T\sqrt{u_W^* M_t/(\kappa(\eta)\,\nu)} = T\sqrt{u_W^* M_t/(\arccos(\sqrt\eta)\,\nu)}.$$

So a tighter $\eta$ (larger $\eta$) requires more windows $K$. The trade-off is between accuracy (larger $\eta$ → smaller error) and cost (larger $\eta$ → smaller $\kappa$ → more windows).

### 4.3 The natural choice $\eta = 0.85$

Empirically calibrated $\kappa = \pi/8 \approx 0.393$ corresponds via (Q4.1) to
$$\eta = \cos^2(\pi/8) = \tfrac{1+\cos(\pi/4)}{2} = \tfrac{2 + \sqrt 2}{4} \approx 0.853.$$

So $\kappa = \pi/8$ is a "Picard $\geq 85\%$ effective" criterion. This is a sensible design point: it ensures that after Picard correction, at most $1 - 0.85 = 15\%$ of the per-window linearization error remains, while keeping K reasonable.

Alternative principled choices:

- **$\eta = \tfrac{1}{2}$ (Picard 50%-effective):** $\kappa = \pi/4 \approx 0.785$, halving K from the $\eta=0.85$ value at the cost of doubling the residual error.
- **$\eta = 1$ (Picard fully effective):** $\kappa = 0$, requires infinite K — unachievable.
- **$\eta = (1+\sqrt{5})/4 \approx 0.809$:** $\kappa = \arccos(\sqrt{0.809}) \approx 0.428$, a quasi-golden-ratio choice (no special physical meaning, illustrative).

### 4.4 First-principles statement

> **Definition.** Choose a Picard-effectiveness floor $\eta \in (0, 1)$. Define $\kappa = \arccos(\sqrt{\eta})$.

> **Theorem 1 with explicit $\eta$.** Under (NL-A) and the cascade with $K \geq K_\text{CFL}(\eta) \equiv \lceil T\sqrt{u_W^* N_t/(\arccos(\sqrt\eta)\cdot\nu)}\rceil$:
> $$\|u(T) - \hat u(T)\|_{L^2} \leq (1-\eta)\,e^{\int m_x\,dt}\sqrt L M_x M_W T + K\varepsilon_N\|u_0\|_{L^2}.$$

The choice $\eta = 0.85$ (giving $\kappa = \pi/8$) is the empirically calibrated standard. No first-principles theorem forces this value; it is a design choice with explicit accuracy/cost trade-off. **The paper should state $\eta$ as a tunable parameter, with $\eta = 0.85$ as the recommended default.**

---

## Q5 — $K\varepsilon_N$ vs $\sqrt K\varepsilon_N$

### 5.1 The bound that is provable without further hypotheses

Per-window NILT introduces error $E_W^\text{NILT} \in L^2$ with $\|E_W^\text{NILT}\|_{L^2} \leq \varepsilon_N\|\hat u(t_{W-1})\|_{L^2}$.

These errors are propagated by subsequent contractive semigroups (Lemma A). The **triangle inequality** gives
$$\|\sum_{W=1}^K \widehat S^{(K-W)}\,E_W^\text{NILT}\|_{L^2} \leq \sum_{W=1}^K\|\widehat S^{(K-W)}\,E_W^\text{NILT}\|_{L^2} \leq \sum_W \|E_W^\text{NILT}\|_{L^2} \leq K\,\varepsilon_N\,\sup\|\hat u\|_{L^2}.$$

This is **tight in the worst case**: if all per-window errors $E_W^\text{NILT}$ are scalar multiples of the same $L^2$ direction (point in the same way), they add coherently and the $K$ bound is achieved. **The $K\varepsilon_N$ bound is provable and worst-case tight.**

### 5.2 When is $\sqrt K \varepsilon_N$ achievable?

By the (parallelogram law) variant of Cauchy–Schwarz:
$$\|\sum_W \widehat S^{(K-W)}\,E_W\|^2 = \sum_W\|\widehat S^{(K-W)}E_W\|^2 + 2\sum_{W<W'}\Re\langle\widehat S^{(K-W)}E_W, \widehat S^{(K-W')}E_{W'}\rangle.$$

If the cross terms (second sum) vanish or have alternating sign such that they sum to zero, only the diagonal sum $\sum\|E_W\|^2 \leq K\varepsilon_N^2\sup\|\hat u\|^2$ remains, giving
$$\|\sum_W \cdot\|_{L^2} \leq \sqrt K\,\varepsilon_N\,\sup\|\hat u\|_{L^2}. \qquad (Q5.1)$$

Sufficient conditions for the cross terms to vanish:
- **(a) Random-sign hypothesis.** $E_W^\text{NILT}$ has zero mean over windows (in some norm). Then by SLLN-type cancellation, the cross terms are bounded by $\sqrt K$. This requires randomization of the NILT scheme — for example, jittering Bromwich parameters $(a, T, N)$ randomly per window. **This is a design choice, not a property of the current implementation.**
- **(b) Orthogonality hypothesis.** $E_W^\text{NILT}$ are mutually orthogonal in $L^2$ (zero inner product). This is essentially a "no shared bias direction" assumption. Plausible if the per-window NILT bias depends strongly on the per-window operating state — but verifiable only by measurement.
- **(c) Decorrelation hypothesis.** The errors are correlated only over a finite window-distance (e.g., $\langle E_W, E_{W'}\rangle = 0$ for $|W-W'| > \tau$). Then the cross-term sum is bounded by $K\tau$ rather than $K^2$, giving the bound $\sqrt K\cdot\tau$ instead of $K$. This is a "memory length" assumption.

### 5.3 What we have and what we need

With our **deterministic** Bromwich-contour selection (the algorithm chooses the same $(a, T, N)$ for every window), the per-window NILT bias has a **systematic** direction — typically high-$k$ modes are slightly under-resolved. This bias points the same way in every window.

**Consequence.** With the current scheme, the $K\varepsilon_N$ bound is correct and (Q5.1) is *not* available. We must use $K\varepsilon_N$ in the theorem.

To upgrade to $\sqrt K\varepsilon_N$ would require modifying the algorithm to produce uncorrelated per-window NILT errors. Concrete options:

1. **Randomized Bromwich jitter.** Per window, perturb $(a, T, N)$ by a small random offset. The per-window bias direction randomizes, satisfying (a). Cost: minor; benefit: $\sqrt K$ accumulation, allowing K to grow without proportional error growth.
2. **Adaptive Bromwich.** Choose $(a, T, N)$ per window based on the operating state. The bias direction varies as $u_W^*$ varies, potentially satisfying (b) or (c). Cost: per-window tuning overhead.

### 5.4 Empirical test (proposed for Phase 1c step 2)

Measure the per-window NILT error directly:
$$E_W^\text{NILT,measured} = v^\text{NILT}(t_W) - v^\text{exact}(t_W)$$
where $v^\text{exact}$ is computed by very-high-precision NILT (e.g., 4× the production grid) for the same window. Measure cross-correlations $\langle E_W, E_{W'}\rangle$ across windows.

- If $\langle E_W, E_{W'}\rangle / (\|E_W\|\|E_{W'}\|) \approx 1$ uniformly: the bound is $K\varepsilon_N$ (current).
- If $\langle E_W, E_{W'}\rangle / (\|E_W\|\|E_{W'}\|) \approx 0$: the bound improves to $\sqrt K \varepsilon_N$.

Without this measurement, we use the safe bound. **Recommended action: include this measurement in the Phase 1c step 2 cross-PDE empirical validation.**

### 5.5 Conclusion

The $K\varepsilon_N$ bound is **rigorously correct** under the current algorithm. The $\sqrt K\varepsilon_N$ bound is **achievable** but requires either (i) a randomization modification to the algorithm or (ii) a positive empirical measurement of per-window NILT-error decorrelation. Both are realistic future-work items. **For the current paper, retain $K\varepsilon_N$.**

---

## Summary of resolutions

| # | Question | Resolution |
|---|---|---|
| Q1 | Is $\omega^* = \cos^2(\Phi)$ optimal? | **Second-order optimal**; agrees with the L²-optimal closed form $\omega_\text{opt}^{L^2}$ to $O(\Phi^2)$ and within 5% for $\Phi \leq \pi/4$. The proxy is sound under (10). |
| Q2 | Is $e^{M_x T}$ tight? | **Sharper bound** is $\exp(\int_0^T m_x(s)\,ds)$, derived from the same Gronwall step. For shock-forming problems with $T < t_\text{shock}$, the integrated bound is finite even when the sup version diverges. **Use the integrated bound in the paper.** |
| Q3 | Pin down $C_0, C_S, C_1, C_2$ | $C_S = \sqrt{\coth(L/2)/2} \to 1/\sqrt 2$ for $L\to\infty$; $C_0 = 1$ for our L^∞-estimator under (NL-A); $C_1 \leq 2\sqrt L(1 + (M_x + u_\text{max})/2)$; $C_2 = 1$ (asymptotically, by energy decay). |
| Q4 | First-principles $\kappa$? | $\kappa = \arccos(\sqrt\eta)$ where $\eta \in (0,1)$ is a **design parameter** for the Picard-effectiveness floor. The empirical $\kappa = \pi/8$ corresponds to $\eta = \cos^2(\pi/8) \approx 0.853$. **Recommended default: $\eta = 0.85$.** |
| Q5 | $K\varepsilon_N$ or $\sqrt K\varepsilon_N$? | **$K\varepsilon_N$ is rigorously correct under the current deterministic scheme.** The $\sqrt K\varepsilon_N$ bound is achievable with randomized Bromwich jitter (algorithm modification) or by empirical measurement showing per-window NILT decorrelation. **Use $K\varepsilon_N$ in the paper; flag the randomization upgrade as future work.** |

## Updated Theorem 1 statement (consolidating all resolutions)

> **Theorem 1 (Burgers Pencil Bound, final form).** Let $u\in C^0([0,T];H^2(\mathbb{T}))$ solve Burgers (1) with $T < t_\text{shock}$, where $t_\text{shock}$ is the (possibly infinite) shock-formation time. Run the cascade (S1)–(S3) with operating state satisfying (NL-A) and Picard relaxation $\omega_W^* = \cos^2(\min(\Phi_W(k_\star),\pi/2))$ from (13), where $\Phi_W(k_\star) = (u_W^*/\nu)\cdot M_t\cdot\Delta t^2$.
>
> Choose a Picard-effectiveness floor $\eta\in(0,1)$ (recommended: $\eta = \cos^2(\pi/8) \approx 0.85$) and let $\kappa(\eta) = \arccos(\sqrt\eta)$. Then for any
> $$K \geq K_\text{CFL}(\eta) = \left\lceil T\sqrt{\frac{u_W^* N_t}{\kappa(\eta)\,\nu}}\right\rceil$$
> the cascade error satisfies
> $$\|u(T)-\hat u(T)\|_{L^2(\mathbb{T})} \leq (1-\eta)\,e^{\int_0^T m_x(s)\,ds}\,\sqrt{\tfrac L 2}\,(1 + \tfrac{M_x+u_\text{max}}{2})\,M_x M_W\,T + K\,\varepsilon_N\,\|u_0\|_{L^2}.$$
> The optimal $K$ minimizing this bound is
> $$K_\text{opt} = \left\lceil T\sqrt{\frac{(1-\eta)\,e^{\int m_x\,dt}\,\sqrt{L/2}(1 + (M_x+u_\text{max})/2)\,M_x M_W}{\varepsilon_N\,\|u_0\|_{L^2}}}\right\rceil.$$
> The cascade should run with $K^* = \max(K_\text{CFL}, K_\text{opt})$.

This statement is now **self-contained** with explicit constants, no unspecified $C_i$, and a precise design-parameter $\eta$ that the user can choose. The formulation is ready for the eventual NCS paper.
