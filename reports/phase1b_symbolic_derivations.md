# Phase 1b — Symbolic Derivations of $\mathcal{D}_\theta$ per PDE

## 0. Reframing for 1D PDEs

The linear paper's framework propagates through a 3D slab; the "pencil" is $\gamma_z(s, k_\perp)$, a complex propagation rate in the $z$-direction. For 1D PDEs (Burgers, Fisher-KPP, Allen-Cahn, KS) there is no $z$ — propagation is purely in time. The natural analogue is the **per-mode complex pole**:
$$\sigma(k; u_0^*) \;\in\; \mathbb{C}, \quad\text{such that}\quad \widehat u(k, t) = \widehat u_0(k)\, e^{\sigma(k; u_0^*)\,t}\quad\text{in window }W.$$

Here $u_0^*$ is the operating state at which the nonlinear operator is linearized in window $W$. The per-window NILT inverts the transfer function $H(s, k; u_0^*) = 1/(s - \sigma(k; u_0^*))$.

The "angular sensitivity" is then the rate at which $\arg \sigma$ (the wave-number-dependent phase velocity contribution) responds to a perturbation of $u_0^*$:
$$\boxed{\;\mathcal{D}_\theta(k; u_0^*) \;\equiv\; \frac{\partial\,\arg\sigma}{\partial u_0^*}\;=\; \Imag\!\left[\frac{1}{\sigma}\,\frac{\partial \sigma}{\partial u_0^*}\right]\;.}$$

The **per-window phase drift** of mode $k$ between window $W$ and $W+1$, after time $\Delta t$, is the imaginary part of the change in $\sigma\Delta t$:
$$\Delta\Phi_W(k) \;=\; \Delta t\,\cdot\,\Imag\!\left[\frac{\partial\sigma}{\partial u_0^*}\,\delta u_0^*_W\right] \;\approx\; \Delta t \,\cdot\, \Imag\!\left[\frac{\partial\sigma}{\partial u_0^*}\right]\, \|u_t\|\,\Delta t.$$

For the linearized propagation to remain phase-coherent across the cascade, $\Delta\Phi_W(k)$ must stay below $\pi$ at the highest physically-recoverable wavenumber. This is the **CFL-like Heisenberg bound** $\text{B2}$ from the previous exploration.

In the rest of this document, $u_x \equiv \partial_x u$, $u_x^{\max}\equiv\sup_x|u_x|$, and "operator drift" means $\partial\sigma/\partial u_0^*$.

---

## 1. Burgers: $\partial_t u + u\,u_x = \nu\,u_{xx}$

**Linearized around $u_0^*$:** $\partial_t u + u_0^*\,u_x = \nu\,u_{xx}$ (constant transport at speed $u_0^*$).

**Per-mode pole.**
$$\boxed{\;\sigma_B(k; u_0^*) \;=\; -\,i\,k\,u_0^* \;-\; \nu\,k^2\;.}$$
Imaginary part = phase velocity contribution; real part = diffusion damping.

**Operator drift.**
$$\frac{\partial \sigma_B}{\partial u_0^*} \;=\; -\,i\,k.$$
Real, with magnitude $|k|$, sign that of $-i$. The drift is **purely phase** — perturbing $u_0^*$ does **not** change the diffusion damping, only the wave speed.

**Angular sensitivity.**
$$\mathcal{D}_\theta(k; u_0^*) \;=\; \Imag\!\left[\frac{-ik}{-iku_0^* - \nu k^2}\right]
\;=\; \Imag\!\left[\frac{-ik(iku_0^* - \nu k^2)}{|σ_B|^2}\right]
\;=\; \frac{-\nu k^3}{k^2(u_0^*)^2 + \nu^2 k^4}
\;=\; \frac{-\nu k}{(u_0^*)^2 + \nu^2 k^2}.$$

This is **negative** (so the phase angle decreases with $u_0^*$ — flow speeds up, decreases the wave-frame phase). Take its absolute value for bounding purposes.

**Maximum over $k$.** $|\mathcal{D}_\theta|$ is maximized where $d|\mathcal{D}_\theta|/dk = 0$, giving the optimum at $k_\star = u_0^*/\nu$:
$$\boxed{\;|\mathcal{D}_\theta(k_\star; u_0^*)| \;=\; \frac{1}{2\,u_0^*}\;,\qquad k_\star \;=\; u_0^*/\nu\;.}$$

**Two important features:**

(a) The peak sensitivity is $\nu$-independent — it is $1/(2u_0^*)$ regardless of Reynolds number.

(b) The wavenumber at which it peaks is **exactly the inverse Reynolds length** $u_0^*/\nu$. So at high Re the binding mode lives at much higher wavenumbers, even though the per-mode sensitivity at that mode is the same.

**Per-window phase drift at $k_\star$.**
$$\Delta\Phi_W(k_\star) \;=\; \Delta t \cdot \Imag\!\left[-i k_\star \,\delta u_0^*_W\right]
\;=\; -\,k_\star\,\delta u_0^*_W\,\Delta t
\;=\; -\,\frac{u_0^*}{\nu}\,\|u_t\|\,\Delta t^2.$$

**CFL bound (B2).** Require $|\Delta\Phi_W(k_\star)| \leq \pi$:
$$\frac{u_0^*\,\|u_t\|}{\nu}\,\Delta t^2 \;\leq\; \pi
\;\;\Longleftrightarrow\;\;
K \;\geq\; T\sqrt{\frac{u_0^*\,\|u_t\|}{\pi\,\nu}}.$$

For Burgers in the convection-dominated regime ($\|u_t\|\sim u_0^* u_x^{\max}$ with $u_x^{\max}, u_0^*$ both $O(1)$ in $\nu$):
$$\boxed{\;K^* \;\propto\; \frac{T}{\sqrt\nu}\,\sqrt{(u_0^*)^2\,u_x^{\max}/\pi}
\;\Longrightarrow\; K^* \;\propto\; \mathrm{Re}^{1/2}\;.}$$

**This recovers the empirical Phase 0 scaling** $K_{\text{opt}}\propto\mathrm{Re}^{0.516}$. The Re-dependence comes from the wavenumber $k_\star = u_0^*/\nu$, not from the per-mode sensitivity. The cubic balance missed it because it never asked "at *what* $k$ does the linearization hurt most?"

**Conservation form for Burgers.** Energy $E[u] = \tfrac12\int u^2\,dx$ satisfies
$$\dot E = -\nu\int u_x^2\,dx \;-\; \int u^2 u_x\,dx = -\nu\,\|u_x\|_{L^2}^2 + 0\quad(\text{periodic BC}).$$
So $u\,u_x$ is energy-conservative — the nonlinearity moves energy across modes (a cascade) but does not create/destroy it. The linearization $u_0^*\,u_x$ in place of $u\,u_x$ does **not** preserve $E$ (it's not a self-adjoint perturbation of energy):
$$\dot E_{\text{lin}} - \dot E_{\text{true}} \;=\; \int u\,(u-u_0^*)\,u_x\,dx
\;=\; \tfrac12\int (u-u_0^*)\,(u^2)_x\,dx
\;=\; -\tfrac12\int (u-u_0^*)_x\,u^2\,dx,$$
which has magnitude $\sim u_x^{\max}\,\|u-u_0^*\|_\infty\,\|u\|_{L^2}^2/2 \sim u_x^{\max}\,\|u_t\|\,\Delta t \cdot \|u\|^2$ per window.

Conservation budget bound (B3): require summed spurious energy $\leq \varepsilon_E\,\|u\|_{L^2}^2$:
$$K\cdot u_x^{\max}\,\|u_t\|\,\frac{T}{K}\,\|u\|^2 \;\leq\; \varepsilon_E\,\|u\|^2
\;\Longrightarrow\; T\,u_x^{\max}\,\|u_t\| \;\leq\; \varepsilon_E.$$

Independent of $K$! So **B3 is not a binding constraint on $K$ for Burgers** — once $\Delta t$ is small enough to satisfy the CFL bound (B2), the conservation residual is automatically controlled. The dominant constraint is B2, the operator-drift CFL.

---

## 2. Fisher-KPP: $\partial_t u = D\,u_{xx} + r\,u\,(1-u)$

**Linearized around $u_0^*$:** $\partial_t u = D u_{xx} + r u_0^*(1-u_0^*) + r(1-2u_0^*)(u - u_0^*)$, i.e.
$$\partial_t u = D u_{xx} + r(1-2u_0^*)\,u + \text{const}.$$

**Per-mode pole.**
$$\boxed{\;\sigma_F(k; u_0^*) \;=\; -\,D\,k^2 \;+\; r\,(1 - 2u_0^*)\;.}$$
**Purely real** — Fisher-KPP has no wave propagation, only growth/decay. The pole sits on the real axis: damped at high $k$ (diffusion) and unstably amplified at low $k$ when $u_0^* < 1/2$ (logistic growth around the unstable equilibrium).

**Operator drift.**
$$\frac{\partial \sigma_F}{\partial u_0^*} \;=\; -\,2r,$$
constant in $k$.

**Angular sensitivity.**
$$\mathcal{D}_\theta = \Imag\!\left[\frac{-2r}{\sigma_F}\right] \;=\; 0\quad\text{since }\sigma_F\in\mathbb{R}.$$

**This is the "Picard OFF" signal that the angular-CFL diagnostic should produce.** The pencil drift between windows produces zero phase shift — there is no wave speed to correct. The Picard residual $-r(u-u_0^*)^2$ is anti-aligned with the linearization error $(u-u_0^*)$ for symmetric reasons (both have the same sign as $u - u_0^*$, but the correction shifts $u$ in the wrong direction for the unstable equilibrium).

**B2 (operator drift CFL) for Fisher-KPP.** The drift of the *real* part is what matters, since Im=0:
$$\delta\sigma_W = -2r\,\delta u_0^*_W = -2r\,\|u_t\|\Delta t.$$
For the linearized propagator $e^{\sigma_W \Delta t}$ to remain accurate as $u_0^*$ drifts:
$$|\delta\sigma_W|\,\Delta t = 2r\,\|u_t\|\,\Delta t^2 \leq \kappa,$$
giving
$$\boxed{\;K \;\geq\; T\sqrt{\frac{2r\,\|u_t\|}{\kappa}}\;.}$$

This is **independent of $D$** — Fisher-KPP's window count is set by the reaction time scale, not by Reynolds-like diffusivity ratios. Empirically the diagnostic should test this on a Fisher front.

**Conservation form: there is none.** Mass is *not* conserved (logistic growth creates/destroys $u$), and the natural Lyapunov functional $\int F(u)\,dx$ with $F'(u) = -r u(1-u)$ is unbounded below — Fisher-KPP is *not* a gradient flow. So B3 has no direct analog. Constraint reduces to B2.

**Picard sign (alignment).** With $\sigma_F$ real, the relevant question is whether the Picard correction reduces or amplifies the linearization error. Quadratic Picard residual: $\rho = -r(u-u_0^*)^2$. Linearization error direction: $u - u_0^*$. Inner product:
$$\langle \rho, u - u_0^*\rangle = -r\int (u-u_0^*)^3\,dx.$$
Sign is $-r\cdot\mathrm{sign}(\int(u-u_0^*)^3)$ which depends on the *skewness* of $u - u_0^*$. For a typical Fisher front advancing into $u\sim 0$, the leading edge has $u > u_0^*$ (skewed positive), so the integral is positive and $\langle\rho, u-u_0^*\rangle = -r\cdot(\text{pos}) < 0$ — anti-aligned.

**Conclusion for Fisher-KPP**: $\mathcal{D}_\theta = 0$ (no phase to correct) and alignment $<0$ (Picard hurts). The tuner should set $\omega_W = 0$. K from B2 only, scaling as $\sqrt{r\|u_t\|}$.

---

## 3. Allen-Cahn: $\partial_t u = \varepsilon^2 u_{xx} + u - u^3$

**Linearized around $u_0^*$:** linearized growth rate $\partial_u(u - u^3)|_{u_0^*} = 1 - 3(u_0^*)^2$.

**Per-mode pole.**
$$\boxed{\;\sigma_A(k; u_0^*) \;=\; -\,\varepsilon^2\,k^2 \;+\; 1 \;-\; 3(u_0^*)^2\;.}$$
Real, like Fisher-KPP. The pole sign depends on $u_0^*$:
- Near unstable equilibrium $u_0^*=0$: $\sigma_A(0) = +1$ — unstable growth.
- Near stable equilibria $u_0^*=\pm 1$: $\sigma_A(0) = -2$ — exponential decay back.
- At transition layer ($u_0^* \approx 0$ but with finite gradient): $\sigma_A$ swings from $+1$ to $-2$ as the cascade progresses.

**Operator drift.**
$$\frac{\partial \sigma_A}{\partial u_0^*} \;=\; -\,6\,u_0^*.$$
Magnitude scales with $|u_0^*|$ — small at the unstable equilibrium, large at the stable wells. **This is unique to Allen-Cahn**: the operator drift varies with the operating state, peaking at the wells.

**Angular sensitivity.**
$$\mathcal{D}_\theta = 0 \quad\text{(}\sigma_A\in\mathbb{R}\text{)}.$$
Like Fisher-KPP — no phase to correct.

**B2 for Allen-Cahn:**
$$|\delta\sigma_W| = 6|u_0^*|\,\|u_t\|\Delta t,\qquad K \geq T\sqrt{\frac{6|u_0^*|\,\|u_t\|}{\kappa}}.$$

The Re-analog here is the *interface-width* parameter $1/\varepsilon$. K is set by the operating state at the transition layer, not by $\varepsilon$ directly — but $\|u_t\|$ at the transition layer scales as $\varepsilon^{-1}$ (sharp interface implies large $u_x$ implies $\varepsilon^2 u_{xx} \sim 1/\varepsilon$).

**Conservation: gradient flow.** Allen-Cahn is the $L^2$ gradient flow of $F[u] = \int (\varepsilon^2/2) u_x^2 + (1-u^2)^2/4\,dx$, so $\dot F \leq 0$ exactly:
$$\dot F = -\int (u_t)^2\,dx.$$
The **linearization injects spurious dissipation** because the linearized operator is the gradient of a *quadratic* approximation to $F$. The error in $\dot F$ per window:
$$|\dot F_{\text{lin}} - \dot F_{\text{true}}| = \left|\int (u-u_0^*)^2\,(3u_0^* - u)\,dx\right| \lesssim 6 u_0^*_{\max}\,\|u-u_0^*\|^2_{L^2}.$$
Same square-root scaling as B2.

**Picard sign:** depends on $u_0^*$. At the transition layer where $u_0^*\approx 0$, the Picard residual $-3u_0^*(u-u_0^*)^2$ vanishes (so Picard is neutral). At the wells ($u_0^*\approx\pm 1$), the residual is large but the linearization is already accurate (the wells are stable). Effectively Picard is least useful where it might help most. **Allen-Cahn argues for omega_max small (e.g., 0.3) and sign-checked.**

---

## 4. Kuramoto-Sivashinsky: $\partial_t u + u u_x + u_{xx} + u_{xxxx} = 0$

**Linearized around $u_0^*$:** $\partial_t u + u_0^* u_x + u_{xx} + u_{xxxx} = 0$ (constant transport plus 2nd/4th-order linear operators).

**Per-mode pole.**
$$\boxed{\;\sigma_{KS}(k; u_0^*) \;=\; -\,i\,k\,u_0^* \;+\; k^2 \;-\; k^4\;.}$$
- Imaginary part: $-k u_0^*$ — wave speed (from advection).
- Real part: $k^2 - k^4$ — **unstable in the band $0 < k < 1$**, damped for $k > 1$. This is what produces KS chaos.

**Operator drift.**
$$\frac{\partial \sigma_{KS}}{\partial u_0^*} \;=\; -\,i\,k\quad\text{(same as Burgers).}$$
Pure imaginary — only the wave speed is modulated by the operating state.

**Angular sensitivity.**
$$\mathcal{D}_\theta = \Imag\!\left[\frac{-ik}{-iku_0^* + k^2 - k^4}\right]
= \Imag\!\left[\frac{-ik((iku_0^* + k^2 - k^4)^*)}{|σ_{KS}|^2}\right]
= \frac{-k(k^2 - k^4)}{(ku_0^*)^2 + (k^2 - k^4)^2}.$$

Maximized at $k$ where the denominator is minimized — and $k^2 - k^4 = 0$ at $k = 0, 1$. The denominator $(ku_0^*)^2 + 0 = (ku_0^*)^2$ at these zeros, and the numerator $-k\cdot 0 = 0$. So $\mathcal{D}_\theta \to 0$ at $k = 0, 1$ — a degenerate maximum at the linearly neutral mode. The peak of $|\mathcal{D}_\theta|$ is somewhere between, and grows as $|u_0^*| \to 0$ (numerator stays finite, denominator shrinks).

For $|u_0^*| = O(1)$ and the active KS regime ($k \sim 1/\sqrt 2$ for max growth), $|\mathcal{D}_\theta(k = 1/\sqrt 2)| \sim 1/(2u_0^*)$ similar to Burgers — but the binding mode is $k = 1/\sqrt 2$, not Re-dependent.

**B2 for KS:**
$$|\delta\sigma_W(k=1/\sqrt 2)| \cdot \Delta t = |k\delta u_0^*|\Delta t = \frac{|u_t|\Delta t^2}{\sqrt 2} \leq \kappa,
\;\;K \geq T\sqrt{\frac{|u_t|}{\sqrt 2 \kappa}}.$$
Notably **no Re-analog** — KS is dimensionless. $K$ scales only with $\|u_t\|$ at fixed $T$.

**Conservation:** Energy decays via $u_{xxxx}$ damping but is replenished by $-u_{xx}$ instability. The energy balance equation:
$$\dot E = \int (u_x^2 - u_{xx}^2)\,dx,$$
which is *neither* monotone nor sign-definite. KS is dissipative on the attractor but transiently energy-injecting. Conservation budget B3 doesn't bound K from below (no Lyapunov target) — but the *long-time* energy mean is set by the attractor's invariant measure, which is a stronger statistical constraint than this exploration covers.

---

## 5. Comparison table — what the four PDEs predict

| PDE | $\sigma(k; u_0^*)$ | $\partial\sigma/\partial u_0^*$ | $\Imag\sigma$? | Binding mode $k_\star$ | $|\mathcal{D}_\theta(k_\star)|$ | Predicted $K^* \propto$ |
|---|---|---|---|---|---|---|
| Burgers | $-iku_0^* - \nu k^2$ | $-ik$ | yes | $u_0^*/\nu = $ Re/L | $1/(2u_0^*)$ | $T\sqrt{|u_t|/\nu} \sim \sqrt{\mathrm{Re}}$ |
| Fisher-KPP | $-Dk^2 + r(1-2u_0^*)$ | $-2r$ | **no** | n/a | **0** | $T\sqrt{r|u_t|}$, $\omega=0$ |
| Allen-Cahn | $-\varepsilon^2 k^2 + 1 - 3(u_0^*)^2$ | $-6u_0^*$ | **no** | n/a | **0** | $T\sqrt{u_0^*|u_t|}$, $\omega$ small |
| KS | $-iku_0^* + k^2 - k^4$ | $-ik$ | yes | $\sim 1/\sqrt 2$ (fixed) | $\sim 1/(2u_0^*)$ | $T\sqrt{|u_t|}$ (no Re-analog) |

**Patterns.**

- **Re-scaling appears via $k_\star$, not via $\mathcal{D}_\theta$.** For Burgers, the per-mode sensitivity at the binding mode is $1/(2u_0^*)$, completely $\nu$-independent. The full constraint scales as $\sqrt{\mathrm{Re}}$ because the binding wavenumber $k_\star = u_0^*/\nu$ scales with Re. **This was the missing ingredient in the cubic balance.** No matter what `u_t` norm we picked, the cubic balance never had a Re-dependent term because we never tracked $k_\star$.

- **Reactive nonlinearities ($\sigma\in\mathbb{R}$) automatically signal "Picard OFF"** through $\mathcal{D}_\theta = 0$. The angular-CFL framework distinguishes wave-like nonlinearities (Burgers, KS) from reactive ones (Fisher-KPP, Allen-Cahn) **purely from the structure of $\sigma$**, with no need for an alignment heuristic. This is cleaner than the current `sign(⟨ρ, u-u₀⟩)` test, which depends on profile-shape statistics.

- **The conservation budget B3 reduces to B2** for Burgers (energy preserved, B3 silent) and Allen-Cahn (gradient flow, same scaling), and is undefined for Fisher-KPP and KS (no conserved Lyapunov functional). So **B2 is the sharp constraint**; B3 was the wrong path.

- **The "pencil" really is the right object**: each PDE produces a one-parameter family $\sigma(\,\cdot\,; u_0^*)$ whose drift $\partial\sigma/\partial u_0^*$ contains all the information needed for both K* and ω*. Wave-like vs. reactive is just $\Imag(\sigma)\neq 0$ vs. $\Imag(\sigma)=0$.

## 6. The new K\* selection rule

For wave-like PDEs (Burgers, KS, and the linear-paper systems generalized):
$$\boxed{\;K^* \;=\; \left\lceil\, T \sup_k\sqrt{\frac{|\Imag(\partial\sigma/\partial u_0^*)|\,\|u_t\|}{\pi}}\,\right\rceil\;.}$$

For reactive PDEs (Fisher-KPP, Allen-Cahn):
$$\boxed{\;K^* \;=\; \left\lceil\, T\sqrt{\frac{|\partial\sigma/\partial u_0^*|\,\|u_t\|}{\kappa_R}}\,\right\rceil\;,\qquad \omega^*=0.\;}$$

The supremum over $k$ is **the new ingredient** that recovers Re-scaling. For Burgers, $\sup_k k = k_\star = u_0^*/\nu$ at the maximum sensitivity wavenumber, giving $K \propto \sqrt{u_0^* \|u_t\|/(\pi\nu)}$.

For ω*, the rule is:
$$\omega^* = \begin{cases}\cos^2(\Delta\Phi(k_\star)) & \text{if } \mathcal{D}_\theta \neq 0\text{ (wave-like)}\\ 0 & \text{if }\mathcal{D}_\theta = 0\text{ (reactive)}\end{cases}$$

## 7. Predictions to test in code

Numerical predictions ready for empirical validation in Phase 1c:

1. **Burgers:** $K^* \propto \sqrt{u_0^* u_x^{\max}/\nu}\cdot T$ → $K\propto\sqrt{\mathrm{Re}}$ for fixed profile. Match Phase 0's empirical $\mathrm{Re}^{0.516}$ exponent.
2. **Burgers:** independent of profile shape (Gaussian, tanh, sine) once $u_0^*$ and $u_x^{\max}$ are correctly computed for that profile — fixes the Phase 1a profile-blind failure.
3. **Fisher-KPP:** $\omega^* = 0$ identically (no need for `omega_max=0` hack); $K \propto \sqrt{r\|u_t\|}\,T$ independent of $D$.
4. **Allen-Cahn:** $K \propto \sqrt{|u_0^*|\,\|u_t\|}\,T$. At the transition layer where $u_0^*\sim 0.5$ and $\|u_t\|\sim\varepsilon^{-1}$, this gives $K \propto T/\sqrt{\varepsilon}$.
5. **KS:** $K\propto\sqrt{\|u_t\|}\,T$ independent of any Re-like parameter (KS is parameter-free in the $\nu=1$ scaling).

If any of (1)–(5) fails empirically, the framework needs another pass; if all pass, Phase 1c (Theorem 1) can proceed.

## 8. Open issues for proof

- **Sup over $k$ vs. effective $k_\star$.** Numerically the sup might be set by Nyquist (grid Nyquist or NILT feasibility cutoff) rather than the theoretical $k_\star$. The theorem must specify whether $k_\star$ is "the sensitivity peak" or "min(sensitivity peak, NILT-recoverable max)".

- **Multiple binding modes.** For PDEs with several local maxima of $|\mathcal{D}_\theta(k)|$ (KS has neighborhoods of $k=0$ and $k=1$), the sup-over-k may be a complicated function of $u_0^*$. Need a proof step showing it's not pathological.

- **The constant $\kappa$.** I've left it as $\pi$ in B2 (Nyquist coherence). The actual value should match the angular-CFL Picard recovery rate; possibly $\kappa = \pi/2$ (corresponding to ω going to zero at $\Delta\Phi=\pi/2$ via $\cos^2$).

- **Multidimensional case.** For 2D PDEs (Burgers 2D, Fisher-KPP 2D), $k$ becomes $(k_x,k_y)$ and the sup is over a 2D wavenumber set. The structure should generalize but the explicit $k_\star$ formula will have to handle anisotropy.
