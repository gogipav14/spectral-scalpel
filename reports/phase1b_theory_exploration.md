# Phase 1b — Theoretical Exploration: Operator Pencil + Heisenberg-like Bounds

**Status:** Mathematical sketch, not yet a theorem. Goal: replace the empirical cubic balance with a principled framework that (a) admits CFL-like uncertainty bounds on inputs, (b) ties the angular shift Δθ to the *operator drift* between windows rather than to a static initial-condition diagnostic, and (c) anchors the linearization error in conservation laws (energy/mass balance) of the underlying PDE.

## 1. Setup — the parameterized linearized pencil

Take a semi-linear PDE on $\mathbb{R}^d_x \times \mathbb{R}_{t \geq 0}$:
$$\partial_t u = L[u] + N[u], \qquad L \text{ linear constant-coefficient}, \quad N \in C^2.$$

Split time into $K$ windows $W = 1, \ldots, K$ of length $\Delta t = T/K$. Within window $W$, freeze the operating state $u_W^* \equiv u(\cdot, t_{W-1})$ and propagate the *linearized* equation
$$\partial_t v = L[v] + N(u_W^*) + N'(u_W^*)\,(v - u_W^*).$$

Define the **per-window propagation pencil** by Fourier+Laplace transform of the linearized operator:
$$\boxed{\;\gamma^2_W(s, k_\perp) \;=\; \gamma^2_L(s, k_\perp) \;+\; \widehat{N'(u_W^*)}(s, k_\perp)\;,}$$
where $\gamma^2_L$ is the dispersion of $L$ alone (the linear-paper pencil) and $\widehat{N'(u_W^*)}$ is the Fourier–Laplace symbol of the multiplier $N'(u_W^*)$. This is a one-parameter family of pencils, parameterized by $u_W^*$.

**Cross-window shift.** Between windows $W$ and $W+1$,
$$\Delta\gamma^2_W \;=\; \gamma^2_{W+1} - \gamma^2_W \;=\; \widehat{N''(\xi_W)\,\delta u_W^*}, \qquad \delta u_W^* = u_{W+1}^* - u_W^*,$$
by the mean-value theorem applied to $N'$. The operator pencil **drifts** between windows at a rate set jointly by $\|N''\|$ (curvature of the nonlinearity) and $\|\delta u_W^*\| \approx \|u_t\| \cdot \Delta t$ (how far the operating state moves per window).

This $\Delta\gamma^2_W$ is the object the new theorem should track. It is the analogue, for the nonlinear case, of the $\gamma_z$ branch shift the linear paper's Eq.~(9) bounds via $L - \delta_s$.

## 2. Conservation-residual error: linearization injects spurious flux

For the true equation, conserved quantities $Q[u] = \int q(u)\,dx$ satisfy $dQ/dt = 0$ when $q$ is a Casimir of the dynamics (mass for Burgers, mass+momentum for Maxwell, etc.). The linearized substep does **not** preserve these in general:

$$\frac{dQ}{dt}\bigg|_{\text{linearized}} \;-\; \frac{dQ}{dt}\bigg|_{\text{true}}
\;=\; \int q'(v)\,\bigl[N(v) - N(u_W^*) - N'(u_W^*)(v-u_W^*)\bigr]\,dx.$$

The bracketed expression is exactly the **Picard residual** $\rho_W(v)$. Taylor expansion to second order:
$$\rho_W(v) \;=\; \tfrac{1}{2}\,N''(u_W^*)\,(v - u_W^*)^2 \;+\; O((v-u_W^*)^3).$$

Time-integrate over a window: $v - u_W^* \sim u_t \cdot \tau$ for $\tau \in [0, \Delta t]$, so
$$\Delta Q_W^{\text{spurious}} \;\lesssim\; \tfrac{1}{2}\,\|q'\|_\infty\,\|N''\|_\infty\,\langle u_t^2\rangle \cdot \tfrac{\Delta t^3}{3}.$$

Summing over $K$ windows:
$$\sum_{W=1}^K \Delta Q_W^{\text{spurious}} \;\lesssim\; \tfrac{1}{6}\,\|q'\|_\infty\,M\,\langle u_t^2\rangle\,\frac{T^3}{K^2}, \qquad M = \|N''\|_\infty.$$

This is exactly the cubic-balance numerator (with the factor $1/6$ rather than $2$). What's new: the **norm choice** is now anchored to a conservation law, not to the dispersion $\tau_\text{cross}$. For Burgers $q(u) = u^2/2$ (energy), $q'(u) = u$, so $\|q'\| = \|u\|_\infty$. The spurious energy injection scales as $\|u\|_\infty \cdot \|u_x\|_\infty \cdot \langle u_t^2\rangle$.

**This is the missing input.** The "right" `u_t` norm in the cubic balance is the rate of change *measured against the conservation law*, not against an arbitrary spatial $L^p$.

## 3. Heisenberg-like bounds on the tuning quartet $(K, \omega, a, T)$

The linear paper's NILT feasibility uses three independent constraints (Nyquist, branch margin, dynamic range). For the nonlinear cascade, four free parameters $(K, \omega, a, T)$ obey four corresponding bounds:

**B1 — Nyquist (resolution per window).** The per-window NILT must resolve frequencies up to $\omega_\text{max}^W = \pi/\Delta t \cdot N$. The spectrum of the *drift-corrected* pencil $\gamma^2_W + \Delta\gamma^2_W/2$ must lie within $[\,a, a + \omega_\text{max}^W\,]$. This bounds $\Delta t$ from above:
$$\Delta t \;\leq\; \frac{\pi N}{a + \sup_W \rho(\gamma^2_W)}.$$

**B2 — CFL on operator drift.** The cross-window pencil shift must be small enough that the linearized propagator computed at $u_W^*$ remains accurate over $[t_W, t_{W+1}]$:
$$\boxed{\;\|\Delta\gamma^2_W\|_\infty \cdot \Delta t \;\leq\; \kappa,\;}$$
for some Courant-like constant $\kappa = O(1)$. Substituting $\|\Delta\gamma^2_W\| \lesssim M \cdot \|u_t\|_\infty \cdot \Delta t$ gives
$$M \cdot \|u_t\|_\infty \cdot \Delta t^2 \;\leq\; \kappa \;\Longrightarrow\; K \;\geq\; T\,\sqrt{\frac{M\,\|u_t\|_\infty}{\kappa}}.$$

This is a **square-root** scaling, distinct from the cubic balance.

**B3 — Conservation budget.** Spurious flux per window (from §2) must not exceed the tolerated drift in $Q$:
$$\sum_W \Delta Q_W^{\text{spurious}} \;\leq\; \varepsilon_Q \cdot \|q\|_\infty.$$
Substituting:
$$\frac{M\,\langle u_t^2\rangle\,T^3}{6\,K^2} \;\leq\; \varepsilon_Q\,\|q\|_\infty
\;\Longrightarrow\; K \;\geq\; T\,\sqrt{\frac{M\,\langle u_t^2\rangle}{6\,\varepsilon_Q\,\|q\|_\infty}}.$$

Also a square-root scaling.

**B4 — Heisenberg uncertainty (information floor).** The product of time- and frequency-domain windows is bounded below by the Bromwich quadrature:
$$\Delta t \cdot \Delta\omega \;\geq\; \frac{\pi}{N}, \qquad \Delta\omega = \pi/T.$$
With $\Delta t = T/K$ this gives $K \leq N$ — an upper bound, not a lower one.

**Combined K\* selection.** Take the *maximum* of B2 (CFL-like operator drift) and B3 (conservation residual), capped by B4:
$$K^* \;=\; \max\!\left(\;T\sqrt{\tfrac{M\|u_t\|_\infty}{\kappa}},\;\;T\sqrt{\tfrac{M\,\langle u_t^2\rangle}{6\,\varepsilon_Q\,\|q\|_\infty}}\;\right) \;\leq\; N.$$

**This predicts $K \propto T \sqrt{M\,\|u_t\|}$, not $K \propto T \,(M\,\|u_t\|^2)^{1/3}$.** The earlier cubic balance balanced *only* the conservation residual against $\varepsilon_N$ (ignoring B2). When B2 (CFL on operator drift) is the binding constraint, the law is square-root and is dominated by the largest local rate — exactly the Re-scaling we observed empirically (the $\frac{1}{2}$-power of Re).

## 4. Angular shift as cross-window pencil rotation

The angular-CFL diagnostic in the current code computes $\Delta\theta_\text{eff} = \arctan(\min(\Gamma_{NL}, 10))$ from the *initial-condition* $\Gamma_{NL}$, then derives $\omega_\text{geo} = \cos^2(\Delta\theta)$. This is static — the same $\Delta\theta$ for every window.

The user's correct observation: $\Delta\theta$ should track how the wave propagation **changes** between windows, not be set once from the IC. Concretely:
$$\Delta\theta_W \;=\; \arg\!\bigl[\gamma_W(i\omega_\text{cross})\bigr] \;-\; \arg\!\bigl[\gamma_{W+1}(i\omega_\text{cross})\bigr].$$

For the parameterized pencil $\gamma^2_W = \gamma^2_L + \widehat{N'(u_W^*)}$, a perturbation expansion gives, to first order in $\delta u_W^*$:
$$\Delta\theta_W \;=\; \frac{\Imag\bigl[\partial\gamma^2/\partial u\,\bigr]}{2\,|\gamma|^2} \cdot \delta u_W^*\bigg|_{i\omega_\text{cross}} \;\equiv\; \mathcal{D}_\theta(u_W^*) \cdot \delta u_W^*.$$

The operator $\mathcal{D}_\theta$ is computable from the dispersion relation (it is the *angular sensitivity* of the pencil to perturbations of the operating state). For the two-term pencil $\gamma^2 = \alpha s + \beta s^2$ with $\alpha, \beta$ both depending on $u_0^*$ (e.g., for Burgers $\alpha = (u_0^*/(2\nu))^2$):
$$\mathcal{D}_\theta(u_0^*)\bigg|_{i\omega_\times} \;=\; \frac{1}{2|\gamma|^2}\,\Imag\!\left[\frac{i\omega_\times \alpha'(u_0^*)}{\gamma(i\omega_\times)}\right].$$

Then the **per-window angular shift** is
$$\Delta\theta_W \;=\; \mathcal{D}_\theta(u_W^*) \cdot \|u_t\|\,\Delta t,$$
and the right Picard relaxation factor is
$$\omega_W \;=\; \cos^2(\Delta\theta_W) \cdot \mathrm{sign}\bigl(\langle\rho_W, u_t\rangle\bigr),$$
where $\langle\rho_W, u_t\rangle$ is the alignment between the Picard residual and the time-derivative direction (the $\mathrm{sign}$ factor is the missing alignment input from Phase 1a).

For Burgers, $\langle\rho_W, u_t\rangle > 0$ (advective steepening aligned with NILT under-prediction) → $\omega_W > 0$.
For Fisher-KPP, $\langle\rho_W, u_t\rangle < 0$ (reactive growth opposes NILT) → $\omega_W = 0$ (Picard OFF).

## 5. The "pencil" as a unified object

Combining §1, §3, §4: the *nonlinear spectral scalpel pencil* is a one-parameter family
$$\Pi(u_0^*) \;=\; \bigl(\gamma^2_L + \widehat{N'(u_0^*)},\;\; q,\;\; \langle\,\cdot,\,u_t\rangle\bigr),$$
consisting of (a) the linearized dispersion, (b) the conservation form $q$, (c) the alignment functional. The **operator drift between windows** is $\delta\Pi_W = \Pi(u_{W+1}^*) - \Pi(u_W^*)$. The cubic-balance/CFL/conservation inequalities are statements about the spectral radius and angular structure of $\delta\Pi_W$.

**Theorem-shaped goal (Phase 1c, after this exploration):**

> **(Conjectural)** Let $u$ solve $\partial_t u = L[u] + N[u]$ with conservation form $q$. The spectral-scalpel cascade with $K$ windows, Picard relaxation $\omega_W = \cos^2(\Delta\theta_W) \cdot \mathrm{sign}(\langle\rho_W, u_t\rangle)$, and per-window NILT tolerance $\varepsilon_N$ produces $\hat u$ with
$$\|u - \hat u\|_{L^2_t L^2_x}^2 \;\leq\; \mathcal{C}_1\,\frac{M\,\|u_t\|_2^2\,T^3}{K^2}\bigl(1 - \omega_*/2\bigr) \;+\; \mathcal{C}_2\,K\,\varepsilon_N \;+\; \mathcal{C}_3\,\sup_W \|\delta\Pi_W\|^2,$$
where the third term enforces the CFL-like bound B2 on operator drift. The optimal $K^*$ minimizes the sum and satisfies the **Heisenberg pair**
$$K^* \cdot \Delta t \;=\; T, \qquad \Delta t \cdot \sup_W\|\delta\Pi_W\| \;\leq\; \kappa.$$

Notably this is **not a single cubic balance** — it is the maximum of two square-root constraints. The cubic balance recovers as a special case when the operator-drift CFL B2 is *not* binding (small $M$, slow $u_t$ — the regime Phase 0's Gaussian Burgers happened to live in).

## 6. What this changes for Phase 1b implementation

If this framing is right, the Phase 1b checklist is:

1. **Compute $\mathcal{D}_\theta(u_0^*)$ symbolically** for each PDE class (Burgers, Fisher-KPP, Allen-Cahn, KS) from their dispersion relations. This is one closed-form per PDE.

2. **Per-window angular shift**: replace the static `Δθ_eff = arctan(Γ_NL)` in [angular_cfl.py:156](../scalpel/nonlinear/angular_cfl.py#L156) with $\Delta\theta_W = \mathcal{D}_\theta(u_W^*) \cdot \|u_t\|\Delta t$. This will *grow* with $\|u_t\|$ and *shrink* with $K$ — recovering the missing Re-dependence.

3. **Conservation budget**: introduce `q_evaluator(u, x) -> Q[u]` for each PDE (mass/energy). The per-window spurious flux can be *measured* during the cascade as a diagnostic; the user can set $\varepsilon_Q$ as a tolerance.

4. **K\* as max of B2 and B3**: implement the new selector, which takes $\max$ of the operator-drift CFL bound (B2) and the conservation-residual bound (B3). Both are $K \propto \sqrt{M\|u_t\|}$ in different norms; the ratio of the two indicates which constraint binds.

5. **Picard sign gating**: also unconditional — multiply $\omega_\text{geo}$ by the alignment sign (the simple fix that's been outstanding since Phase 1a).

## 7. Open mathematical questions

- **Existence of the Heisenberg pair.** The bound $\Delta t \cdot \sup_W \|\delta\Pi_W\| \leq \kappa$ is conjectural. Proof would parallel the linear-paper proof of the feasibility bound (Eq.~3) but with $\delta\Pi_W$ in place of $\alpha_c$. Likely requires assumptions on $N''$ regularity (bounded second derivative on the orbit).

- **What is $\kappa$?** In the linear case, $\kappa = L - \delta_s$ (dynamic range minus margin). In the nonlinear case, $\kappa$ should depend on the geometry of the pencil family; a candidate is $\kappa = $ injectivity radius of the parameterization $u_0^* \mapsto \gamma^2(u_0^*)$.

- **When does B2 *not* bind?** For "almost-linear" problems (small $M$, small $\|u_t\|$) the cubic balance survives. For shock-formation problems (Burgers at high Re), B2 dominates and the law becomes square-root. The boundary is exactly $M \|u_t\| T^2 \sim$ conservation-budget terms — a Reynolds-number-like threshold.

- **Connection to the linear paper's $\Lambda_-$.** The angular sensitivity $\mathcal{D}_\theta$ at the crossover should reduce, in the linear limit, to $-1/(2\pi\Lambda_-) \cdot$ universal constants. Verifying this would tie Track B back to Paper 1's universality theorems.

## 8. Recommendation for next step

Before implementing anything, I suggest **one round of symbolic derivation per PDE**: write out $\mathcal{D}_\theta$ for Burgers, Fisher-KPP, Allen-Cahn, and KS by hand, and predict what the per-window angular shift looks like as a function of $u_0^*$. If the formulas are clean and PDE-specific in instructive ways (e.g., Burgers gives an angular shift proportional to $u_0^*/\nu$ — exactly Re), the framework is on the right track. If they don't simplify, we should consider an alternative: a fully *operator-theoretic* formulation that doesn't require a per-PDE closed form (e.g., treat $\mathcal{D}_\theta$ as a numerically-computed operator).

If the symbolic check passes, then Phase 1b becomes:
- (a) implement $\mathcal{D}_\theta$ for one PDE (Burgers),
- (b) re-run Phase 1a's validation harness with the per-window angular shift,
- (c) confirm Re-scaling agreement to within 10%,
- (d) only then write Theorem 1.

If symbolic check fails, we revisit the framing before writing more code.
