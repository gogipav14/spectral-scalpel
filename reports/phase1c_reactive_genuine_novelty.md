# What is genuinely novel about handling reactive PDEs in the spectral-scalpel framework?

The user's pushback: Strang splitting + NILT-linear + closed-form-reaction is not a new idea. Any operator-splitting text from the 1990s covers it. For the spectral scalpel to claim reactive PDEs as in-scope, we need more than a packaging of existing machinery.

Three candidate angles of genuine novelty, in order of interest:

## Angle 1 — Gauge transforms that convert pointwise reactions into spatial-derivative structure

This is the most promising direction. The core idea:

> **A pointwise reactive nonlinearity $N(u) = f(u)$ can often be "linearized" by a gauge transform $w = \Phi(u)$, converting the PDE into $\partial_t w = L'[w] + \tilde N[w]$ where $\tilde N$ has spatial-derivative structure (and thus fits the pencil cascade of Theorem 1).**

If this works, we don't need splitting at all — reactive PDEs become a special case of the pencil cascade under the right change of variables.

### 1.1 Fisher-KPP via logit transform

Define $w(t,x) = \log(u/(1-u))$ (logit transform). Then $u = \sigma(w) = 1/(1+e^{-w})$. Applying the chain rule:

$$\sigma'(w)\,\partial_t w = D\bigl[\sigma''(w)\,w_x^2 + \sigma'(w)\,w_{xx}\bigr] + r\,\sigma(w)(1-\sigma(w)).$$

Using $\sigma'(w) = \sigma(1-\sigma)$ and $\sigma(w)(1-\sigma(w)) = \sigma'(w)$:

$$\sigma'(w)\,\partial_t w = D\sigma'(w)\,w_{xx} + D\sigma''(w)\,w_x^2 + r\,\sigma'(w).$$

Divide through by $\sigma'(w) > 0$ (valid for $u \in (0,1)$):

$$\partial_t w \;=\; D\,w_{xx} \;+\; D\,(1 - 2\sigma(w))\,w_x^2 \;+\; r.$$

**This is the key structural change.** Compare:

- In $u$-space, Fisher-KPP has LINEAR diffusion + POINTWISE REACTION. The reaction is the obstruction.
- In $w$-space, Fisher-KPP has LINEAR diffusion + QUASILINEAR $w_x^2$ term + CONSTANT SOURCE. **The reaction has become a constant source** (trivial to handle); the remaining nonlinearity is quasilinear and has the spatial-derivative structure the pencil cascade handles.

### 1.2 Linearization in logit space

Linearize $w$-space Fisher-KPP around a scalar $w_0^*$. Since $w_0^*$ is constant, $w_{0,x}^* = 0$. The functional derivative of $N[w] = D(1-2\sigma(w))w_x^2$ at $w = w_0^* + h$:

Taylor expand: $\sigma(w_0^* + h) = \sigma(w_0^*) + \sigma'(w_0^*)h + O(h^2)$, and $(w_0^* + h)_x = h_x$. So
$$N[w_0^* + h] = D\bigl[(1-2\sigma(w_0^*)) - 2\sigma'(w_0^*)h + O(h^2)\bigr]\,h_x^2 = D(1-2\sigma(w_0^*))\,h_x^2 + O(h\,h_x^2).$$

**The linearization of $N$ at a scalar $w_0^*$ is zero** (the leading term $D(1-2\sigma(w_0^*))h_x^2$ is quadratic in $h$, not linear). So the per-window linear substep is simply:
$$\partial_t v = D v_{xx} + r, \qquad v(t_{W-1}) = \hat w(t_{W-1}).$$

Fourier-diagonalizable. The DC mode evolves as $\hat v_0(\tau) = \hat v_0(0) + r\tau$ (linear growth), and the non-DC modes decay as $\hat v_n(\tau) = \hat v_n(0) e^{-D k_n^2 \tau}$. **Trivially handled by NILT** (one-shot Laplace inversion with a simple symbol).

The Picard residual is:
$$\rho_W = D(1-2\sigma(w))w_x^2.$$

This is **quadratic in $w_x$**. In the cascade, Picard integrates this against the semigroup over a window. Per-window Duhamel bound:
$$\|E_W^\text{lin}\|_{L^2} \leq \int_0^{\Delta t} \|\rho_W\|_{L^2}\,d\tau \leq D\cdot 1\cdot \|w_x\|_{L^\infty}\cdot \|w_x\|_{L^2}\cdot \Delta t.$$

Since $\|w_x\|$ is bounded by the profile's regularity (for $u$ bounded away from 0 and 1), this is $O(\Delta t)$ per window — **same as Burgers' pencil residual**.

**Remarkably, Fisher-KPP in logit space fits into the Theorem 1 pencil template.** The reactive nonlinearity is absorbed into a constant source; the remaining nonlinearity is spatial-derivative-structured.

### 1.3 Limitations of the logit transform

- **Invariant region:** logit is singular at $u = 0$ and $u = 1$. If the solution approaches these extremes, $w$ blows up. For smooth fronts that stay strictly in $(0, 1)$, this is fine; for solutions with $u \equiv 0$ support (bumps in a zero background), logit is ill-defined on the zero-set.
- **Workaround:** use a regularized transform $w = \log((u+\varepsilon)/(1-u+\varepsilon))$ with small $\varepsilon$. The regularized PDE has slightly different $N$ but the structure is preserved.
- **Physical interpretation:** $w$ is a "reaction extent" or "odds log-ratio." In chemical kinetics this is the standard way to linearize autocatalytic reactions.

### 1.4 Allen-Cahn analogue

Allen-Cahn $\partial_t u = \varepsilon^2 u_{xx} + u - u^3$. The autonomous reaction $\dot u = u - u^3$ has invariants $u = 0, \pm 1$ (fixed points). The "natural" gauge is related to the elliptic integral of $1/(u - u^3)$.

Define $w = \text{arctanh}(u)$ so $u = \tanh(w)$, $u^3 = \tanh^3(w)$, $u - u^3 = \tanh(w)\,\text{sech}^2(w)$.

Applying the chain rule:
$$\text{sech}^2(w)\,\partial_t w = \varepsilon^2\bigl[\text{sech}^2(w)\,w_{xx} - 2\,\text{sech}^2(w)\tanh(w)\,w_x^2\bigr] + \tanh(w)\,\text{sech}^2(w).$$

Divide by $\text{sech}^2(w)$:
$$\partial_t w = \varepsilon^2 w_{xx} - 2\varepsilon^2\tanh(w)\,w_x^2 + \tanh(w).$$

Similar structure: diffusion + quasilinear $w_x^2$ term + **a nonlinear source $\tanh(w)$**. The source isn't constant but it's purely pointwise in $w$. 

For Allen-Cahn, the source $\tanh(w)$ is bounded and smooth. It can be treated as a "slow" linear term: $\tanh(w) \approx \tanh(w_0^*) + \text{sech}^2(w_0^*)(w-w_0^*) + O((w-w_0^*)^2)$, which gives a linearization with both a constant-in-space (scalar $\tanh(w_0^*)$) and a linear (in $w$) contribution. Fourier-diagonalizable, NILT-compatible.

**So Allen-Cahn also admits a gauge transform that fits into the pencil cascade framework**, with a slightly richer linear structure (diffusion + linear drift in $w$, rather than just diffusion + constant source for Fisher-KPP).

### 1.5 The novel contribution

This is not standard in operator-splitting literature. The use of a gauge transform to convert a pointwise reactive nonlinearity into a spatial-derivative quasilinear structure, combined with the pencil cascade in the gauge coordinates, is — as far as I can tell — not a standard trick. It's motivated by the Burgers/Hopf-Cole precedent (which linearizes Burgers to the heat equation via $u = -2\nu\partial_x\log\phi$), but applied differently.

**Claim for the paper**: "The spectral scalpel extends to reactive PDEs via a gauge transform that converts pointwise nonlinearities into spatial-derivative quasilinear structure, bringing them into the scope of Theorem 1 without modification."

This is a **genuinely spectral-scalpel result** — it's about the PDE's spectral structure after gauge, not about splitting.

## Angle 2 — Certified Strang splitting with NILT feasibility bounds

If gauge transforms don't work for some PDE, the fallback is Strang splitting. The spectral-scalpel contribution here is narrower but still real:

### 2.1 Standard Strang has no a priori accuracy bound

Textbook Strang splitting gives local error $O(\Delta t^3)$ and global $O(\Delta t^2)$. The constant is a BCH commutator norm that is **never computed** in practice. Users just refine $\Delta t$ until they look converged.

### 2.2 Spectral scalpel's contribution: feasibility-certified linear substep

The NILT-based linear substep inherits the feasibility bounds from the linear paper:
$$a\cdot t_\text{max} + \log(C/\varepsilon_\text{tail}) \leq L - \delta_s.$$

This gives an **a priori error bound** on the linear substep's accuracy as a function of Bromwich parameters $(a, T, N)$. Standard Strang with implicit-Euler linear substep doesn't have this — users pick time steps and hope.

### 2.3 Combined with commutator bound

The Strang commutator $[[L, N], L] + [[L, N], N]$ can be bounded analytically for Fisher-KPP and Allen-Cahn (I computed $[L, N] = -2Dr u_x^2$ for Fisher-KPP in the earlier document). Combined with the NILT's certified feasibility, this gives a **fully a priori** accuracy bound for the Strang + NILT scheme:
$$\|u(T) - \hat u(T)\|_{L^2} \leq C_\text{comm}(u_0, L, N)\cdot T^3/K^2 + K\cdot\varepsilon_\text{NILT}.$$

Every constant on the right is computable **before running the scheme**. That is the spectral-scalpel contribution — **certification**.

### 2.4 Claim strength

This is weaker than Angle 1. It's "standard scheme + certification," which is a useful engineering contribution but not a new method. For the NCS paper, this would be a supplementary section, not a main claim.

## Angle 3 — Laplace-domain nonlinearity inversion

An angle I explored briefly but does not pan out: the Laplace transform of the pointwise nonlinear reactive solution $\Phi_N(t)[u_0]$ has no clean closed form. Specifically, $\mathcal{L}[u(1-u)](s)$ involves the Bromwich convolution $\hat u * \hat u$, which is not algebraic.

Similarly, the Laplace transform of the full Fisher-KPP solution does not factor through a simple transfer function in the spectral-scalpel sense. Abandoning this direction.

## Recommendation

**Pursue Angle 1 (gauge transforms).** It's a genuine mathematical contribution: the insight that Fisher-KPP in logit coordinates fits the pencil-cascade structure is (to my knowledge) not standard, and it brings reactive PDEs into the same theorem as wave-like PDEs. The proof template (Lemma A–G) transfers with the new linear operator $L = D\partial_{xx}$ and new nonlinearity $N = D(1-2\sigma)w_x^2$ — both fit the theorem's assumptions once we work in $w$-space.

**Fallback to Angle 2 only for PDEs where no linearizing gauge is known.** For the four-PDE slate of the NCS paper (Burgers, KS, Fisher-KPP, Allen-Cahn), all four admit gauge transforms:
- Burgers: no gauge needed (already has spatial-derivative structure).
- KS: no gauge needed.
- Fisher-KPP: logit $w = \log(u/(1-u))$.
- Allen-Cahn: arctanh $w = \text{arctanh}(u)$.

The paper's theorem becomes: **"For any PDE in the class $\partial_t u = L[u] + N[u]$ with $L$ linear and constant-coefficient, and $N$ either (a) containing a spatial derivative (direct) or (b) pointwise autonomous (via gauge transform $w = \Phi(u)$ that produces a constant/linear source plus quasilinear term), the pencil cascade of Theorem 1 applies."**

That's a genuinely unifying claim and it respects the user's insistence on something beyond standard splitting.

## What to verify before committing

Before writing the paper this way, verify numerically:

1. **Logit-space Fisher-KPP cascade converges in K** (fixing the Phase 1c step 2 pathology). Predicted cubic balance.
2. **Arctanh-space Allen-Cahn similarly.**
3. **The gauge-space tuner gives $K_r \in [0.8, 1.2]$** across the same 15-case profile sweep as before.
4. **Edge cases:** what happens when u approaches 0 or 1? Need to test the regularized gauge and confirm it doesn't introduce pathologies.

These are short runs, mostly adaptations of the existing harness. If they confirm, we have a unified four-PDE theorem under genuine spectral-scalpel novelty.

## Summary in a sentence

**Strang + NILT is packaging. Gauge transforms + pencil cascade in gauge coordinates is a genuine spectral-scalpel contribution** — it converts the "pointwise reactive nonlinearity has no spatial-derivative structure" obstruction into "the Burgers-analogue spatial-derivative nonlinearity after $w = \Phi(u)$."
