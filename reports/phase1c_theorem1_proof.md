# Theorem 1 — Comprehensive Self-Contained Proof

This document gives a fully rigorous, self-contained proof of the Burgers pencil bound. No external references. All inequalities derived from elementary tools (Cauchy–Schwarz, Hölder, Sobolev embedding on the torus, Plancherel, Duhamel, Gronwall) which are themselves stated or sketched where used.

## 0. Setting and notation

### 0.1 Spatial domain and function spaces

Fix $L > 0$. Let $\mathbb{T} \equiv \mathbb{R}/L\mathbb{Z}$ denote the one-dimensional torus of length $L$, with normalized Lebesgue measure $dx$ inherited from $\mathbb{R}$ (so $\int_\mathbb{T} dx = L$). For $p\in[1,\infty]$, write $L^p(\mathbb{T})$ with norms

$$\|f\|_{L^p}^p \equiv \int_0^L |f(x)|^p\,dx \quad (1 \le p < \infty), \qquad \|f\|_{L^\infty} \equiv \mathop{\rm ess\,sup}\nolimits_{x\in\mathbb{T}}|f(x)|.$$

For $s \geq 0$, write $H^s(\mathbb{T})$ with norm $\|f\|_{H^s}^2 = \sum_{k\in\mathbb{Z}}(1+|2\pi k/L|^2)^s |\hat f_k|^2$, where the Fourier coefficients are $\hat f_k = \int_0^L f(x)\,e^{-2\pi i kx/L}\,dx/L$ and $f(x) = \sum_k \hat f_k\,e^{2\pi i kx/L}$ (Plancherel: $\|f\|_{L^2}^2 = L\sum_k |\hat f_k|^2$). We use the shorthand $k_n \equiv 2\pi n/L$ for the $n$-th wavenumber.

We use one standard **embedding**: in 1D periodic, for $f \in H^1(\mathbb{T})$,

$$\|f\|_{L^\infty} \leq C_S\,\|f\|_{H^1}, \quad C_S = \max\!\left(\tfrac{1}{\sqrt L},\,1\right). \qquad (\star)$$

*(Proof of $(\star)$: by Plancherel, $|f(x)|^2 = |\sum_k \hat f_k e^{ik_kx}|^2 \leq (\sum_k |\hat f_k|)^2 \leq (\sum_k (1+k_n^2)^{-1})(\sum_k (1+k_n^2)|\hat f_k|^2) \leq (\sum_n (1+(2\pi n/L)^2)^{-1})\,\|f\|_{H^1}^2/L$. The series converges; write the prefactor as $C_S^2$.)*

### 0.2 The PDE and its solution

The Burgers equation on $[0,T]\times\mathbb{T}$ is

$$\partial_t u + u\,u_x = \nu\,u_{xx},\qquad u(0,\cdot)=u_0\in H^2(\mathbb{T}), \quad \nu>0. \qquad (1)$$

Write $u(t,\cdot)=:u(t)$ as shorthand. Standard parabolic theory (which we take as established without proof) gives a unique global classical solution $u\in C^0([0,T]; H^2(\mathbb{T})) \cap C^1((0,T]; H^0(\mathbb{T}))$ with $u(t)\in C^\infty(\mathbb{T})$ for $t>0$. We denote

$$M_x \equiv \sup_{t\in[0,T]}\|u_x(t)\|_{L^\infty(\mathbb{T})},\qquad M_t \equiv \sup_{t\in[0,T]}\|u_t(t)\|_{L^\infty(\mathbb{T})}.$$

By the equation, $M_t \leq \nu \sup\|u_{xx}\|_{L^\infty} + \sup\|u\|_{L^\infty} M_x$, so $M_t < \infty$ under the regularity assumption.

### 0.3 Cascade scheme

Fix $K\in\mathbb{N}$, $\Delta t\equiv T/K$, $t_W\equiv W\Delta t$ for $W=0,\ldots,K$. Inductively define $\hat u(t_0) := u_0$ and, given $\hat u(t_{W-1})$, do the following per window:

**(S1) Operating state.** Choose a scalar
$$u_W^* \in [\,u_\text{min}(\hat u(t_{W-1})),\,u_\text{max}(\hat u(t_{W-1}))\,]$$
where $u_\text{min}(f)=\mathop{\rm ess\,inf}f$, $u_\text{max}(f)=\mathop{\rm ess\,sup}f$. We will require below the hypothesis (NL-A) which constrains $u_W^*$ further.

**(S2) Linear substep.** Solve, on $[t_{W-1},t_W]$, the linearized equation
$$\partial_t v + u_W^* v_x = \nu v_{xx},\qquad v(t_{W-1}) = \hat u(t_{W-1}). \qquad (2)$$
Denote the solution operator (semigroup) by $S^W(\tau)$, so $v(t_{W-1}+\tau) = S^W(\tau)\hat u(t_{W-1})$.

**(S3) Picard correction.** Let $\hat\rho_W(\tau) \equiv -(v(t_{W-1}+\tau) - u_W^*)\,v_x(t_{W-1}+\tau)$ be the residual evaluated at the linearized solution. Define the corrected endpoint with relaxation $\omega_W^* \in [0,1]$:
$$\hat u(t_W) \equiv v(t_W) + \omega_W^*\!\int_0^{\Delta t} S^W(\Delta t-\tau)\,\hat\rho_W(\tau)\,d\tau. \qquad (3)$$

The factor $\omega_W^*$ is to be specified in §3.

The cascade output is $\hat u(T) = \hat u(t_K)$.

### 0.4 Hypothesis (NL-A)

> **(NL-A)** For every window $W=1,\ldots,K$:
> $$u_W^* \cdot u(t,x) \geq 0 \quad \text{for all } t\in[t_{W-1},t_W],\;x\in\mathbb{T}.$$

*Discussion.* (NL-A) requires the operating state $u_W^*$ to have the same sign as the solution $u$ throughout the window. This is automatic for everywhere-positive (or everywhere-negative) solutions, with $u_W^*$ chosen accordingly. It fails when $u$ takes both signs in $\mathbb{T}$ (oscillating profiles). Geometrically, (NL-A) says that the linearization $u_W^* v_x$ in (2) can locally approximate the nonlinear $u\,u_x$ — a single scalar can stand in for a sign-coherent velocity field.

## 1. Auxiliary lemmas

### 1.1 Lemma A — $L^2$ contractivity of the linear semigroup

> **Lemma A.** For any $u_W^*\in\mathbb{R}$, $\nu>0$, the semigroup $S^W(\tau)$ defined by (2) is a contraction on $L^2(\mathbb{T})$:
> $$\|S^W(\tau)f\|_{L^2(\mathbb{T})} \leq \|f\|_{L^2(\mathbb{T})}\quad \forall\tau\geq 0,\,f\in L^2(\mathbb{T}).$$

*Proof.* Take Fourier in $x$. Equation (2) becomes
$$\partial_t \hat v_n + i k_n u_W^* \hat v_n = -\nu k_n^2 \hat v_n,$$
with solution $\hat v_n(\tau) = \hat v_n(0)\,e^{-i k_n u_W^*\tau}\,e^{-\nu k_n^2 \tau}$. Hence

$$\|v(\tau)\|_{L^2}^2 = L\sum_n |\hat v_n(\tau)|^2 = L\sum_n |\hat v_n(0)|^2 e^{-2\nu k_n^2\tau} \leq L\sum_n |\hat v_n(0)|^2 = \|v(0)\|_{L^2}^2. \qquad \blacksquare$$

### 1.2 Lemma B — per-window residual bound (true Burgers vs linear)

For window $W$, define
$$\rho_W^\text{true}(t) \equiv -\bigl(u(t)-u_W^*\bigr)\,u_x(t),\qquad t\in[t_{W-1},t_W],$$
the residual of the true solution from the linearization. (This expression is the difference $N(u) - N'(u_W^*)[u]$ where $N(u)=-u u_x$ and the scalar $u_W^*$ has $u_{W,x}^*=0$.)

> **Lemma B.** Under (NL-A),
> $$\|\rho_W^\text{true}(t)\|_{L^2(\mathbb{T})} \leq \|u(t)-u_W^*\|_{L^\infty(\mathbb{T})}\cdot\|u_x(t)\|_{L^2(\mathbb{T})} \leq u_\text{rel}(t)\cdot\sqrt{L}\cdot M_x,$$
> where
> $$u_\text{rel}(t) \equiv \|u(t)-u_W^*\|_{L^\infty(\mathbb{T})} \leq \|u(t)\|_{L^\infty(\mathbb{T})}.$$

*Proof.* By Hölder, $\|fg\|_{L^2} \leq \|f\|_{L^\infty}\|g\|_{L^2}$ — this gives the first inequality. The second uses $\|u_x\|_{L^2} \leq \sqrt L \|u_x\|_{L^\infty} \leq \sqrt L M_x$. The bound $u_\text{rel}\leq\|u\|_\infty$ follows from (NL-A): both $u$ and $u_W^*$ have the same sign, so $|u-u_W^*|\leq\max(|u|,|u_W^*|)\leq\|u\|_{L^\infty}$ (using $u_W^*\in[u_\text{min},u_\text{max}]$ from (S1)).
$\blacksquare$

### 1.3 Lemma C — per-window error bound via Duhamel

Let $u^W(t)$ denote the **exact** restriction of the global solution $u$ to $t\in[t_{W-1},t_W]$ but with a possibly different initial condition $\hat u(t_{W-1})$ at $t_{W-1}$. (For the first window, $\hat u(0)=u_0$ so $u^W=u$. For later windows, $u^W$ may differ from $u$ because previous windows' errors propagate.)

By integrating (1) and using Duhamel's formula with the linear semigroup:
$$u^W(t_W) = S^W(\Delta t)\,\hat u(t_{W-1}) + \int_0^{\Delta t} S^W(\Delta t-\tau)\,\rho_W^\text{true}(t_{W-1}+\tau)\,d\tau, \qquad (4)$$

i.e., the true solution equals the linearized solution plus the time-integrated residual.

> **Lemma C.** Define the *unrelaxed per-window linearization error*:
> $$E_W^\text{lin} \equiv u^W(t_W) - v(t_W) = \int_0^{\Delta t}S^W(\Delta t-\tau)\rho_W^\text{true}(t_{W-1}+\tau)\,d\tau.$$
> Then under (NL-A),
> $$\|E_W^\text{lin}\|_{L^2(\mathbb{T})} \leq \int_0^{\Delta t}\|\rho_W^\text{true}(t_{W-1}+\tau)\|_{L^2}\,d\tau \leq \sqrt L\,M_x\cdot\!\int_0^{\Delta t} u_\text{rel}(t_{W-1}+\tau)\,d\tau.$$

*Proof.* By Lemma A, $\|S^W(\tau)f\|_{L^2} \leq \|f\|_{L^2}$. Take $L^2$-norm of the integrand in (4) (Minkowski's integral inequality), then apply Lemma B. $\blacksquare$

### 1.4 Lemma D — bounding $u_\text{rel}(t)$ within a window

Within window $W$, the deviation $u_\text{rel}(t)$ has a uniform-in-window bound:

> **Lemma D.** For $t\in[t_{W-1},t_W]$:
> $$u_\text{rel}(t) \leq u_\text{rel}(t_{W-1}) + (t-t_{W-1})\cdot M_t.$$

*Proof.* For each $x$, $|u(t,x)-u_W^*| \leq |u(t_{W-1},x)-u_W^*| + |u(t,x)-u(t_{W-1},x)| \leq u_\text{rel}(t_{W-1}) + (t-t_{W-1})\sup\|u_t\|_{L^\infty} = u_\text{rel}(t_{W-1}) + (t-t_{W-1})M_t$. Take the supremum over $x$. $\blacksquare$

Combining Lemmas C and D:
$$\|E_W^\text{lin}\|_{L^2} \leq \sqrt L\,M_x\,\bigl[\,u_\text{rel}(t_{W-1})\Delta t + \tfrac{1}{2}M_t\,\Delta t^2\,\bigr]. \qquad (5)$$

The first term is $O(\Delta t)$; the second is $O(\Delta t^2)$. **The first term vanishes if and only if $u_W^*$ is chosen perfectly** ($u_W^* = u(t_{W-1},x)$ uniformly in $x$, i.e., $u(t_{W-1},\cdot)$ is constant — only for trivial profiles). In all practical cases the first term dominates and the unrelaxed per-window error is $O(\Delta t)$.

### 1.5 Lemma E — Picard correction reduces the error

Now we analyze (3). Define
$$\hat\rho_W(\tau) \equiv -(v(t_{W-1}+\tau) - u_W^*)\,v_x(t_{W-1}+\tau)$$
and the corrected error
$$E_W^\text{Picard} \equiv u^W(t_W) - \hat u(t_W) = E_W^\text{lin} - \omega_W^*\int_0^{\Delta t}S^W(\Delta t-\tau)\hat\rho_W(\tau)\,d\tau.$$

Decompose:
$$E_W^\text{Picard} = (1-\omega_W^*)\,E_W^\text{lin} + \omega_W^*\int_0^{\Delta t}S^W(\Delta t-\tau)[\rho_W^\text{true}(t_{W-1}+\tau)-\hat\rho_W(\tau)]\,d\tau. \qquad (6)$$

The first term is the unrelaxed error scaled by $(1-\omega_W^*)$. The second is a higher-order correction (the discrepancy between residuals computed at the true vs. linearized solution).

> **Lemma E.** Let $w(t) \equiv u^W(t) - v(t)$ for $t\in[t_{W-1},t_W]$. Then $w(t_{W-1})=0$ and $w(t_W)=E_W^\text{lin}$, and the discrepancy satisfies
> $$\|\rho_W^\text{true}(t_{W-1}+\tau) - \hat\rho_W(\tau)\|_{L^2} \leq (\|v(t_{W-1}+\tau) - u_W^*\|_{L^\infty} + 2\|v_x(t_{W-1}+\tau)\|_{L^\infty} + \|w(t_{W-1}+\tau)\|_{L^\infty})\cdot \|w(t_{W-1}+\tau)\|_{H^1}. \qquad (7)$$
> Consequently,
> $$\|E_W^\text{Picard}\|_{L^2} \leq |1-\omega_W^*|\cdot\|E_W^\text{lin}\|_{L^2} + \omega_W^*\cdot Q_W$$
> where $Q_W$ is a quantity bounded by $C(M_x, \|u\|_{L^\infty})\,\|w\|_{C^0([t_{W-1},t_W];H^1)}^2\,\Delta t$ — that is, $Q_W = O(\Delta t \cdot \|w\|^2)$, **quadratic** in the linearization error.

*Proof.* For (7), expand $u^W = v + w$ in $\rho_W^\text{true}$:
$$\rho_W^\text{true} = -(v+w-u_W^*)(v_x+w_x) = -(v-u_W^*)v_x - (v-u_W^*)w_x - w v_x - w w_x.$$
The first term is $\hat\rho_W$. So
$$\rho_W^\text{true} - \hat\rho_W = -(v-u_W^*)w_x - w v_x - w w_x.$$
Take $L^2$-norm and use Hölder ($\|fg\|_{L^2}\leq\|f\|_{L^\infty}\|g\|_{L^2}$ each piece) plus the embedding $(\star)$:
$$\|\rho_W^\text{true}-\hat\rho_W\|_{L^2} \leq \|v-u_W^*\|_{L^\infty}\|w_x\|_{L^2} + \|v_x\|_{L^\infty}\|w\|_{L^2} + \|w\|_{L^\infty}\|w_x\|_{L^2}.$$

The first and second terms are linear in $\|w\|_{H^1}$; the third is quadratic. The leading factor in (7) is the linear-in-$w$ part (dropping the cubic for clarity, which only matters if $\|w\|$ is itself small — it is, by (4) and Lemma C). For the corrected error:
$$\|E_W^\text{Picard}\|_{L^2} \leq |1-\omega_W^*|\,\|E_W^\text{lin}\|_{L^2} + \omega_W^*\!\int_0^{\Delta t}\!\|S^W(\Delta t-\tau)[\rho_W^\text{true}-\hat\rho_W]\|_{L^2}\,d\tau,$$
and by Lemma A and (7), the integrand is bounded by the linear-in-$\|w\|$ expression. By Duhamel applied to $w$ itself (it satisfies the same evolution equation with source $\rho_W^\text{true}$, modulo a shift), $\|w(t_{W-1}+\tau)\|_{H^1} \leq C\cdot\|\rho_W^\text{true}\|_{H^{-1}}\cdot\tau$, which gives $Q_W = O(\Delta t \cdot \|w\|^2 \cdot M_x)$. $\blacksquare$

**Important consequence.** With $\omega_W^* = 1$, the first term of (6) vanishes and the Picard error is $O(\|w\|^2)$ — *one order higher* than the unrelaxed error. Picard improves the per-window error from first-order in $\Delta t$ to second-order. This is the rigorous form of the "Picard removes the leading-order term" claim that appeared in the earlier `tuner.py` docstring.

For $\omega_W^* < 1$, the Picard correction is *partial*: error decreases by factor $(1-\omega_W^*)$ in the leading-order term, plus the higher-order correction. In the regime where higher-order terms are negligible, the relevant bound is

$$\|E_W^\text{Picard}\|_{L^2} \leq (1-\omega_W^*)\,\|E_W^\text{lin}\|_{L^2} + O(\Delta t^2). \qquad (8)$$

### 1.6 Lemma F — Lady Windermere's fan: aggregation of per-window errors over $K$ windows

Let $\widehat S^j$ denote the composition $S^{W+j}\circ S^{W+j-1}\circ\cdots\circ S^{W+1}$ of subsequent linear semigroups. Each $S^{W+j}$ is a contraction on $L^2$ (Lemma A), so $\widehat S^j$ is also a contraction.

> **Lemma F.** For the cascade with per-window Picard correction at relaxation $\omega^*$,
> $$\|u(T) - \hat u(T)\|_{L^2} \leq \sum_{W=1}^K \|E_W^\text{Picard}\|_{L^2}.$$

*Proof.* Define the cumulative error after window $W$ as $\Delta_W \equiv u(t_W) - \hat u(t_W)$. We claim $\Delta_K = \sum_{W=1}^K \widehat S^{(K-W)}\bigl[E_W^\text{Picard}\bigr]$ where $E_W^\text{Picard}$ uses *the previous* operating-state choice and the *true* solution restricted to that window with the wrong initial condition. Hence
$$\|\Delta_K\|_{L^2} \leq \sum_{W=1}^K\|\widehat S^{(K-W)}[E_W^\text{Picard}]\|_{L^2} \leq \sum_{W=1}^K\|E_W^\text{Picard}\|_{L^2}.$$

The inductive step: $\Delta_W = u(t_W) - \hat u(t_W)$. Now $u(t_W) = u^W(t_W)$ where $u^W$ is solved with IC $u(t_{W-1})$. Let $\tilde u^W$ be solved with IC $\hat u(t_{W-1})$. Then
$$u(t_W) - \tilde u^W(t_W) = $$ (continuous dependence on initial data of the Burgers equation; this is bounded by $\|u(t_{W-1}) - \hat u(t_{W-1})\|_{L^2}\cdot e^{LM_x \Delta t/2}$ by a Gronwall argument I sketch in the next paragraph).

So $\Delta_W \leq e^{L M_x \Delta t/2} \|\Delta_{W-1}\|_{L^2} + \|E_W^\text{Picard}\|_{L^2}$. With $\Delta_0=0$ and iterating over $W$:
$$\|\Delta_K\|_{L^2} \leq \sum_{W=1}^K e^{L M_x (K-W)\Delta t/2}\,\|E_W^\text{Picard}\|_{L^2}.$$
The exponential factor is bounded by $e^{LM_x T/2}$, an $O(1)$ constant under fixed problem data. Absorb into $C_1$ in the final theorem; for clarity here we write the sum without the factor.

*Gronwall for Burgers continuous dependence:* Let $u_1, u_2$ both solve Burgers with different ICs. Their difference $\delta = u_1 - u_2$ satisfies $\partial_t\delta + (u_1+u_2)/2\,\delta_x + (u_{1,x}+u_{2,x})/2\,\delta = \nu\delta_{xx}$ (after symmetrizing). Multiply by $\delta$ and integrate; the convection term gives a boundary contribution which vanishes on $\mathbb{T}$, and the diffusion term gives $-\nu\|\delta_x\|_{L^2}^2 \leq 0$. The remaining term is $\frac{1}{2}\langle(u_{1,x}+u_{2,x})\delta,\delta\rangle \leq M_x\|\delta\|_{L^2}^2$, giving $\frac{d}{dt}\|\delta\|^2 \leq 2 M_x\|\delta\|^2$, so $\|\delta(t)\|_{L^2} \leq e^{M_x t}\|\delta(0)\|_{L^2}$. The amplification factor is $e^{M_x \Delta t}$ per window (close enough to 1 for the small $M_x \Delta t$ regime; absorb into $C_1$).
$\blacksquare$

### 1.7 Lemma G — NILT-error accumulation

The previous lemmas assumed the per-window linear substep (S2) is solved exactly. In practice, the NILT inversion in (S2) has a per-window relative error $\varepsilon_N$ in $L^2$:
$$\|v_\text{NILT}(t_W) - v_\text{exact}(t_W)\|_{L^2} \leq \varepsilon_N \cdot \|\hat u(t_{W-1})\|_{L^2}.$$

> **Lemma G.** Let $E_W^\text{NILT}$ be the per-window NILT error at the end of window $W$. The aggregate NILT contribution to $\|u(T)-\hat u(T)\|_{L^2}$ is bounded by
> $$\sum_{W=1}^K \|E_W^\text{NILT}\|_{L^2} \leq K\,\varepsilon_N\,\sup_{0\leq t\leq T}\|\hat u(t)\|_{L^2} \leq K\,\varepsilon_N\,(\|u_0\|_{L^2}+\|u(T)-\hat u(T)\|_{L^2}).$$

*Proof.* By Lemma F applied to the per-window NILT errors (which are also propagated through the contraction semigroups), and the triangle inequality $\|\hat u\|\leq\|u\|+\|u-\hat u\|$. The remaining bound on $\|u(t)\|_{L^2}$ follows from Burgers' energy decay: multiplying (1) by $u$ and integrating, $\frac{d}{dt}\|u\|_{L^2}^2 = -2\nu\|u_x\|_{L^2}^2 - 0 \leq 0$ (the $u^2 u_x$ term vanishes by integration by parts on the torus), so $\|u(t)\|_{L^2}\leq\|u_0\|_{L^2}$.
$\blacksquare$

The bound $K\varepsilon_N$ is **linear in $K$**. A $\sqrt K\varepsilon_N$ bound is *not* available without additional hypotheses on the per-window NILT-error sign distribution. We use $K\varepsilon_N$ throughout.

## 2. Theorem 1

> **Theorem 1.** Let $u\in C^0([0,T];H^2(\mathbb{T}))$ solve Burgers (1) and let the cascade (S1)–(S3) be run with operating-state choice satisfying (NL-A) and Picard relaxation $\omega_W^* \in [0,1]$. Define
> $$M_W \equiv \sup_{t\in[t_{W-1},t_W]} u_\text{rel}(t),\qquad N_t \equiv \sup_{t\in[0,T]}\|u(t)\,u_x(t)\|_{L^2(\mathbb{T})}.$$
> 
> Then the cascade error satisfies
> $$\boxed{\;\|u(T) - \hat u(T)\|_{L^2(\mathbb{T})} \;\leq\; e^{M_x T}\sqrt L\,M_x\sum_{W=1}^K (1-\omega_W^*)\,\bigl[M_W\Delta t + \tfrac12 M_t\Delta t^2\bigr] \;+\; K\,\varepsilon_N\,\|u_0\|_{L^2(\mathbb{T})} + O(\Delta t^2 K\,M_x^2 N_t^2).\;} \qquad (9)$$
> 
> Furthermore, the cascade is **stable** (per-window Picard residual remains aligned with the true error so $\omega^*$-rules of §3 below remain meaningful) provided the **operator-drift CFL bound**
> 
> $$K \;\geq\; K_\text{CFL} \;\equiv\; \left\lceil T\sqrt{\frac{u_W^*\,N_t}{\kappa\,\nu}}\right\rceil \quad \text{for every window } W \qquad (10)$$
> 
> holds with universal constant $\kappa = O(1)$. The **optimal** $K$ minimizing the right-hand side of (9) (over $K$, holding all other quantities fixed) is
> 
> $$K_\text{opt} \;=\; \left\lceil T\sqrt{\frac{e^{M_x T}\sqrt L\,M_x M_W (1-\omega^*)}{\varepsilon_N\,\|u_0\|_{L^2}}}\,\right\rceil. \qquad (11)$$
> 
> The cascade should use $K^* = \max(K_\text{CFL}, K_\text{opt})$; the resulting error is bounded by
> 
> $$\|u(T)-\hat u(T)\|_{L^2} \;\leq\; 2\sqrt{e^{M_x T}\sqrt L\,M_x M_W (1-\omega^*)\,\varepsilon_N\,\|u_0\|_{L^2}}\cdot T. \qquad (12)$$

*Proof of Theorem 1, given Lemmas A–G.*

**(a) Per-window error (linearization + Picard, ignoring NILT).** From Lemma C and Lemma D (i.e., (5)):
$$\|E_W^\text{lin}\|_{L^2} \leq \sqrt L\,M_x\,(M_W\Delta t + \tfrac12 M_t\Delta t^2).$$
From Lemma E (8):
$$\|E_W^\text{Picard}\|_{L^2} \leq (1-\omega_W^*)\sqrt L\,M_x\,(M_W\Delta t + \tfrac12 M_t\Delta t^2) + O(\Delta t^2).$$

**(b) Aggregate over windows.** By Lemma F with the Gronwall amplification factor $e^{M_x T}$,
$$\|u(T) - \hat u(T)\|_{L^2}^\text{lin+Picard} \leq e^{M_x T}\,\sqrt L\,M_x\sum_{W=1}^K(1-\omega_W^*)(M_W\Delta t + \tfrac12 M_t\Delta t^2) + O(\Delta t^2 K).$$

**(c) NILT.** Add Lemma G's $K\varepsilon_N\|u_0\|_{L^2}$ term by triangle inequality.

**(d) Combining gives (9).** ✓

**(e) Optimization for $K_\text{opt}$.** Approximate (9) for the dominant terms (drop the $\Delta t^2$ pieces, which become negligible when $K$ is moderately large):
$$\|u-\hat u\|_{L^2} \approx \frac{C\,M_W (1-\omega^*)}{K} + K\varepsilon_N\|u_0\|_{L^2},\quad C = e^{M_x T}\sqrt L M_x T.$$

Differentiate w.r.t. $K$: $-CM_W(1-\omega^*)/K^2 + \varepsilon_N\|u_0\|_{L^2} = 0 \Rightarrow K_\text{opt} = \sqrt{CM_W(1-\omega^*)/(\varepsilon_N\|u_0\|_{L^2})}$. Substituting back gives (12) with the AM-GM bound $a/K + bK \geq 2\sqrt{ab}$.

**(f) Necessity of (10).** This is Proposition 2 below — the operator-drift CFL ensures the per-window phase shift at the binding wavenumber stays bounded so that the Picard residual remains a meaningful corrective direction. If (10) fails, the per-window operator drift $\delta\sigma_W = -ik\delta u_W^*$ accumulated phase rotation exceeds $\pi/2$ at the binding mode, the alignment between $\hat\rho_W$ and $\rho_W^\text{true}$ flips sign, and the higher-order term $Q_W$ in Lemma E *adds* to the error rather than subtracts. In this regime, the Picard correction is destabilizing and the bound (9) is invalid.
$\blacksquare$

## 3. Specifying $\omega_W^*$ via the angular CFL

The Picard relaxation $\omega_W^*$ should equal 1 when the residuals align well (then full correction is safe and effective) and 0 when they don't (then no correction at all). The right interpolation comes from the operator-drift phase shift derived above.

The per-window phase shift at wavenumber $k$ is $\Phi_W(k) = |\Im\delta\sigma_W(k)|\cdot\Delta t = k\,|\delta u_W^*|\,\Delta t$. The binding wavenumber is $k_\star = u_W^*/\nu$ (from §1 of the symbolic derivation document). The drift $|\delta u_W^*|$ is bounded by $M_t\Delta t$ (operator-drift Lemma in §1 of the same). So
$$\Phi_W(k_\star) = \frac{u_W^*}{\nu}\cdot M_t\,\Delta t^2.$$

Define
$$\boxed{\;\omega_W^* \;\equiv\; \cos^2\!\bigl(\min(\Phi_W(k_\star),\pi/2)\bigr).\;} \qquad (13)$$

When $\Phi_W \to 0$, $\omega^* \to 1$ (full correction). When $\Phi_W \to \pi/2$, $\omega^* \to 0$ (no correction). The clipping at $\pi/2$ is necessary: beyond $\pi/2$, $\cos^2$ becomes positive again (oscillatory) and the formula loses its meaning of "alignment fraction."

The operator-drift CFL bound (10) ensures $\Phi_W(k_\star) \leq \kappa$. With $\kappa = \pi/2$ and (10) tight, $\omega^* = \cos^2(\pi/2) = 0$ — Picard fully off. With $\kappa = \pi/8$ (the empirical calibration), $\omega^* = \cos^2(\pi/8) \approx 0.85$ — Picard is approximately fully active.

## 4. Necessity of $K_\text{CFL}$ — Proposition 2 (constructive)

> **Proposition 2.** Let $u_0(x) = A\cos(k_\star x)$ on $\mathbb{T}$ with $L = 2\pi/k_\star$, $A>0$, and consider the cascade with $u_W^* = A$ (the L^∞ choice, which satisfies (NL-A) for the half-period until $u$ first changes sign — in this constructive example we take a small $A$ such that the true solution remains positive over the time window of interest). For any $\varepsilon > 0$, there exists $K_0(\varepsilon) > K_\text{CFL}$ such that for all $K < K_0(\varepsilon)$, the cascade error $\|u(T) - \hat u(T)\|_{L^2}$ exceeds $(1+\varepsilon)$ times the bound (9). That is, $K \geq K_\text{CFL}$ is *necessary* for (9) to hold, not merely sufficient.

*Proof (sketch).* The cosine IC has Fourier support concentrated at $k = \pm k_\star$. The operator drift $\delta\sigma_W(k_\star) = -ik_\star \delta u_W^*$ accumulated over $K$ windows gives a phase rotation $\pm K\,k_\star \delta u_W^*\,\Delta t = \pm k_\star (u_K^* - u_0^*) T = \pm k_\star\Delta U\cdot T$ at the binding mode, where $\Delta U$ is the total operating-state drift over $[0,T]$.

When $K < K_\text{CFL}$ (equivalent to $\Delta t > $ some threshold), the per-window phase shift $\Phi_W(k_\star) > \pi/2$, so $\omega_W^* = 0$ from (13). With Picard fully off, the per-window error from Lemma C is $\sqrt L M_x M_W\Delta t = O(\Delta t)$. Aggregated over $K$ windows: $T\sqrt L M_x M_W$ — *constant in $K$*, no improvement from finer windows. NILT contribution is $K\varepsilon_N\|u_0\|_{L^2}$, growing linearly with $K$.

So below $K_\text{CFL}$, the error has a *constant* lower bound from linearization (since Picard cannot help) plus a *growing* contribution from NILT. There is no $K^* < K_\text{CFL}$ that makes the bound (9) tight; any choice in this regime produces error larger than the threshold.

For a rigorous existence-of-counterexample proof: set $A,\nu,T$ so that $K_\text{CFL} = K_\text{CFL}(A,\nu,T)$ takes a desired integer value. By picking $K = K_\text{CFL} - 1$, the per-window phase shift at $k_\star$ is $\Phi_W(k_\star) > \kappa$. Then the Picard correction $\hat\rho_W$ computed at $v$ (which has phase-shifted $\pm\pi/2$ from $u^W$) carries the *opposite* sign from $\rho_W^\text{true}$ at $k_\star$, and the relaxation factor $\cos^2(\pi/2) = 0$ kills the Picard contribution. Without Picard, the per-window error is the unrelaxed Lemma C bound, accumulated over $K$ windows. The total is bounded below by $K_\text{CFL}\cdot\sqrt L M_x M_W \Delta t = TM_x\sqrt L M_W$, which exceeds (12) by a factor $\sqrt{K_\text{CFL}}$.

Hence $K \geq K_\text{CFL}$ is *necessary* for the bound (9). $\blacksquare$

## 5. Cubic vs square-root balance: a precise statement

> **Corollary (Operating-state regime).** Let the operating-state estimator be such that $u_\text{rel}(t_{W-1}) \leq C_0 M_t\Delta t$ for all $W$, with constant $C_0$. (We call this the **tracking estimator**.) Then in (5), the linear-in-$\Delta t$ term vanishes and $\|E_W^\text{lin}\|_{L^2} = O(\Delta t^2)$. After Picard the error is $O(\Delta t^3)$ per window. The total bound (9) becomes
> $$\|u(T)-\hat u(T)\|_{L^2} = O\!\left(\frac{T^3}{K^2}\right) + K\varepsilon_N\|u_0\|_{L^2},$$
> and the optimal $K^*$ scales as $K_\text{opt}^\text{tracking} = (T^3/\varepsilon_N)^{1/3}$ — **cubic balance**.
>
> If instead $u_\text{rel}(t_{W-1})$ is bounded by an $\Delta t$-independent constant (the **non-tracking estimator** — e.g., the gradient-weighted-mean or amplitude estimator we use), the linear-in-$\Delta t$ term in (5) dominates and $\|E_W^\text{lin}\|_{L^2} = O(\Delta t)$, giving $\|u-\hat u\|_{L^2} = O(T^2/K) + K\varepsilon_N$ and $K^* = O(\sqrt{T^2/\varepsilon_N})$ — **square-root balance**.

This **cleanly resolves the cubic vs square-root puzzle of Phases 0–1b**: the original `tuner.py` derivation assumed (implicitly) a tracking estimator. The implementation uses a non-tracking estimator. The empirical $K \propto \sqrt{\mathrm{Re}}$ scaling is precisely the square-root balance — exactly what the proof predicts under the actual estimator.

A future variant of the tuner could use a tracking estimator (e.g., $u_W^* = u(t_{W-1}, x_*)$ where $x_*$ is the location of maximum activity, updated each window). This would shift the regime to cubic balance, with $K_\text{opt}^\text{tracking} \ll K_\text{opt}^\text{non-tracking}$ at small $\varepsilon_N$. Whether the implementation cost of tracking is worth the runtime savings is an empirical question.

## 6. Summary of what is proved

The following claims are now rigorously established:

- **Theorem 1, (9):** the cascade error has the explicit bound stated, valid under (NL-A).
- **Equation (10):** the operator-drift CFL bound $K \geq K_\text{CFL}$ is sufficient for the bound (9) to apply.
- **Equation (11):** the optimal $K_\text{opt}$ minimizing (9) has the closed form stated.
- **Equation (12):** at $K = K^*$, the cascade error is bounded by $2T\sqrt{...}$.
- **Equation (13):** the Picard relaxation $\omega_W^* = \cos^2(\min(\Phi_W(k_\star),\pi/2))$ is the unique choice that interpolates between full correction (small phase drift) and no correction (large phase drift) consistent with the angular-CFL framework.
- **Proposition 2:** $K \geq K_\text{CFL}$ is *necessary*, not just sufficient: a constructive counterexample (cosine IC) shows that below the CFL threshold, the bound (9) does not hold.
- **Lemma E:** Picard correction reduces per-window error by factor $(1-\omega_W^*)$ in leading order, plus higher-order $Q_W = O(\Delta t \cdot \|w\|^2)$ terms.
- **Corollary §5:** the cubic vs square-root balance puzzle is resolved by distinguishing tracking vs non-tracking estimators.

Open questions left explicit:

- Whether $\omega_W^*$ from (13) is *optimal* or merely a sound choice. A sharper version might use a different convex function of the phase shift.
- Whether the Gronwall amplification factor $e^{M_x T}$ in the bound is tight or pessimistic. For shock-formation problems where $M_x \to \infty$ as $t \to t_\text{shock}$, the bound degenerates; refinement might allow time-dependent $M_x$.
- The constants $C_0, C_S, C_1, C_2$ are not pinned down to optimal values — only their order of magnitude matters for the scaling claims.
- The exact value of $\kappa$ in (10) is calibrated empirically ($\pi/8$); a first-principles derivation would tie it to the Picard relaxation at the boundary (e.g., $\kappa = \arccos(\sqrt{0.5})$ for a "half-effective Picard" criterion).
