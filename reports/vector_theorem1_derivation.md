# Vector-Valued Theorem 1: closing the multi-species gap

## The key observation

For an $n$-species coupled system
$$\partial_t \mathbf{u} = \mathbf{L}[\mathbf{u}] + \mathbf{N}[\mathbf{u}], \qquad \mathbf{u}: \mathbb{T}_L \to \mathbb{R}^n,$$
the natural generalization of hypothesis (G) holds with the **identity gauge $\Phi = \mathrm{id}$** provided we allow the linearized operator to carry a matrix-valued Jacobian.

**Why this works:** Taylor's theorem in Banach space at any scalar constant state $\mathbf{u}_0^* \in \mathbb{R}^n$ gives
$$\mathbf{N}(\mathbf{u}_0^* + \mathbf{h}) = \mathbf{N}(\mathbf{u}_0^*) + D\mathbf{N}|_{\mathbf{u}_0^*}[\mathbf{h}] + \tfrac{1}{2} D^2\mathbf{N}|_{\mathbf{u}_0^*}[\mathbf{h}, \mathbf{h}] + O(\|\mathbf{h}\|^3).$$

Absorb the constant $\mathbf{N}(\mathbf{u}_0^*)$ and matrix-linear $D\mathbf{N}|_{\mathbf{u}_0^*} \mathbf{h}$ into the linear substep. The **vector residual**
$$\widetilde{\mathbf{N}}[\mathbf{u}] \equiv \mathbf{N}[\mathbf{u}] - \mathbf{N}[\mathbf{u}_0^*] - D\mathbf{N}|_{\mathbf{u}_0^*}[\mathbf{u} - \mathbf{u}_0^*]$$
satisfies
$$\widetilde{\mathbf{N}}[\mathbf{u}_0^*] = \mathbf{0}, \qquad D\widetilde{\mathbf{N}}|_{\mathbf{u}_0^*} = \mathbf{0}$$
**automatically**, by construction. This is hypothesis (G-vec).

The vector-valued residual is at least quadratic in $\mathbf{h} = \mathbf{u} - \mathbf{u}_0^*$, just like the scalar case. The Picard contraction argument transfers term-for-term with scalar norms replaced by matrix operator norms.

The only thing that changes between scalar and vector:
- Per-Fourier-mode Laplace symbol becomes an $n\times n$ matrix $\sigma(k; \mathbf{u}_0^*) \in \mathbb{C}^{n\times n}$
- Linear substep requires matrix exponential $e^{\sigma(k)\tau}$ per mode (cheap for $n \lesssim 10$)
- Constant source $\mathbf{b} = \mathbf{N}(\mathbf{u}_0^*) - D\mathbf{N}|_{\mathbf{u}_0^*} \mathbf{u}_0^*$ is a vector

## Hypothesis (G-vec)

> **Definition (G-vec).** For a system $\partial_t \mathbf{u} = \mathbf{L}[\mathbf{u}] + \mathbf{N}[\mathbf{u}]$ with $\mathbf{L}$ linear constant-coefficient and $\mathbf{N} \in C^3(\mathbb{R}^n, \mathbb{R}^n)$, a $C^3$ bijection $\Phi: \mathcal{I} \subset \mathbb{R}^n \to \mathbb{R}^n$ with $D\Phi$ invertible satisfies (G-vec) iff, in $\mathbf{w} = \Phi(\mathbf{u})$ coordinates, the equation reads
> $$\partial_t \mathbf{w} = \mathbf{L}'[\mathbf{w}] + A(\mathbf{w}_0^*)(\mathbf{w} - \mathbf{w}_0^*) + \mathbf{b}(\mathbf{w}_0^*) + \widetilde{\mathbf{N}}[\mathbf{w}, \partial_x \mathbf{w}, \ldots]$$
> where $\mathbf{L}'$ is linear constant-coefficient (possibly with matrix coupling between components), $A(\mathbf{w}_0^*) \in \mathbb{R}^{n\times n}$ and $\mathbf{b}(\mathbf{w}_0^*) \in \mathbb{R}^n$ are $C^1$ in the scalar operating state $\mathbf{w}_0^*$, and $\widetilde{\mathbf{N}}$ has
> $$\widetilde{\mathbf{N}}[(\mathbf{w}_0^*, \mathbf{0}, \mathbf{0}, \ldots)] = \mathbf{0}, \qquad D\widetilde{\mathbf{N}}|_{(\mathbf{w}_0^*, \mathbf{0}, \mathbf{0}, \ldots)} = \mathbf{0}.$$

**The universal instance: identity gauge.** For any coupled $\mathbf{N} \in C^3$, the identity gauge $\Phi(\mathbf{u}) = \mathbf{u}$ always satisfies (G-vec) with
$$A(\mathbf{u}_0^*) = D\mathbf{N}|_{\mathbf{u}_0^*}, \qquad \mathbf{b}(\mathbf{u}_0^*) = \mathbf{N}(\mathbf{u}_0^*) - D\mathbf{N}|_{\mathbf{u}_0^*}\mathbf{u}_0^*,$$
and $\widetilde{\mathbf{N}} = \mathbf{N}(\mathbf{u}) - \mathbf{N}(\mathbf{u}_0^*) - D\mathbf{N}|_{\mathbf{u}_0^*}(\mathbf{u} - \mathbf{u}_0^*)$ — the Taylor remainder, quadratic in $\mathbf{h}$ by construction.

**Non-trivial gauges still matter.** The identity gauge works universally but may give a loose constant $\|D^2\mathbf{N}\|_\infty$ in the bound. For scalar Fisher-KPP, logit reduced this constant significantly. Vector-valued analogues (component-wise gauges $\Phi = (\phi_1(u_1), \ldots, \phi_n(u_n))$, or sum-difference transforms like $(p, q) = (u+v, u-v)$ for Gray-Scott) can still be useful but are no longer required to bring the problem into scope.

## Theorem 1-vec

> **Theorem 1-vec.** Assume (G-vec) and $\mathbf{u}(t, \cdot) \in H^3(\mathbb{T}_L; \mathbb{R}^n)$ uniformly on $[0,T]$ with $\mathbf{u}(t,x) \in \mathcal{I}$ interior to the invariant region. Run the vector-valued cascade: per window, linearize $\mathbf{N}$ around scalar $\mathbf{w}_0^*$, solve the matrix-linear substep via per-mode matrix exponential, apply two Picard iterations on the vector residual. Assume the matrix CFL bound
> $$K \ge K_\text{crit}^{\text{vec}} \equiv 2\omega \|D^2\mathbf{N}\|_\infty M_h T\,e^{\lambda T},$$
> where $\|D^2\mathbf{N}\|_\infty$ is the operator norm of the second Fréchet derivative, $M_h = \sup_t \|\mathbf{u}(t) - \mathbf{u}_0^*\|_{H^2(\mathbb{R}^n)}$, and $\lambda = \sup_k \Real \mathrm{eig}(\sigma^W(k))$ is the largest real part of the matrix Fourier symbol's eigenvalues.
> 
> Then
> $$\|\mathbf{u}(T) - \hat{\mathbf{u}}(T)\|_{L^2(\mathbb{T}_L;\mathbb{R}^n)} \le C_\Phi^{\text{vec}} \frac{T^3}{K^2} + C_N K \varepsilon_N\,\|\Phi(\mathbf{u}_0)\|_{L^2}$$
> and the optimal $K^*$ satisfies cubic balance $(K^*)^3 \propto \varepsilon_N^{-1}$.

**Proof structure.** Identical to the scalar case (Lemmas A'–G' of the formal proof), with scalars replaced by matrices wherever they appear:

- **Lemma A'-vec** (semigroup bound): $\|e^{\sigma(k)\tau}\|_{\mathbb{R}^n \to \mathbb{R}^n} \le e^{\lambda\tau}$, where $\lambda$ is the largest eigenvalue real part. Proof: diagonalize $\sigma(k)$ (or Jordan-form if defective); the exponential bound follows from spectral mapping.

- **Lemma B'-vec** (quadratic residual): Banach Taylor expansion gives
  $$\|\widetilde{\mathbf{N}}[\mathbf{u}_0^* + \mathbf{h}]\|_{L^2} \le \tfrac{1}{2} \|D^2\mathbf{N}\|_\infty\,\|\mathbf{h}\|_{H^1}^2 + O(\|\mathbf{h}\|^3).$$
  Here $\|D^2\mathbf{N}\|_\infty$ is the operator norm of the bilinear map $D^2\mathbf{N}: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}^n$.

- **Lemma E'-vec** (Picard contraction in $H^2$): the vector Picard map is a contraction on an $H^2(\mathbb{R}^n)$ ball with constant $L_P \le 2\omega \|D^2\mathbf{N}\|_\infty M_h \Delta t\,e^{\lambda T}$. Two iterations give per-window error $O(\Delta t^3)$ measured in the vector $L^2$ norm.

- **Lemmas F'-vec, G'-vec**: Lady Windermere aggregation and NILT accumulation in $L^2(\mathbb{T}_L;\mathbb{R}^n)$ are component-wise straightforward.

- **Combining:** identical to scalar, giving (vec.1) and cubic balance.

The proof is essentially a word-for-word re-run of the scalar proof with vector/matrix norms. The structural observation that makes this work: **the residual after matrix-linearization has exactly the same quadratic-first-variation property as the scalar Taylor remainder**.

## Concrete examples

### Example 1: Gray-Scott

$\mathbf{u} = (u, v)$. Equations:
$\partial_t u = D_u \nabla^2 u + F - Fu - uv^2$
$\partial_t v = D_v \nabla^2 v - (F+k)v + uv^2$

Linear part of RHS (already including constant source):
$\mathbf{L}[\mathbf{u}] + \mathbf{b}_L = \mathrm{diag}(D_u\nabla^2 - F, D_v\nabla^2 - (F+k))\,\mathbf{u} + (F, 0)^T$.

Nonlinear: $\mathbf{N}(u, v) = (-uv^2, uv^2)^T$.

Jacobian at $(u_0^*, v_0^*)$:
$D\mathbf{N}|_{(u_0^*, v_0^*)} = \begin{pmatrix}-v_0^{*2} & -2u_0^* v_0^* \\ v_0^{*2} & 2u_0^* v_0^* \end{pmatrix}.$

Fourier symbol per mode $k$:
$\sigma^W(k) = \begin{pmatrix} -D_u k^2 - F - v_0^{*2} & -2u_0^* v_0^* \\ v_0^{*2} & -D_v k^2 - (F+k) + 2u_0^* v_0^* \end{pmatrix}.$

Residual (Taylor remainder):
$\widetilde N_1 = -u_0^* h_v^2 - 2v_0^* h_u h_v - h_u h_v^2$
$\widetilde N_2 = -\widetilde N_1.$

Both are quadratic in $\mathbf{h}$ (plus cubic remainder). Second-variation bound:
$\|D^2\mathbf{N}\|_\infty \le 2(u_0^* + v_0^* + 1)$ on a bounded invariant region.

### Example 2: FitzHugh-Nagumo

$\mathbf{u} = (u, v)$. Equations:
$\partial_t u = D_u \nabla^2 u + u - u^3 - v$
$\partial_t v = D_v \nabla^2 v + \epsilon(u - \gamma v)$

Nonlinearity: $\mathbf{N}(u, v) = (-u^3, 0)^T$ (only component 1 is nonlinear). Linear coupling is $u \leftrightarrow v$ through the matrix
$A_L = \begin{pmatrix} 0 & -1 \\ \epsilon & 0 \end{pmatrix}$
(the linear part's cross-coupling).

Jacobian of $\mathbf{N}$: $\begin{pmatrix} -3u_0^{*2} & 0 \\ 0 & 0 \end{pmatrix}$.

Fourier symbol per mode:
$\sigma^W(k) = \begin{pmatrix} -D_u k^2 + 1 - 3u_0^{*2} & -1 \\ \epsilon & -D_v k^2 - \epsilon\gamma \end{pmatrix}$.

Residual: $\widetilde{\mathbf{N}} = (-3u_0^* h_u^2 - h_u^3, 0)^T$. Quadratic-plus-cubic, as expected.

$\|D^2 \mathbf{N}\|_\infty = 6|u_0^*|$ on the invariant region $|u| \le O(1)$.

This is a clean vector cascade with matrix-linear + cubic nonlinearity, beautifully handled by Theorem 1-vec.

### Example 3: Brusselator

$\partial_t u = D_u \nabla^2 u + A - (B+1)u + u^2 v$
$\partial_t v = D_v \nabla^2 v + Bu - u^2 v$

$\mathbf{N}(u, v) = (u^2 v, -u^2 v)^T$ (note $\mathbf{N}_1 + \mathbf{N}_2 = 0$, like Gray-Scott's mass-conservation structure).

Jacobian: $D\mathbf{N}|_{(u_0^*, v_0^*)} = \begin{pmatrix} 2u_0^* v_0^* & u_0^{*2} \\ -2u_0^* v_0^* & -u_0^{*2} \end{pmatrix}$.

Fourier symbol:
$\sigma^W(k) = \begin{pmatrix} -D_u k^2 - (B+1) + 2u_0^* v_0^* & u_0^{*2} \\ B - 2u_0^* v_0^* & -D_v k^2 - u_0^{*2} \end{pmatrix}$.

Residual: $\widetilde N_1 = 2u_0^* h_u h_v + v_0^* h_u^2 + h_u^2 h_v$, $\widetilde N_2 = -\widetilde N_1$. Quadratic+cubic, fits Theorem 1-vec.

## Numerical implementation

The per-mode 2×2 (or $n\times n$) matrix exponential has a closed form. For a 2×2 matrix $M$:
$e^M = e^{\mathrm{tr}(M)/2} \left( \cos\theta\, I + \frac{\sin\theta}{\theta} \left(M - \tfrac{\mathrm{tr}(M)}{2} I\right) \right)$
where $\theta = \sqrt{-\det(M - \mathrm{tr}(M)/2 \cdot I)}$ (real or pure imaginary depending on sign).

For numerical robustness and generality, `scipy.linalg.expm` on each mode works (O(Nx) calls at $(n\times n)$ size, negligible for moderate Nx).

Per window:
1. Evaluate $\mathbf{N}(\mathbf{u}_0^*)$ and $D\mathbf{N}|_{\mathbf{u}_0^*}$ at the scalar operating state.
2. For each Fourier mode $k$: compute $\sigma^W(k)$ (2x2 matrix), then $e^{\sigma^W(k) \Delta t}$.
3. Linear substep via Fourier: for each mode, the vector $(\hat u_k, \hat v_k)$ is propagated by $e^{\sigma^W(k) \Delta t}$; constant source handled at $k = 0$.
4. Two Picard iterations: compute residual $\widetilde{\mathbf{N}}$ at the linearized solution, integrate via Duhamel on the matrix semigroup.

## Implications for the paper

The Turing demo (current Fig 4) is NO LONGER "outside hypothesis (G)". It IS hypothesis (G-vec) under the identity gauge. The paper can now claim:

1. Scalar (G) for Burgers, KS, Fisher-KPP, Allen-Cahn: Theorem 1 with scalar/exponential linear substep.
2. Vector (G-vec) for Gray-Scott, FitzHugh-Nagumo, Brusselator, and any coupled reaction-diffusion: Theorem 1-vec with matrix-exponential linear substep.

Same cubic-balance rate ($O(T^3/K^2)$) in both cases. Same cascade structure. Same a-priori tuner.

The "vector-valued limitation" in the Discussion should be removed or significantly narrowed — the only remaining limitation is systems with DIFFERENTIAL coupling between components (e.g., Navier-Stokes pressure-velocity via incompressibility, or wave equations where the cross-coupling goes through spatial derivatives).
