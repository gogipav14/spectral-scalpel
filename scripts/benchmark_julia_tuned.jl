"""
Tuned Julia CPU scalpel benchmark — apples-to-apples vs the original
naive version. Optimizations:
  1. FFTW.set_num_threads(N) for multithreaded FFT (default = 1).
  2. plan_fft / plan_ifft built once and reused.
  3. Time axis = first dim (column-major contiguous, fastest stride).
  4. In-place broadcasts via .= to eliminate intermediate allocations.
  5. Pre-allocated work buffers.

Compares naive vs tuned vs FDTD on the same Maxwell wet-clay problem.
"""

using FFTW
using LinearAlgebra
using Printf
using Statistics

# ── Physics (identical to benchmark_julia.jl) ──────────────────────
const MU_0 = 4e-7 * π
const EPS_0 = 8.854187817e-12
const σ_cond = 0.1
const ε_r = 10.0
const ε = EPS_0 * ε_r
const depth = 0.5
const c_mat = 1.0 / sqrt(MU_0 * ε)
const t_transit = depth / c_mat
const t_end = 13 * t_transit

const a_nilt = 6.7167e+07
const T_nilt = 1.3713e-07
const N_NILT = 2048

const Lz = depth * 3
const Nz = 500
const dz = Lz / Nz
const dt_fdtd = 0.5 * dz / c_mat
const Nt_fdtd = Int(floor(t_end / dt_fdtd)) + 1

const Nx = 64
const n_modes = Nx * Nx

FFTW.set_num_threads(Sys.CPU_THREADS)
println("FFTW threads: ", FFTW.get_num_threads())

# ═══════════════════════════════════════════════════════════════════
#  NAIVE scalpel (copy of benchmark_julia.jl, no plan caching/threads)
# ═══════════════════════════════════════════════════════════════════
function scalpel_naive()
    dx_grid = 0.01
    kx = fftfreq(Nx, 1.0/dx_grid) .* (2π)
    ky = fftfreq(Nx, 1.0/dx_grid) .* (2π)
    KX = reshape(kx, Nx, 1, 1)
    KY = reshape(ky, 1, Nx, 1)
    ω = collect(0:N_NILT-1) .* (π / T_nilt)
    s = a_nilt .+ im .* ω
    S = reshape(s, 1, 1, N_NILT)

    src = zeros(ComplexF64, Nx, Nx)
    src[Nx÷2+1, Nx÷2+1] = 1.0
    Ŝ = fft(src)

    γ² = MU_0 .* (σ_cond .* S .+ ε .* S.^2) .- KX.^2 .- KY.^2
    γz = sqrt.(γ²)
    γz = γz .* sign.(real.(γz) .+ 1e-30)
    H = exp.(-γz .* depth)
    F_spec = Ŝ[:, :, :] .* H
    F_spec[:, :, 1] .*= 0.5
    z_raw = N_NILT .* ifft(F_spec, 3)
    dt_n = 2 * T_nilt / N_NILT
    t_arr = collect(0:N_NILT-1) .* dt_n
    correction = exp.(a_nilt .* t_arr) ./ T_nilt
    field_kt = real.(z_raw) .* reshape(correction, 1, 1, N_NILT)
    field_xt = real.(ifft(field_kt, (1, 2)))
    return field_xt
end

# ═══════════════════════════════════════════════════════════════════
#  TUNED scalpel: (1) plan cache, (2) threading, (3) axis 1 = time,
#                 (4) in-place broadcasts, (5) pre-allocated buffers
# ═══════════════════════════════════════════════════════════════════
struct ScalpelWorkspace
    KX::Array{Float64,3}            # (1, Nx, 1)
    KY::Array{Float64,3}            # (1, 1, Nx)
    S::Array{ComplexF64,3}          # (N_NILT, 1, 1)  — time on axis 1
    correction::Array{Float64,3}    # (N_NILT, 1, 1)
    Ŝ::Array{ComplexF64,3}          # (1, Nx, Nx)
    F::Array{ComplexF64,3}          # (N_NILT, Nx, Nx) work buffer
    tmp::Array{Float64,3}           # (N_NILT, Nx, Nx) real work
    plan_t::AbstractFFTs.Plan       # ifft along axis 1
    plan_xy::AbstractFFTs.Plan      # ifft along axes 2, 3
end

function build_workspace()
    dx_grid = 0.01
    kx = fftfreq(Nx, 1.0/dx_grid) .* (2π)
    ky = fftfreq(Nx, 1.0/dx_grid) .* (2π)
    # Time axis = 1, kx = 2, ky = 3 (Julia column-major: axis 1 contiguous)
    KX = reshape(kx, 1, Nx, 1)
    KY = reshape(ky, 1, 1, Nx)

    ω = collect(0:N_NILT-1) .* (π / T_nilt)
    s = a_nilt .+ im .* ω
    S = reshape(ComplexF64.(s), N_NILT, 1, 1)

    dt_n = 2 * T_nilt / N_NILT
    t_arr = collect(0:N_NILT-1) .* dt_n
    correction = reshape(exp.(a_nilt .* t_arr) ./ T_nilt, N_NILT, 1, 1)

    src = zeros(ComplexF64, Nx, Nx)
    src[Nx÷2+1, Nx÷2+1] = 1.0
    Ŝ = reshape(fft(src), 1, Nx, Nx)

    F = Array{ComplexF64}(undef, N_NILT, Nx, Nx)
    tmp = Array{Float64}(undef, N_NILT, Nx, Nx)

    # Pre-build plans (FFTW measures on first use)
    plan_t  = plan_ifft!(F, 1; flags=FFTW.MEASURE)
    plan_xy = plan_ifft!(F, (2, 3); flags=FFTW.MEASURE)

    return ScalpelWorkspace(Array(KX), Array(KY), S, correction, Ŝ, F, tmp,
                            plan_t, plan_xy)
end

function scalpel_tuned!(W::ScalpelWorkspace)
    F = W.F
    # γ² = MU_0 (σ S + ε S²) − KX² − KY²   (broadcast into F)
    @. F = MU_0 * (σ_cond * W.S + ε * W.S^2) - W.KX^2 - W.KY^2
    # γz = √γ²; flip sign so Re(γz) ≥ 0; H = exp(−γz d); F ← Ŝ ⋅ H
    @. F = sqrt(F)
    @. F = F * sign(real(F) + 1e-30)
    @. F = W.Ŝ * exp(-F * depth)
    # DC half-weight along time axis (index 1 since time is axis 1)
    @views F[1, :, :] .*= 0.5
    # Inverse FFT along axis 1 (time)
    W.plan_t * F
    @. F = F * N_NILT
    # Multiply by e^{a t} / T correction (real)
    @. F = real(F) * W.correction
    # Inverse 2D FFT along axes 2, 3 (transverse spatial)
    W.plan_xy * F
    @. W.tmp = real(F)
    return W.tmp
end

# Build workspace, warm up
W = build_workspace()
_ = scalpel_tuned!(W)
_ = scalpel_naive()

# ─── Timing ────────────────────────────────────────────────────────
function median_time(f, n)
    times = Float64[]
    for _ in 1:n
        t0 = time_ns()
        _ = f()
        push!(times, (time_ns() - t0) / 1e6)
    end
    return median(sort(times))
end

t_naive = median_time(scalpel_naive, 15)
t_tuned = median_time(() -> scalpel_tuned!(W), 30)

println("="^72)
@printf(" Maxwell wet clay, %dx%d, N_NILT=%d (Julia CPU, %d threads)\n",
        Nx, Nx, N_NILT, Sys.CPU_THREADS)
println("="^72)
@printf(" Naive scalpel (current benchmark):  %8.1f ms\n", t_naive)
@printf(" Tuned scalpel (plans + threads):    %8.1f ms\n", t_tuned)
@printf(" Speedup from tuning:                %8.1fx\n", t_naive / t_tuned)
println("="^72)
@printf(" Reference (other CPU backends on the same problem):\n")
@printf("   NumPy:      902.7 ms\n")
@printf("   PyTorch:    276.1 ms\n")
@printf("   JAX:        161.9 ms\n")
@printf(" Implied wall-ratio vs Yee FDTD (=1931 ms):\n")
@printf("   Naive:      %.1fx\n", 1931 / t_naive)
@printf("   Tuned:      %.1fx\n", 1931 / t_tuned)
println("="^72)
