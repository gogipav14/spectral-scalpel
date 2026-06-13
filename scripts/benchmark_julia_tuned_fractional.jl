"""
Tuned fractional Caputo Julia CPU benchmark.

Same problem as benchmark_julia_fractional.jl. Compares naive scalpel
implementation against a properly tuned version with FFTW plans + threads
+ axis-1 time layout.
"""

using FFTW
using LinearAlgebra
using Printf
using Statistics

const ALPHA = 0.7
const D_diff = 1e-2
const L_dom = 0.32
const Nx = 32
const Ny = 32
const dx = L_dom / Nx
const sigma_src = 0.03
const d_obs = 0.10
const t_end = 0.020
const N_NILT = 2048

FFTW.set_num_threads(Sys.CPU_THREADS)

# Source
function build_src()
    x = [(i - Nx / 2 - 0.5) * dx for i in 1:Nx]
    src = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        src[i, j] = exp(-(x[i]^2 + x[j]^2) / (2.0 * sigma_src^2))
    end
    return src
end
const src_plane = build_src()

# ── NAIVE (matches benchmark_julia_fractional.jl scalpel_run) ────
function build_constants_naive()
    a_nilt = 2.3 / t_end
    T_nilt = 2.0 * t_end
    kx = [2π * (i <= Nx ÷ 2 ? i - 1 : i - 1 - Nx) / (Nx * dx) for i in 1:Nx]
    ky = [2π * (j <= Ny ÷ 2 ? j - 1 : j - 1 - Ny) / (Ny * dx) for j in 1:Ny]
    KX = reshape(Float64.(kx), Nx, 1, 1)
    KY = reshape(Float64.(ky), 1, Ny, 1)
    omega = collect(0:N_NILT - 1) .* (π / T_nilt)
    s_arr = reshape(ComplexF64.(a_nilt .+ im .* omega), 1, 1, N_NILT)
    g2 = (s_arr .^ ALPHA) ./ D_diff .+ KX .^ 2 .+ KY .^ 2
    gz = sqrt.(g2)
    gz = gz .* sign.(real.(gz) .+ 1e-30)
    H = exp.(-gz .* d_obs)
    half_cpu = ones(Float64, 1, 1, N_NILT); half_cpu[1, 1, 1] = 0.5
    H_half = H .* half_cpu
    dt_n = 2 * T_nilt / N_NILT
    correction = reshape(exp.(a_nilt .* (collect(0:N_NILT - 1) .* dt_n)) ./ T_nilt, 1, 1, N_NILT)
    return H_half, correction
end

function scalpel_naive(src_arr, H_half, correction)
    Sh = fft(src_arr)
    G = reshape(Sh, Nx, Ny, 1) .* H_half
    z_raw = N_NILT .* ifft(G, 3)
    fkt = real.(z_raw) .* correction
    return real.(ifft(fkt, (1, 2)))
end

# ── TUNED: time on axis 1, plan cache, in-place broadcasts ───────
struct TunedWorkspace
    H_half::Array{ComplexF64,3}   # (N_NILT, Nx, Ny)
    correction::Array{Float64,3}  # (N_NILT, 1, 1)
    Sh::Array{ComplexF64,3}       # (1, Nx, Ny)
    F::Array{ComplexF64,3}        # (N_NILT, Nx, Ny) work
    tmp::Array{Float64,3}         # (N_NILT, Nx, Ny) real out
    plan_xy_fwd::AbstractFFTs.Plan
    plan_t::AbstractFFTs.Plan
    plan_xy::AbstractFFTs.Plan
end

function build_tuned()
    a_nilt = 2.3 / t_end
    T_nilt = 2.0 * t_end
    kx = [2π * (i <= Nx ÷ 2 ? i - 1 : i - 1 - Nx) / (Nx * dx) for i in 1:Nx]
    ky = [2π * (j <= Ny ÷ 2 ? j - 1 : j - 1 - Ny) / (Ny * dx) for j in 1:Ny]
    # Layout: time axis = 1 (fastest stride), kx = 2, ky = 3
    KX = reshape(Float64.(kx), 1, Nx, 1)
    KY = reshape(Float64.(ky), 1, 1, Ny)
    omega = collect(0:N_NILT - 1) .* (π / T_nilt)
    s_arr = reshape(ComplexF64.(a_nilt .+ im .* omega), N_NILT, 1, 1)
    g2 = @. (s_arr ^ ALPHA) / D_diff + KX ^ 2 + KY ^ 2
    gz = @. sqrt(g2)
    gz = @. gz * sign(real(gz) + 1e-30)
    H = @. exp(-gz * d_obs)
    H_half = copy(H)
    @views H_half[1, :, :] .*= 0.5
    dt_n = 2 * T_nilt / N_NILT
    correction = reshape(exp.(a_nilt .* (collect(0:N_NILT - 1) .* dt_n)) ./ T_nilt,
                         N_NILT, 1, 1)

    src_arr = src_plane
    Sh_plain = fft(src_arr)
    Sh = reshape(ComplexF64.(Sh_plain), 1, Nx, Ny)

    F = Array{ComplexF64}(undef, N_NILT, Nx, Ny)
    tmp = Array{Float64}(undef, N_NILT, Nx, Ny)

    plan_xy_fwd = plan_fft(zeros(ComplexF64, Nx, Ny); flags=FFTW.MEASURE)
    plan_t  = plan_ifft!(F, 1; flags=FFTW.MEASURE)
    plan_xy = plan_ifft!(F, (2, 3); flags=FFTW.MEASURE)

    return TunedWorkspace(H_half, correction, Sh, F, tmp, plan_xy_fwd, plan_t, plan_xy)
end

function scalpel_tuned!(W::TunedWorkspace)
    @. W.F = W.Sh * W.H_half
    W.plan_t * W.F
    @. W.F = W.F * N_NILT
    @. W.F = real(W.F) * W.correction
    W.plan_xy * W.F
    @. W.tmp = real(W.F)
    return W.tmp
end

# Build, warm up
H_half_n, correction_n = build_constants_naive()
W = build_tuned()
_ = scalpel_naive(src_plane, H_half_n, correction_n)
_ = scalpel_tuned!(W)

function median_time(f, n)
    times = Float64[]
    for _ in 1:n
        t0 = time_ns()
        _ = f()
        push!(times, (time_ns() - t0) / 1e6)
    end
    return median(sort(times))
end

t_naive = median_time(() -> scalpel_naive(src_plane, H_half_n, correction_n), 15)
t_tuned = median_time(() -> scalpel_tuned!(W), 30)

println("="^72)
@printf(" Fractional Caputo, %dx%dx%d, alpha=%.2f, N_NILT=%d (CPU, %d threads)\n",
        Nx, Ny, 32, ALPHA, N_NILT, Sys.CPU_THREADS)
println("="^72)
@printf(" Naive scalpel (current benchmark):  %8.1f ms\n", t_naive)
@printf(" Tuned scalpel (plans + threads):    %8.1f ms\n", t_tuned)
@printf(" Speedup from tuning:                %8.1fx\n", t_naive / t_tuned)
println("="^72)
@printf(" Reference CPU backends on same problem:\n")
@printf("   NumPy:      74 ms     PyTorch CPU: 36 ms     JAX CPU: 34 ms\n")
@printf(" Implied wall-ratio vs FTCS+L1 (=597 ms):\n")
@printf("   Naive:      %.1fx\n", 597 / t_naive)
@printf("   Tuned:      %.1fx\n", 597 / t_tuned)
println("="^72)
