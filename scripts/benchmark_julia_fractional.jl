"""
Julia (CPU + CUDA) fractional Caputo subdiffusion benchmark for panel (b).

Mirrors the Python `benchmark_fractional_heat_3d_all_backends.py` problem
exactly:
  PDE:  d_t^alpha u = D nabla^2 u  with alpha = 0.7
  Domain: 32^3, periodic xy, Dirichlet z, Heaviside surface source.
  t_end = 20 ms, d_obs = 10 cm, D = 1e-2.

Methods timed per backend:
  1. Spectral scalpel (FFT_xy + NILT_z with s^alpha dispersion)
  2. FTCS + L1 fractional time scheme

The L1 history convolution is implemented as a matrix-vector product on
the (Nx*Ny*Nz, n-1) diffs slice, which dispatches to CUBLAS gemv on GPU
and OpenBLAS on CPU. This avoids per-step Python-style scalar reads on
CuArrays (which CUDA.jl forbids by default for performance).
"""

using CUDA
using CUDA.CUFFT
using FFTW
using LinearAlgebra
using SpecialFunctions: gamma
using Statistics
using Printf

# ── Problem parameters (must match the Python script) ─────────────────
const ALPHA = 0.7
const D_diff = 1e-2
const L_dom = 0.32
const Nx = 32
const Ny = 32
const Nz = 32
const dx = L_dom / Nx
const sigma_src = 0.03
const d_obs = 0.10
const t_end = 0.020
const N_NILT = 2048

const cou = 0.5
const gamma_2_minus_alpha = gamma(2.0 - ALPHA)
const dt_max = (dx^2 / (6.0 * D_diff * gamma_2_minus_alpha))^(1.0 / ALPHA)
const dt_l1_init = cou * dt_max
const N_t = Int(ceil(t_end / dt_l1_init))
const dt_l1 = t_end / N_t
const coef = gamma_2_minus_alpha * dt_l1^ALPHA * D_diff
const M_cell = Nx * Ny * Nz

# L1 weights b_k = (k+1)^(1-alpha) - k^(1-alpha), 1-indexed: b_arr_cpu[k+1] = b_k
const b_arr_cpu = [(Float64(k + 1))^(1.0 - ALPHA) - (Float64(k))^(1.0 - ALPHA)
                   for k in 0:N_t]

# Source plane
function build_src()
    x = [(i - Nx / 2 - 0.5) * dx for i in 1:Nx]
    src = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        src[i, j] = exp(-(x[i]^2 + x[j]^2) / (2.0 * sigma_src^2))
    end
    return src
end

const src_plane = build_src()

println("=" ^ 75)
@printf(" Julia fractional benchmark, alpha = %.2f\n", ALPHA)
@printf(" N = %d^3, dx = %.2f mm, t_end = %.0f ms\n", Nx, dx * 1000, t_end * 1e3)
@printf(" L1: dt = %.1f us, N_t = %d, history = %.1f MB\n",
        dt_l1 * 1e6, N_t, N_t * M_cell * 8 / 2^20)
@printf(" Scalpel: N_NILT = %d\n", N_NILT)
println("=" ^ 75)


# ──────────────────────────────────────────────────────────────────────
# Spectral scalpel: 2D FFT_xy + NILT_z with s^alpha dispersion.
# Constants precomputed outside the timed pipeline (matches the Python).
# ──────────────────────────────────────────────────────────────────────
function build_scalpel_constants(arrtype)
    a_nilt = 2.3 / t_end
    T_nilt = 2.0 * t_end

    kx = [2π * (i <= Nx ÷ 2 ? i - 1 : i - 1 - Nx) / (Nx * dx) for i in 1:Nx]
    ky = [2π * (j <= Ny ÷ 2 ? j - 1 : j - 1 - Ny) / (Ny * dx) for j in 1:Ny]
    KX = arrtype(reshape(Float64.(kx), Nx, 1, 1))
    KY = arrtype(reshape(Float64.(ky), 1, Ny, 1))

    omega = collect(0:N_NILT - 1) .* (π / T_nilt)
    s_arr = arrtype(reshape(ComplexF64.(a_nilt .+ im .* omega), 1, 1, N_NILT))

    s_alpha = s_arr .^ ALPHA
    g2 = s_alpha ./ D_diff .+ KX .^ 2 .+ KY .^ 2
    gz = sqrt.(g2)
    gz = gz .* sign.(real.(gz) .+ 1e-30)
    H = exp.(-gz .* d_obs)

    half_cpu = ones(Float64, 1, 1, N_NILT)
    half_cpu[1, 1, 1] = 0.5
    half = arrtype(half_cpu)
    H_half = H .* half

    dt_n = 2 * T_nilt / N_NILT
    correction = arrtype(reshape(exp.(a_nilt .* (collect(0:N_NILT - 1) .* dt_n))
                                  ./ T_nilt, 1, 1, N_NILT))

    return H_half, correction
end

function scalpel_run(src_arr, H_half, correction)
    Sh = fft(src_arr)
    G = reshape(Sh, Nx, Ny, 1) .* H_half
    z_raw = N_NILT .* ifft(G, 3)
    fkt = real.(z_raw) .* correction
    return real.(ifft(fkt, (1, 2)))
end


# ──────────────────────────────────────────────────────────────────────
# FTCS + L1: periodic xy via circshift, Dirichlet z via split stencil
# (interior + two faces). diffs is laid out (Nx, Ny, Nz, N_t) so the
# weighted history sum at step n is a single (M, n-1) * (n-1,) gemv.
# ──────────────────────────────────────────────────────────────────────
function laplacian!(out, u)
    @views out .= (circshift(u, (1, 0, 0)) .+ circshift(u, (-1, 0, 0))
                   .+ circshift(u, (0, 1, 0)) .+ circshift(u, (0, -1, 0))
                   .- 4.0 .* u) ./ dx^2
    @views begin
        out[:, :, 2:Nz - 1] .+= (u[:, :, 1:Nz - 2] .+ u[:, :, 3:Nz]
                                  .- 2.0 .* u[:, :, 2:Nz - 1]) ./ dx^2
        out[:, :, 1] .+= (u[:, :, 2] .- 2.0 .* u[:, :, 1]) ./ dx^2
        out[:, :, Nz] .+= (u[:, :, Nz - 1] .- 2.0 .* u[:, :, Nz]) ./ dx^2
    end
    return out
end

function ftcs_l1_run!(diffs, u_prev, u_new, lap_buf, wbuf, src_arr,
                      b_arr_cpu, arrtype)
    fill!(u_prev, 0.0)

    @inbounds for n in 1:N_t
        laplacian!(lap_buf, u_prev)

        if n > 1
            # weights w_k = b_{n-k} for k = 1..n-1, paired with diffs[:,:,:,k]
            w_n_cpu = b_arr_cpu[n:-1:2]              # length n-1, on CPU
            w_n = arrtype(w_n_cpu)                    # tiny H2D copy if GPU
            diffs_mat = reshape(view(diffs, :, :, :, 1:n - 1), M_cell, n - 1)
            wbuf_flat = diffs_mat * w_n               # (M_cell,) via gemv
            wbuf .= reshape(wbuf_flat, Nx, Ny, Nz)
        else
            fill!(wbuf, 0.0)
        end

        u_new .= u_prev .- wbuf .+ coef .* lap_buf
        view(u_new, :, :, 1) .= src_arr               # Heaviside surface BC

        view(diffs, :, :, :, n) .= u_new .- u_prev
        u_prev .= u_new
    end
end


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
function bench_backend(name::String, arrtype, sync_fn::Function)
    println("\n--- $name ---")
    src_arr = arrtype(src_plane)
    H_half, correction = build_scalpel_constants(arrtype)

    # Scalpel: warmup + 5 timed
    _ = scalpel_run(src_arr, H_half, correction)
    sync_fn()
    times = Float64[]
    for _ in 1:5
        sync_fn()
        t0 = time_ns()
        _ = scalpel_run(src_arr, H_half, correction)
        sync_fn()
        push!(times, (time_ns() - t0) / 1e6)
    end
    sc_ms = median(sort(times))
    @printf("  scalpel: %.1f ms\n", sc_ms)

    # FTCS+L1: preallocate then run once
    diffs = arrtype(zeros(Float64, Nx, Ny, Nz, N_t))
    u_prev = arrtype(zeros(Float64, Nx, Ny, Nz))
    u_new = arrtype(zeros(Float64, Nx, Ny, Nz))
    lap_buf = arrtype(zeros(Float64, Nx, Ny, Nz))
    wbuf = arrtype(zeros(Float64, Nx, Ny, Nz))

    sync_fn()
    t0 = time_ns()
    ftcs_l1_run!(diffs, u_prev, u_new, lap_buf, wbuf, src_arr,
                 b_arr_cpu, arrtype)
    sync_fn()
    fl_ms = (time_ns() - t0) / 1e6
    @printf("  FTCS+L1: %.0f ms\n", fl_ms)
    @printf("  speedup: %.1fx\n", fl_ms / sc_ms)

    return (name, sc_ms, fl_ms)
end


results = []
push!(results, bench_backend("Julia CPU", Array, () -> nothing))
if CUDA.functional()
    push!(results, bench_backend("Julia CUDA", CuArray, CUDA.synchronize))
else
    println("\nCUDA not functional; skipping Julia GPU.")
end

println("\n", "=" ^ 75)
@printf(" %-14s  %10s  %10s  %10s\n", "Backend", "Scalpel", "FTCS+L1", "Speedup")
println("-" ^ 75)
for (name, sc, fl) in results
    @printf(" %-14s  %9.1fms  %9.0fms  %9.1fx\n", name, sc, fl, fl / sc)
end
println("=" ^ 75)
