"""
Fair Julia benchmark: Scalpel vs FDTD on CPU.
Julia compiles the FDTD loop to native code — no interpreter overhead.
This is the FDTD best case.
"""

using FFTW
using LinearAlgebra
using Printf
using Statistics

# ── Physics ────────────────────────────────────────────────────────
const MU_0 = 4e-7 * π
const EPS_0 = 8.854187817e-12
const σ_cond = 0.1       # wet clay
const ε_r = 10.0
const ε = EPS_0 * ε_r
const depth = 0.5
const c_mat = 1.0 / sqrt(MU_0 * ε)
const t_transit = depth / c_mat
const t_end = 13 * t_transit

# NILT params
const a_nilt = 6.7167e+07
const T_nilt = 1.3713e-07
const N_NILT = 2048

# FDTD params
const Lz = depth * 3
const Nz = 500
const dz = Lz / Nz
const dt_fdtd = 0.5 * dz / c_mat
const Nt_fdtd = Int(floor(t_end / dt_fdtd)) + 1

const Nx = 64
const n_modes = Nx * Nx

const c1_v = (1 - σ_cond*dt_fdtd/(2ε)) / (1 + σ_cond*dt_fdtd/(2ε))
const c2_v = (dt_fdtd/(ε*dz)) / (1 + σ_cond*dt_fdtd/(2ε))
const c3_v = dt_fdtd / (MU_0 * dz)
const mur_v = (c_mat*dt_fdtd - dz) / (c_mat*dt_fdtd + dz)
const obs_idx = Int(floor(depth / dz))
const tc = 3 * t_transit
const tw = t_transit * 0.15

nfe_scalpel = n_modes * N_NILT
nfe_fdtd = n_modes * 2 * Nz * Nt_fdtd

println("="^75)
@printf(" Julia BENCHMARK: %dx%d, wet clay, d=%.1fm\n", Nx, Nx, depth)
@printf(" Scalpel: N_NILT=%d | FDTD: Nz=%d, Nt=%d\n", N_NILT, Nz, Nt_fdtd)
@printf(" NFE: scalpel=%.2e, fdtd=%.2e, ratio=%.0fx\n",
        nfe_scalpel, nfe_fdtd, nfe_fdtd/nfe_scalpel)
println("="^75)

# ═══════════════════════════════════════════════════════════════════
#  Scalpel (Julia CPU)
# ═══════════════════════════════════════════════════════════════════
function scalpel_julia()
    dx_grid = 0.01
    kx = fftfreq(Nx, 1.0/dx_grid) .* (2π)
    ky = fftfreq(Nx, 1.0/dx_grid) .* (2π)

    KX = reshape(kx, Nx, 1, 1)
    KY = reshape(ky, 1, Nx, 1)

    ω = collect(0:N_NILT-1) .* (π / T_nilt)
    s = a_nilt .+ im .* ω
    S = reshape(s, 1, 1, N_NILT)

    # Source
    src = zeros(ComplexF64, Nx, Nx)
    src[Nx÷2+1, Nx÷2+1] = 1.0
    Ŝ = fft(src)

    # Dispersion
    γ² = MU_0 .* (σ_cond .* S .+ ε .* S.^2) .- KX.^2 .- KY.^2
    γz = sqrt.(γ²)
    # Branch cut: Re(γz) ≥ 0
    γz = γz .* sign.(real.(γz) .+ 1e-30)

    # Transfer function
    H = exp.(-γz .* depth)

    # Apply source and DC half-weight
    F_spec = Ŝ[:, :, :] .* H
    F_spec[:, :, 1] .*= 0.5

    # NILT inverse
    z_raw = N_NILT .* ifft(F_spec, 3)
    dt_n = 2 * T_nilt / N_NILT
    t_arr = collect(0:N_NILT-1) .* dt_n
    correction = exp.(a_nilt .* t_arr) ./ T_nilt
    field_kt = real.(z_raw) .* reshape(correction, 1, 1, N_NILT)

    # Inverse 2D FFT
    field_xt = real.(ifft(field_kt, (1, 2)))
    return field_xt
end

# Warmup
_ = scalpel_julia()

# Time
t_scalpel = let
    times = Float64[]
    for _ in 1:20
        t0 = time_ns()
        _ = scalpel_julia()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

@printf("\n Scalpel (Julia CPU): %.1f ms\n", t_scalpel)


# ═══════════════════════════════════════════════════════════════════
#  FDTD batched (Julia CPU) — compiled loop, no interpreter overhead
# ═══════════════════════════════════════════════════════════════════
function fdtd_batched_julia!(E, H, E_prev)
    @inbounds for n in 1:Nt_fdtd
        t_n = (n-1) * dt_fdtd

        # Store boundary
        @simd for col in 1:n_modes
            E_prev[col] = E[col, Nz-1]  # E[:, -2] equivalent
        end

        # Update H
        @inbounds for j in 1:Nz-1
            @simd for col in 1:n_modes
                H[col, j] += c3_v * (E[col, j+1] - E[col, j])
            end
        end

        # Update E interior
        @inbounds for j in 2:Nz-1
            @simd for col in 1:n_modes
                E[col, j] = c1_v * E[col, j] + c2_v * (H[col, j] - H[col, j-1])
            end
        end

        # Source BC
        src_val = exp(-0.5 * ((t_n - tc) / tw)^2)
        @simd for col in 1:n_modes
            E[col, 1] = src_val
        end

        # Mur ABC right
        @simd for col in 1:n_modes
            E[col, Nz] = E_prev[col] + mur_v * (E[col, Nz-1] - E[col, Nz])
        end
    end
    return nothing
end

E = zeros(n_modes, Nz)
H = zeros(n_modes, Nz-1)
Ep = zeros(n_modes)

# Warmup
fill!(E, 0.0); fill!(H, 0.0); fill!(Ep, 0.0)
fdtd_batched_julia!(E, H, Ep)

# Time
t_fdtd = let
    times = Float64[]
    for _ in 1:3
        fill!(E, 0.0); fill!(H, 0.0); fill!(Ep, 0.0)
        t0 = time_ns()
        fdtd_batched_julia!(E, H, Ep)
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

@printf(" FDTD    (Julia CPU): %.0f ms\n", t_fdtd)
@printf(" Wall ratio: %.0fx\n", t_fdtd / t_scalpel)
@printf(" NFE ratio:  %.0fx\n", nfe_fdtd / nfe_scalpel)


# ═══════════════════════════════════════════════════════════════════
#  Also time a single FDTD column (to show per-column cost)
# ═══════════════════════════════════════════════════════════════════
function fdtd_1col_julia()
    E = zeros(Nz)
    H = zeros(Nz-1)
    E_prev_right = 0.0

    @inbounds for n in 1:Nt_fdtd
        t_n = (n-1) * dt_fdtd
        em2 = E[Nz-1]
        @simd for j in 1:Nz-1
            H[j] += c3_v * (E[j+1] - E[j])
        end
        @simd for j in 2:Nz-1
            E[j] = c1_v * E[j] + c2_v * (H[j] - H[j-1])
        end
        E[1] = exp(-0.5 * ((t_n - tc) / tw)^2)
        E[Nz] = E_prev_right + mur_v * (E[Nz-1] - E[Nz])
        E_prev_right = em2
    end
    return E[obs_idx]
end

_ = fdtd_1col_julia()
t_1col = let
    times = Float64[]
    for _ in 1:10
        t0 = time_ns()
        _ = fdtd_1col_julia()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

@printf("\n Single FDTD column: %.2f ms (%d steps x %d cells)\n",
        t_1col, Nt_fdtd, Nz)
@printf(" %d columns sequential: %.0f ms\n", n_modes, t_1col * n_modes)

println("\n", "="^75)
@printf(" JULIA SUMMARY: %dx%d grid\n", Nx, Nx)
println("="^75)
@printf(" Scalpel:    %8.1f ms\n", t_scalpel)
@printf(" FDTD batch: %8.0f ms\n", t_fdtd)
@printf(" FDTD seq:   %8.0f ms  (%d x %.2fms)\n",
        t_1col * n_modes, n_modes, t_1col)
@printf(" Wall ratio (batch): %.0fx\n", t_fdtd / t_scalpel)
@printf(" Wall ratio (seq):   %.0fx\n", t_1col * n_modes / t_scalpel)
@printf(" NFE ratio:          %.0fx\n", nfe_fdtd / nfe_scalpel)
println("="^75)
