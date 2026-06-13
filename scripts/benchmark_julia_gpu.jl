"""
Definitive benchmark: Julia CUDA — both scalpel and FDTD on GPU.
No Python, no interpreter overhead. Native compiled GPU kernels.
This is the fairest possible comparison.
"""

using CUDA
using CUDA.CUFFT
using FFTW
using Statistics
using Printf

# ── Physics ────────────────────────────────────────────────────────
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
println(" Device: ", CUDA.name(CUDA.device()))
@printf(" Julia CUDA BENCHMARK: %dx%d, wet clay, d=%.1fm\n", Nx, Nx, depth)
@printf(" Scalpel: N_NILT=%d | FDTD: Nz=%d, Nt=%d\n", N_NILT, Nz, Nt_fdtd)
@printf(" NFE ratio: %.0fx\n", nfe_fdtd / nfe_scalpel)
println("="^75)


# ═══════════════════════════════════════════════════════════════════
#  Scalpel on GPU (CuArray + CUFFT)
# ═══════════════════════════════════════════════════════════════════
function scalpel_gpu()
    dx_grid = 0.01

    # Build wavenumber grids on GPU
    kx_cpu = [2π * (i <= Nx÷2 ? i-1 : i-1-Nx) / (Nx * dx_grid) for i in 1:Nx]
    ky_cpu = [2π * (j <= Nx÷2 ? j-1 : j-1-Nx) / (Nx * dx_grid) for j in 1:Nx]

    KX = CuArray(reshape(Float64.(kx_cpu), Nx, 1, 1))
    KY = CuArray(reshape(Float64.(ky_cpu), 1, Nx, 1))

    ω_cpu = collect(0:N_NILT-1) .* (π / T_nilt)
    s_cpu = a_nilt .+ im .* ω_cpu
    S = CuArray(reshape(ComplexF64.(s_cpu), 1, 1, N_NILT))

    # Source (build on CPU, transfer to GPU)
    src_cpu = zeros(ComplexF64, Nx, Nx)
    src_cpu[Nx÷2+1, Nx÷2+1] = 1.0
    src = CuArray(src_cpu)
    Ŝ = fft(src)

    # Dispersion + transfer function (all on GPU)
    γ² = MU_0 .* (σ_cond .* S .+ ε .* S.^2) .- KX.^2 .- KY.^2
    γz = sqrt.(γ²)
    γz = γz .* sign.(real.(γz) .+ Float64(1e-30))

    H = exp.(-γz .* depth)

    # Apply source spectrum
    F_spec = reshape(Ŝ, Nx, Nx, 1) .* H

    # DC half-weight (build on CPU, transfer)
    half_cpu = ones(Float64, 1, 1, N_NILT)
    half_cpu[1, 1, 1] = 0.5
    half = CuArray(half_cpu)
    G = F_spec .* half

    # NILT inverse
    z_raw = N_NILT .* ifft(G, 3)

    dt_n = 2 * T_nilt / N_NILT
    t_cpu = collect(0:N_NILT-1) .* dt_n
    correction = CuArray(reshape(exp.(a_nilt .* t_cpu) ./ T_nilt, 1, 1, N_NILT))
    field_kt = real.(z_raw) .* correction

    # Inverse 2D FFT
    field_xt = real.(ifft(field_kt, (1, 2)))
    return field_xt
end

# Warmup + compile
_ = scalpel_gpu()
CUDA.synchronize()

# Time
t_scalpel = let
    times = Float64[]
    for _ in 1:20
        CUDA.synchronize()
        t0 = time_ns()
        _ = scalpel_gpu()
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

@printf("\n Scalpel (CUDA):  %.1f ms\n", t_scalpel)


# ═══════════════════════════════════════════════════════════════════
#  FDTD batched on GPU (CuArray, Julia kernel)
# ═══════════════════════════════════════════════════════════════════
function fdtd_gpu_batched!()
    E = CUDA.zeros(Float64, n_modes, Nz)
    H = CUDA.zeros(Float64, n_modes, Nz - 1)
    E_prev = CUDA.zeros(Float64, n_modes)
    em2_buf = CUDA.zeros(Float64, n_modes)

    for n in 1:Nt_fdtd
        t_n = (n-1) * dt_fdtd

        # Store boundary
        em2_buf .= @view E[:, Nz-1]

        # Update H: H += c3*(E[:,2:end] - E[:,1:end-1])
        H .+= c3_v .* (@view(E[:, 2:Nz]) .- @view(E[:, 1:Nz-1]))

        # Update E interior
        @view(E[:, 2:Nz-1]) .= c1_v .* @view(E[:, 2:Nz-1]) .+
            c2_v .* (@view(H[:, 2:Nz-1]) .- @view(H[:, 1:Nz-2]))

        # Source BC
        src_val = exp(-0.5 * ((t_n - tc) / tw)^2)
        @view(E[:, 1]) .= src_val

        # Mur ABC right
        @view(E[:, Nz]) .= E_prev .+ mur_v .* (@view(E[:, Nz-1]) .- @view(E[:, Nz]))
        E_prev .= em2_buf
    end

    return E[:, obs_idx]
end

# Warmup
_ = fdtd_gpu_batched!()
CUDA.synchronize()

# Time
t_fdtd = let
    times = Float64[]
    for _ in 1:3
        CUDA.synchronize()
        t0 = time_ns()
        _ = fdtd_gpu_batched!()
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

@printf(" FDTD    (CUDA):  %.0f ms\n", t_fdtd)
@printf(" Wall ratio:      %.0fx\n", t_fdtd / t_scalpel)


# ═══════════════════════════════════════════════════════════════════
#  Also run CPU for comparison (same Julia session)
# ═══════════════════════════════════════════════════════════════════
function scalpel_cpu()
    dx_grid = 0.01
    kx = [2π * (i <= Nx÷2 ? i-1 : i-1-Nx) / (Nx * dx_grid) for i in 1:Nx]
    ky = [2π * (j <= Nx÷2 ? j-1 : j-1-Nx) / (Nx * dx_grid) for j in 1:Nx]
    KX = reshape(Float64.(kx), Nx, 1, 1)
    KY = reshape(Float64.(ky), 1, Nx, 1)
    ω = collect(0:N_NILT-1) .* (π / T_nilt)
    s = a_nilt .+ im .* ω
    S = reshape(ComplexF64.(s), 1, 1, N_NILT)

    src = zeros(ComplexF64, Nx, Nx)
    src[Nx÷2+1, Nx÷2+1] = 1.0
    Ŝ = fft(src)

    γ² = MU_0 .* (σ_cond .* S .+ ε .* S.^2) .- KX.^2 .- KY.^2
    γz = sqrt.(γ²)
    γz = γz .* sign.(real.(γz) .+ 1e-30)
    H = exp.(-γz .* depth)
    F_spec = reshape(Ŝ, Nx, Nx, 1) .* H
    F_spec[:, :, 1] .*= 0.5
    z_raw = N_NILT .* ifft(F_spec, 3)
    dt_n = 2 * T_nilt / N_NILT
    t_arr = collect(0:N_NILT-1) .* dt_n
    correction = reshape(exp.(a_nilt .* t_arr) ./ T_nilt, 1, 1, N_NILT)
    return real.(ifft(real.(z_raw) .* correction, (1, 2)))
end

function fdtd_cpu_seq()
    E = zeros(Nz)
    H = zeros(Nz - 1)
    ep = 0.0
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
        E[Nz] = ep + mur_v * (E[Nz-1] - E[Nz])
        ep = em2
    end
    return E[obs_idx]
end

# CPU scalpel
_ = scalpel_cpu()
t_scalpel_cpu = let
    times = Float64[]
    for _ in 1:10
        t0 = time_ns()
        _ = scalpel_cpu()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end

# CPU FDTD (sequential, one column at a time)
_ = fdtd_cpu_seq()
t_1col = let
    times = Float64[]
    for _ in 1:10
        t0 = time_ns()
        _ = fdtd_cpu_seq()
        push!(times, (time_ns() - t0) / 1e6)
    end
    median(sort(times))
end
t_fdtd_cpu = t_1col * n_modes


# ═══════════════════════════════════════════════════════════════════
println("\n", "="^75)
@printf(" JULIA COMPLETE SUMMARY: %dx%d grid, Nz=%d, Nt=%d\n", Nx, Nx, Nz, Nt_fdtd)
@printf(" NFE ratio: %.0fx\n", nfe_fdtd / nfe_scalpel)
println("="^75)
@printf(" %-14s  %10s  %10s  %10s\n", "Backend", "Scalpel", "FDTD", "Wall ratio")
@printf(" %-14s  %9.1fms  %9.0fms  %9.0fx\n",
        "Julia CPU", t_scalpel_cpu, t_fdtd_cpu, t_fdtd_cpu / t_scalpel_cpu)
@printf(" %-14s  %9.1fms  %9.0fms  %9.0fx\n",
        "Julia CUDA", t_scalpel, t_fdtd, t_fdtd / t_scalpel)
println("="^75)
