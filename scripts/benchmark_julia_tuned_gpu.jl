"""
Tuned Julia CUDA scalpel — uses CUFFT plans + in-place broadcasts.
Verifies whether the GPU number is already near-optimal.
"""

using CUDA
using CUDA.CUFFT
using Statistics
using Printf

# ── Maxwell wet clay (panel a) ───────────────────────────────────
const MU_0 = 4e-7 * π
const EPS_0 = 8.854187817e-12
const σ_cond = 0.1
const ε_r = 10.0
const ε = EPS_0 * ε_r
const depth = 0.5
const a_nilt = 6.7167e+07
const T_nilt = 1.3713e-07
const N_NILT = 2048
const Nx = 64

# Layout: time on axis 1, transverse on (2,3)
function build_gpu()
    dx_grid = 0.01
    kx = [2π * (i <= Nx÷2 ? i-1 : i-1-Nx) / (Nx * dx_grid) for i in 1:Nx]
    ky = [2π * (j <= Nx÷2 ? j-1 : j-1-Nx) / (Nx * dx_grid) for j in 1:Nx]
    KX = CuArray(reshape(Float64.(kx), 1, Nx, 1))
    KY = CuArray(reshape(Float64.(ky), 1, 1, Nx))
    ω = collect(0:N_NILT-1) .* (π / T_nilt)
    S = CuArray(reshape(ComplexF64.(a_nilt .+ im .* ω), N_NILT, 1, 1))

    γ² = @. MU_0 * (σ_cond * S + ε * S^2) - KX^2 - KY^2
    γz = @. sqrt(γ²)
    γz = @. γz * sign(real(γz) + Float64(1e-30))
    H = @. exp(-γz * depth)
    @views H[1, :, :] .*= 0.5
    src_cpu = zeros(ComplexF64, Nx, Nx); src_cpu[Nx÷2+1, Nx÷2+1] = 1.0
    Ŝ = CuArray(reshape(fft(src_cpu), 1, Nx, Nx))

    dt_n = 2 * T_nilt / N_NILT
    correction = CuArray(reshape(exp.(a_nilt .* collect(0:N_NILT-1) .* dt_n) ./ T_nilt,
                                  N_NILT, 1, 1))
    F = CuArray{ComplexF64}(undef, N_NILT, Nx, Nx)
    plan_t  = plan_ifft!(F, 1)
    plan_xy = plan_ifft!(F, (2, 3))
    return (; Ŝ, H, correction, F, plan_t, plan_xy)
end

function scalpel_gpu_tuned!(W)
    @. W.F = W.Ŝ * W.H
    W.plan_t * W.F
    @. W.F = W.F * N_NILT
    @. W.F = real(W.F) * W.correction
    W.plan_xy * W.F
    return W.F
end

W = build_gpu()
# Warmup
_ = scalpel_gpu_tuned!(W); CUDA.synchronize()

function median_gpu(f, n)
    times = Float64[]
    for _ in 1:n
        CUDA.synchronize()
        t0 = time_ns()
        _ = f()
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end
    return median(sort(times))
end

t_tuned = median_gpu(() -> scalpel_gpu_tuned!(W), 30)

println("="^72)
@printf(" Julia CUDA Maxwell (tuned, plans+axis-1): %.2f ms\n", t_tuned)
@printf(" Reported in current CSV (naive):          18.2 ms\n")
@printf(" Implied speedup from tuning:              %.2fx\n", 18.2 / t_tuned)
println("="^72)
