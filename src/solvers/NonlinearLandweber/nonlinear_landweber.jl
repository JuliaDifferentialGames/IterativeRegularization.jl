"""
    NonlinearLandweber(; kwargs...)

Nonlinear Landweber iteration for solving F(x) = y.

Update rule: x_{k+1} = x_k - τ F'(x_k)^T (F(x_k) - y)

# Keyword Arguments
- `maxiter::Int=100`: Maximum number of iterations
- `tol::Float64=1e-6`: Convergence tolerance on residual
- `τ::Float64=1.0`: Step size (should satisfy 0 < τ < 2/||F'||^2)
- `τ_schedule=nothing`: Optional schedule for τ
- `verbose::Bool=false`: Print iteration information
- `save_history::Bool=true`: Save iteration history
"""
struct NonlinearLandweber{T} <: AbstractIterativeRegularizationAlgorithm
    params::AlgorithmParams
    τ::T
    τ_schedule
    
    function NonlinearLandweber(; 
        maxiter=100, tol=1e-6, τ=1.0, τ_schedule=nothing, 
        verbose=false, save_history=true, kwargs...
    )
        params = AlgorithmParams(; maxiter, tol, verbose, save_history, kwargs...)
        new{typeof(τ)}(params, τ, τ_schedule)
    end
end