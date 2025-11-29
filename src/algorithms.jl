"""
    AbstractIterativeRegularizationAlgorithm

Abstract base type for all iterative regularization algorithms.
"""
abstract type AbstractIterativeRegularizationAlgorithm end

# Base algorithm struct with common parameters
Base.@kwdef struct AlgorithmParams
    maxiter::Int = 100
    tol::Float64 = 1e-6
    α::Float64 = 1.0  # Regularization parameter
    α_schedule = nothing  # Regularization parameter schedule
    verbose::Bool = false
    save_history::Bool = true
    callback = nothing
end
