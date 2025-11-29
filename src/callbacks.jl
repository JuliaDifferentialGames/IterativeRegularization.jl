"""
    AbstractIterativeRegularizationCallback

Abstract type for callbacks during iterative regularization.
"""
abstract type AbstractIterativeRegularizationCallback end

"""
    DiscrepancyPrinciple(δ, τ=1.1)

Stopping criterion based on Morozov's discrepancy principle.
Stops when ||F(x_k) - y|| ≤ τδ, where δ is the noise level.

# Arguments
- `δ`: Estimated noise level
- `τ`: Safety factor (typically 1.1)
"""
struct DiscrepancyPrinciple{T} <: AbstractIterativeRegularizationCallback
    δ::T
    τ::T
end

DiscrepancyPrinciple(δ; τ=1.1) = DiscrepancyPrinciple(δ, τ)

function (dp::DiscrepancyPrinciple)(x, resid, iter)
    if resid <= dp.τ * dp.δ
        @info "Discrepancy principle satisfied at iteration $iter"
        return true  # Signal to stop
    end
    return false
end

"""
    ResidualCallback(frequency=1)

Callback that prints residual information.

# Arguments
- `frequency`: Print every `frequency` iterations
"""
struct ResidualCallback
    frequency::Int
end

ResidualCallback() = ResidualCallback(1)

function (rc::ResidualCallback)(x, resid, iter)
    if iter % rc.frequency == 0
        @info "Iteration $iter: ||F(x) - y|| = $resid"
    end
    return false
end

"""
    ComposedCallback(callbacks...)

Compose multiple callbacks together.
"""
struct ComposedCallback{T<:Tuple}
    callbacks::T
end

ComposedCallback(callbacks...) = ComposedCallback(callbacks)

function (cc::ComposedCallback)(x, resid, iter)
    for cb in cc.callbacks
        if cb(x, resid, iter)
            return true  # Stop if any callback signals
        end
    end
    return false
end

"""
    SolutionSaver(solutions::Vector)

Callback that saves solutions at each iteration.
"""
mutable struct SolutionSaver{T}
    solutions::Vector{T}
end

SolutionSaver() = SolutionSaver(Any[])

function (ss::SolutionSaver)(x, resid, iter)
    push!(ss.solutions, copy(x))
    return false
end