"""
    AbstractInverseProblem

Abstract base type for all inverse problems.
"""
abstract type AbstractInverseProblem end

"""
    InverseProblem{uType, yType, fType, pType, kType}

Defines an inverse problem of the form:
    y = F(x) + noise

where we seek to recover x from noisy observations y.

# Fields
- `y::yType`: Observed data
- `F::fType`: Forward operator F: X → Y
- `x0::uType`: Initial guess for the solution
- `p::pType`: Additional parameters for the forward operator
- `kwargs::kType`: Additional keyword arguments (priors, noise model, etc.)

# Keyword Arguments
- `prior`: Prior distribution or regularization term
- `noise_model`: Noise characteristics (e.g., Gaussian, Poisson)
- `operator_norm`: Known or estimated norm of the operator
- `jacobian`: Optional Jacobian of F (if available analytically)
- `constraints`: Constraints on the solution space

# Example
```julia
F(x) = A * x  # Forward operator
y = F(x_true) + noise  # Noisy observations

prob = InverseProblem(y, F, x0=zeros(n))
sol = solve(prob, IRGN())
```
"""
struct InverseProblem{uType, yType, fType, pType, kType} <: AbstractInverseProblem
    y::yType
    F::fType
    x0::uType
    p::pType
    kwargs::kType
    
    function InverseProblem{uType, yType, fType, pType, kType}(
        y, F, x0, p, kwargs
    ) where {uType, yType, fType, pType, kType}
        new{uType, yType, fType, pType, kType}(y, F, x0, p, kwargs)
    end
end

function InverseProblem(y, F; x0=nothing, p=nothing, kwargs...)
    if x0 === nothing
        error("Initial guess x0 must be provided")
    end
    
    uType = typeof(x0)
    yType = typeof(y)
    fType = typeof(F)
    pType = typeof(p)
    kType = typeof(kwargs)
    
    InverseProblem{uType, yType, fType, pType, kType}(y, F, x0, p, kwargs)
end

# Convenience accessors
get_forward_operator(prob::InverseProblem) = prob.F
get_data(prob::InverseProblem) = prob.y
get_initial_guess(prob::InverseProblem) = prob.x0
get_parameters(prob::InverseProblem) = prob.p

# Check for specific kwargs
has_prior(prob::InverseProblem) = haskey(prob.kwargs, :prior)
has_jacobian(prob::InverseProblem) = haskey(prob.kwargs, :jacobian)
has_constraints(prob::InverseProblem) = haskey(prob.kwargs, :constraints)

"""
    LinearInverseProblem

Specialization for linear inverse problems where F(x) = A*x.
"""
struct LinearInverseProblem{uType, yType, AType, pType, kType} <: AbstractInverseProblem
    y::yType
    A::AType
    x0::uType
    p::pType
    kwargs::kType
end

function LinearInverseProblem(y, A; x0=nothing, p=nothing, kwargs...)
    if x0 === nothing
        x0 = zeros(eltype(A), size(A, 2))
    end
    LinearInverseProblem{typeof(x0), typeof(y), typeof(A), typeof(p), typeof(kwargs)}(
        y, A, x0, p, kwargs
    )
end

get_forward_operator(prob::LinearInverseProblem) = x -> prob.A * x
get_matrix(prob::LinearInverseProblem) = prob.A