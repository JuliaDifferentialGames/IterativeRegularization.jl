"""
    AbstractInverseProblemSolution

Abstract base type for solutions to inverse problems.
"""
abstract type AbstractInverseProblemSolution end

"""
    InverseProblemSolution

Solution object for inverse problems, following SciML conventions.

# Fields
- `u`: Final solution
- `resid`: Final residual norm
- `prob`: Original problem
- `alg`: Algorithm used
- `retcode`: Return code indicating solution status
- `iterations`: Number of iterations performed
- `history`: History of residuals, solutions, and other metrics
- `destats`: Detailed statistics about the solve
"""
struct InverseProblemSolution{T, N, uType, P, A, H, S} <: AbstractInverseProblemSolution
    u::uType
    resid::T
    prob::P
    alg::A
    retcode::Symbol
    iterations::Int
    history::H
    destats::S
end

function InverseProblemSolution(
    u, resid, prob, alg;
    retcode=:Success,
    iterations=0,
    history=nothing,
    destats=nothing
)
    T = typeof(resid)
    N = ndims(u)
    InverseProblemSolution{T, N, typeof(u), typeof(prob), typeof(alg), 
                          typeof(history), typeof(destats)}(
        u, resid, prob, alg, retcode, iterations, history, destats
    )
end

# Common solution interface
get_solution(sol::InverseProblemSolution) = sol.u
get_residual(sol::InverseProblemSolution) = sol.resid
get_iterations(sol::InverseProblemSolution) = sol.iterations
was_successful(sol::InverseProblemSolution) = sol.retcode == :Success

"""
    IterativeHistory

Stores the history of an iterative regularization solve.
"""
struct IterativeHistory{uType, rType}
    us::Vector{uType}          # Solution iterates
    residuals::Vector{rType}   # Residual norms
    reg_params::Vector{Float64} # Regularization parameters
    times::Vector{Float64}      # Computation times
end

function IterativeHistory(::Type{uT}, ::Type{rT}) where {uT, rT}
    IterativeHistory{uT, rT}(uT[], rT[], Float64[], Float64[])
end

"""
    DEStats

Detailed statistics following SciMLBase conventions.
"""
struct DEStats
    nf::Int              # Number of forward operator evaluations
    njacs::Int           # Number of Jacobian evaluations
    naccept::Int         # Number of accepted steps
    nreject::Int         # Number of rejected steps
    total_time::Float64  # Total computation time
end

DEStats() = DEStats(0, 0, 0, 0, 0.0)