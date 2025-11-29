"""
    residual_history(sol::InverseProblemSolution)

Extract residual history from solution.
"""
function residual_history(sol::InverseProblemSolution)
    if sol.history === nothing
        @warn "No history saved in solution"
        return nothing
    end
    return sol.history.residuals
end

"""
    solution_history(sol::InverseProblemSolution)

Extract solution history from solution.
"""
function solution_history(sol::InverseProblemSolution)
    if sol.history === nothing
        @warn "No history saved in solution"
        return nothing
    end
    return sol.history.us
end

"""
    convergence_rate(sol::InverseProblemSolution)

Estimate convergence rate from residual history.
"""
function convergence_rate(sol::InverseProblemSolution)
    residuals = residual_history(sol)
    if residuals === nothing || length(residuals) < 2
        return nothing
    end
    
    # Estimate rate using log(r_{k+1}/r_k)
    rates = Float64[]
    for i in 1:length(residuals)-1
        if residuals[i] > 0 && residuals[i+1] > 0
            push!(rates, log(residuals[i+1] / residuals[i]))
        end
    end
    
    return mean(rates)
end

"""
    RegularizationPath

Store solutions along a regularization path (varying α).
"""
struct RegularizationPath{T, S}
    alphas::Vector{T}
    solutions::Vector{S}
    residuals::Vector{T}
end

function RegularizationPath()
    RegularizationPath(Float64[], Any[], Float64[])
end

"""
    compute_reg_path(prob, alg_template, alphas)

Compute regularization path for different α values.

# Arguments
- `prob`: Inverse problem
- `alg_template`: Algorithm with parameters (α will be varied)
- `alphas`: Vector of regularization parameters to try

# Returns
- `RegularizationPath`: Solutions for each α
"""
function compute_reg_path(
    prob::AbstractInverseProblem,
    alg_template::AbstractIterativeRegularizationAlgorithm,
    alphas::AbstractVector
)
    path = RegularizationPath()
    
    for α in alphas
        # Create new algorithm with updated α
        # This is a simplified version - actual implementation would need
        # to handle different algorithm types appropriately
        alg = update_regularization_param(alg_template, α)
        
        sol = solve(prob, alg)
        
        push!(path.alphas, α)
        push!(path.solutions, sol.u)
        push!(path.residuals, sol.resid)
    end
    
    return path
end

# Helper function to update regularization parameter
function update_regularization_param(alg, α)
    # This would need specific implementations for each algorithm type
    # Placeholder for structure
    error("update_regularization_param not implemented for $(typeof(alg))")
end

"""
    lcurve_corner(path::RegularizationPath)

Find the corner of the L-curve using the regularization path.

Returns the index of the optimal regularization parameter.
"""
function lcurve_corner(path::RegularizationPath)
    if length(path.alphas) < 3
        error("Need at least 3 points for L-curve analysis")
    end
    
    # Compute solution norms
    sol_norms = [norm(s) for s in path.solutions]
    
    # Compute curvature (simplified)
    # In practice, would use more sophisticated corner detection
    curvatures = Float64[]
    for i in 2:length(path.alphas)-1
        # Approximate curvature using finite differences
        dx1 = log(path.residuals[i]) - log(path.residuals[i-1])
        dy1 = log(sol_norms[i]) - log(sol_norms[i-1])
        dx2 = log(path.residuals[i+1]) - log(path.residuals[i])
        dy2 = log(sol_norms[i+1]) - log(sol_norms[i])
        
        # Curvature approximation
        κ = abs((dx2*dy1 - dx1*dy2) / (dx1^2 + dy1^2)^1.5)
        push!(curvatures, κ)
    end
    
    # Return index of maximum curvature (+ 1 due to loop starting at 2)
    return argmax(curvatures) + 1
end

"""
    relative_error(x_computed, x_true)

Compute relative error between computed and true solution.
"""
function relative_error(x_computed, x_true)
    return norm(x_computed - x_true) / norm(x_true)
end