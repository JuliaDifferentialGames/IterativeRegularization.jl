# examples/basic_usage.jl
# Example usage of IterativeRegularization.jl

using IterativeRegularization
using LinearAlgebra
using Random

# ==============================================================================
# Example 1: Simple Linear Inverse Problem
# ==============================================================================

println("Example 1: Linear Inverse Problem with Nonlinear Landweber")
println("="^70)

# Create a simple linear forward operator
n = 100
m = 80
Random.seed!(1234)
A = randn(m, n)
x_true = randn(n)
x_true[50:end] .= 0  # Sparse solution

# Generate noisy data
noise_level = 0.01
y = A * x_true + noise_level * randn(m)

# Define forward operator
F(x) = A * x

# Create inverse problem
x0 = zeros(n)
prob = InverseProblem(y, F, x0=x0)

# Solve with Nonlinear Landweber
alg = NonlinearLandweber(maxiter=200, tol=1e-6, τ=0.01, verbose=true)
sol = solve(prob, alg)

println("Solution found: ", was_successful(sol))
println("Final residual: ", get_residual(sol))
println("Iterations: ", get_iterations(sol))
println("Relative error: ", relative_error(sol.u, x_true))
println()

# ==============================================================================
# Example 2: Using Callbacks and Discrepancy Principle
# ==============================================================================

println("Example 2: Using Discrepancy Principle")
println("="^70)

# Create callback with discrepancy principle
dp_callback = DiscrepancyPrinciple(noise_level * sqrt(m), τ=1.1)

# Solve with callback
alg2 = NonlinearLandweber(maxiter=500, τ=0.01, callback=dp_callback)
sol2 = solve(prob, alg2, verbose=false)

println("Stopped at iteration: ", sol2.iterations)
println("Final residual: ", sol2.resid)
println("Relative error: ", relative_error(sol2.u, x_true))
println()

# ==============================================================================
# Example 3: IRGN with Decreasing Regularization
# ==============================================================================

println("Example 3: Iteratively Regularized Gauss-Newton")
println("="^70)

# IRGN with decreasing regularization parameter
alg3 = IRGN(
    maxiter=50,
    tol=1e-6,
    α0=1.0,
    α_reduction=0.7,
    min_α=1e-8,
    verbose=true
)

# Note: This will error since IRGN is not fully implemented
# sol3 = solve(prob, alg3)
println("IRGN algorithm defined but not yet implemented")
println()

# ==============================================================================
# Example 4: Levenberg-Marquardt with Adaptive Damping
# ==============================================================================

println("Example 4: Levenberg-Marquardt Algorithm")
println("="^70)

alg4 = LevenbergMarquardt(
    maxiter=100,
    tol=1e-6,
    λ0=1e-2,
    λ_increase=10.0,
    λ_decrease=0.1,
    verbose=true
)

# Note: This will error since LM is not fully implemented
# sol4 = solve(prob, alg4)
println("Levenberg-Marquardt algorithm defined but not yet implemented")
println()

# ==============================================================================
# Example 5: Nonlinear Problem
# ==============================================================================

println("Example 5: Nonlinear Inverse Problem")
println("="^70)

# Define a nonlinear forward operator
function nonlinear_F(x)
    return A * (x.^2)  # Nonlinear forward map
end

# Generate data from nonlinear model
x_true_nl = abs.(randn(n)) .+ 0.5
y_nl = nonlinear_F(x_true_nl) + 0.01 * randn(m)

# Create nonlinear inverse problem
prob_nl = InverseProblem(y_nl, nonlinear_F, x0=ones(n))

# Solve (would need Jacobian or AD integration)
# alg5 = NonlinearLandweber(maxiter=100, τ=0.001, verbose=true)
# sol5 = solve(prob_nl, alg5)
println("Nonlinear problem defined - requires AD integration for gradient")
println()

# ==============================================================================
# Example 6: Regularization Path and L-curve
# ==============================================================================

println("Example 6: Computing Regularization Path")
println("="^70)

# Define range of regularization parameters
alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

println("Would compute solutions for α = ", alphas)
println("Then use lcurve_corner() to find optimal regularization parameter")
println()

# ==============================================================================
# Example 7: Saving Solution History
# ==============================================================================

println("Example 7: Saving and Analyzing Solution History")
println("="^70)

# Create solution saver callback
sol_saver = SolutionSaver()

# Solve with history saving
alg6 = NonlinearLandweber(maxiter=50, τ=0.01, save_history=true)
sol6 = solve(prob, alg6, save_everystep=true, callback=sol_saver)

if sol6.history !== nothing
    println("Residual history length: ", length(sol6.history.residuals))
    println("First 5 residuals: ", sol6.history.residuals[1:min(5, end)])
    
    # Estimate convergence rate
    rate = convergence_rate(sol6)
    if rate !== nothing
        println("Estimated convergence rate: ", rate)
    end
end
println()

# ==============================================================================
# Example 8: Using LinearInverseProblem
# ==============================================================================

println("Example 8: LinearInverseProblem Specialization")
println("="^70)

# Use specialized linear problem type
prob_linear = LinearInverseProblem(y, A, x0=x0)

# The forward operator is automatically constructed from the matrix
F_auto = get_forward_operator(prob_linear)
println("Forward operator constructed from matrix: ", F_auto(x0) == A * x0)
println()

# ==============================================================================
# Example 9: Problem with Prior Information
# ==============================================================================

println("Example 9: Inverse Problem with Prior")
println("="^70)

# Define a prior (e.g., sparsity-promoting)
prior(x) = sum(abs, x)  # L1 norm for sparsity

# Create problem with prior
prob_prior = InverseProblem(
    y, F, 
    x0=x0,
    prior=prior,
    noise_model=:gaussian
)

println("Problem with prior defined")
println("Has prior: ", has_prior(prob_prior))
println()

println("="^70)
println("Examples complete!")
println("Note: Many algorithms show structure but need full implementation")
println("Nonlinear Landweber is the reference implementation to follow")