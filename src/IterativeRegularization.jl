module IterativeRegularization

using RecursiveArrayTools, SciMLBase
using LinearAlgebra
using Statistics

# Export main types and functions
export InverseProblem, InverseProblemSolution
export solve

# Export algorithm types
export AbstractIterativeRegularizationAlgorithm
export NonlinearLandweber, DerivativeFreeLandweber, LandweberKaczmarz
export LevenbergMarquardt, IRGN, BroydenMethod
export NonlinearMultigrid, NonlinearFullMultigrid
export LevelSetMethod
export AdversarialRegularization, NETT, LISTA, LearnedProximal

# Export callback and analysis utilities
export IterativeRegularizationCallback, DiscrepancyPrinciple
export ResidualHistory, RegularizationPath

include("problems.jl")
include("solutions.jl")
include("algorithms.jl")
include("solve.jl")
include("callbacks.jl")
include("analysis.jl")

end # module