# IterativeRegularization.jl

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Build Status](https://github.com/yourusername/IterativeRegularization.jl/workflows/CI/badge.svg)](https://github.com/yourusername/IterativeRegularization.jl/actions)
[![Coverage](https://codecov.io/gh/yourusername/IterativeRegularization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/IterativeRegularization.jl)

A Julia package for solving nonlinear inverse problems using iterative regularization methods, following the [SciML](https://sciml.ai/) interface standards.

## Purpose

IterativeRegularization.jl provides fast, flexible implementations of standard and cutting-edge iterative regularization methods for nonlinear ill-posed inverse problems. The package follows the familiar interface of DifferentialEquations.jl, making it easy to use for anyone familiar with the SciML ecosystem.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/JuliaDifferentialGames/IterativeRegularization.jl.git")
```

## Development Status

**Currently Implemented:**
- ✅ TODO

**In Progress:**
- 🚧 Package structure following SciML standards
- 🚧 Problem types and solution interface
- 🚧 Algorithm type hierarchy
- 🚧 Callback system
- 🚧 Reference implementation (Nonlinear Landweber)
- 🚧 Analysis tools (residual history, convergence rate, L-curve)
- 🚧 Full algorithm implementations
- 🚧 Automatic differentiation integration
- 🚧 Neural network regularizers
- 🚧 GPU support
- 🚧 Comprehensive documentation

**Planned:**
- 📋 PINNs integration
- 📋 Tensor decomposition methods
- 📋 Uncertainty quantification
- 📋 Parallel solvers

## Forthcoming Methods

### Classical Iterative Methods
- **Nonlinear Landweber** - Gradient descent-based method with step size control
- **Derivative-free Landweber** - Uses finite differences for derivative-free optimization
- **Landweber-Kaczmarz** - Component-wise iterative updates
- **Levenberg-Marquardt** - Adaptive damping for nonlinear least squares
- **Iteratively Regularized Gauss-Newton (IRGN)** - Decreasing regularization with inner linear solves
- **Broyden's Method** - Quasi-Newton approach with Jacobian approximation

### Advanced Methods
- **Nonlinear Multigrid** - Multigrid acceleration for inverse problems
- **Nonlinear Full Multigrid** - Full multigrid with nested iterations
- **Level Set Methods** - For problems with sharp interfaces

### Learning-Based Methods
- **Adversarial Regularization** - GAN-based regularization
- **NETT (Neural Network Tikhonov)** - Learned regularization functionals
- **LISTA** - Learned Iterative Soft-Thresholding Algorithm
- **Learned Proximal Operators** - Neural network proximal operators

## Features

### Problem Types
- `InverseProblem`: General nonlinear inverse problems
- `LinearInverseProblem`: Specialized type for linear problems
- Support for priors, constraints, and noise models

### Callbacks and Analysis
- `DiscrepancyPrinciple`: Morozov's discrepancy principle for stopping
- `ResidualCallback`: Monitor convergence
- `SolutionSaver`: Save intermediate solutions
- Regularization path computation and L-curve analysis

### Integration with Julia Ecosystem
- Automatic differentiation via ForwardDiff.jl and Zygote.jl (planned)
- Neural network regularizers via Lux.jl (planned)
- GPU acceleration support (planned)
- Statistical priors via Distributions.jl (planned)


## Theory and References

The package implements methods from:

1. Kaltenbacher, B., Neubauer, A., & Scherzer, O. (2008). *Iterative Regularization Methods for Nonlinear Ill-Posed Problems*. Walter de Gruyter.

<!--
2. Engl, H. W., Hanke, M., & Neubauer, A. (1996). *Regularization of Inverse Problems*. Springer.

3. Scherzer, O. (Ed.). (2011). *Handbook of Mathematical Methods in Imaging*. Springer.

For learning-based methods:
- Lunz, S., et al. (2018). "Adversarial Regularizers in Inverse Problems"
- Chen, D., et al. (2021). "Neural Network Based Regularization"
-->

## Contributing

Contributions are welcome! The package is structured to make adding new algorithms straightforward:

1. Define your algorithm struct inheriting from `AbstractIterativeRegularizationAlgorithm`
2. Implement a `_solve` method for your algorithm
3. Add tests and documentation
4. Submit a pull request

See `src/algorithms.jl` and the Nonlinear Landweber implementation in `src/solve.jl` as examples.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{iterativeregularization_jl,
  author = {Outland, Bennet},
  title = {IterativeRegularization.jl: Iterative Regularization Methods for Inverse Problems},
  year = {2025},
  url = {https://github.com/BennetOutland/IterativeRegularization.jl}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This package follows the design principles of the [SciML](https://sciml.ai/) ecosystem and draws inspiration from:
- DifferentialEquations.jl
- Optimization.jl
- RegularizationTools.jl

## Disclosure of Generative AI Usage

Generative AI, Claude, was used in the creation of this library as a programming aid including guided code generation, assistance with performance optimization, and for assistance in writing documentation. All code and documentation included in this repository, whether written by the author(s) or generative AI, has been reviewed by the author(s) for accuracy and has completed a verification and validation process upon release.
