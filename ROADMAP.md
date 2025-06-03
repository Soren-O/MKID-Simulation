# Project Roadmap for MKID Quasiparticle Dynamics Simulator

This document outlines planned features, improvements, and long-term goals for the Quasiparticle Dynamics Simulator. The aim is to enhance its physical accuracy, numerical robustness, performance, and usability.

## Phase 1: Core Physics Accuracy and Foundational Robustness

* [ ] **Refine Recombination Term Implementation:**
    * [ ] Double-check the derivation and implementation of the recombination term in `ScatteringSolver.scattering_step` to ensure units and physical interpretation are perfectly aligned with $n_{density}$ representing QPs/eV/µm and $\mathcal{G}^r$ elements representing rates ($1/\text{ns}$). Ensure the "number of partners" term $(n_j \Delta E_j \Delta x_j)$ is correctly applied.
    * [ ] Update docstrings for `Gr_array` units in `ScatteringSolver.create_scattering_arrays` to accurately reflect $1/\text{ns}$.
* [ ] **Implement Pauli Blocking Factors $(1-f)$ for Scattering:**
    * [ ] Modify the scattering-in term in `ScatteringSolver.scattering_step` to include the $(1-f_k)$ factor for the final state $E_k$, where $f_k = n_k / (2 N_0 \rho_k)$.
    * [ ] Modify the scattering-out term (or the calculation of `_Gs_array` elements) to correctly incorporate the $(1-f_j)$ factor for the final state $E_j$ into which scattering occurs.
* [ ] **Enhance Numerical Stability Checks:**
    * [ ] Implement more robust error handling (e.g., specific exceptions) in `DiffusionSolver.create_thomas_factors` if denominators become critically small, beyond current logging.
    * [ ] Systematically review all physics calculations for robust handling of potential divisions by zero or singularities (e.g., Dynes DOS, diffusion coefficient $D(E,x)$ near $E=\Delta$).

## Phase 2: Advanced Physics Modeling

* [ ] **Self-Consistent Gap Suppression:**
    * [ ] Implement dynamic feedback: $\Delta(x,t) = \Delta_0(x) (1 - \alpha \frac{N_{qp}(x,t)}{N_{cp}(x)})$.
    * [ ] Define $N_{qp}(x,t) = \int n(x,E,t) dE$.
    * [ ] Define $N_{cp}(x)$ (e.g., based on $2 N_0 \Delta_0(x) \cdot E_{char}$).
    * [ ] Implement dynamic recalculation of $\Delta(x,t)$ and all $\Delta$-dependent matrices ($D$, $\mathcal{G}^s$, $\mathcal{G}^r$) during the simulation loop.
    * [ ] Investigate need for iterative self-consistency within each time step.
* [ ] **Accurate Temperature-Dependent Equilibrium Gap $\Delta(T)$:**
    * [ ] For improved equilibrium studies, implement an iterative solver for the full BCS self-consistency equation for $\Delta(T)$ in `BCSPhysics`.

## Phase 3: Numerical Methods and Performance Optimization

* [ ] **Operator Splitting Schemes:**
    * [ ] Implement Strang splitting (or other higher-order schemes) in `QuasiparticleSimulator.step` for potentially improved temporal accuracy.
* [ ] **Alternative Diffusion Boundary Conditions:**
    * [ ] Extend `DiffusionSolver` to support user-selectable Dirichlet and Robin boundary conditions.
* [ ] **Adaptive Time Stepping:**
    * [ ] Explore and implement adaptive $\Delta t$ adjustment based on error estimation or solution change, with appropriate safeguards.
* [ ] **Energy Grid Strategies:**
    * [ ] Allow user-defined static non-uniform energy grids (finer resolution near $\Delta(x)$, etc.).
    * [ ] (Long-term) Investigate Adaptive Mesh Refinement (AMR) for the energy grid.
* [ ] **Performance Profiling and Optimization:**
    * [ ] Profile the code to identify bottlenecks.
    * [ ] Apply `numba.jit` or further vectorization to critical loops where beneficial.

## Phase 4: Usability, Workflow, and Broader Modeling

* [ ] **Enhanced Configuration Validation:**
    * [ ] Augment `ConfigurationManager` with schema validation (e.g., using Pydantic) for loaded YAML files.
* [ ] **Command-Line Interface (CLI):**
    * [ ] Develop a CLI using `argparse` or `Click`/`Typer` to run simulations with configuration files.
* [ ] **Expanded Gap Function Configuration:**
    * [ ] Allow specification of more generic gap functions or loading custom $\Delta(x)$ profiles from data files via the configuration system.
* [ ] **Dynamic Phonon Bath (Hot Phonons):**
    * [ ] (Long-term) Model non-equilibrium phonon dynamics ($N_p(\omega, x, t)$) coupled to the QP equations.
* [ ] **Impurity Scattering:**
    * [ ] (Long-term) Add terms for elastic scattering by impurities.

## Phase 5: Validation, Testing, and Documentation

* [ ] **Expand Test Suite:**
    * [ ] Implement comprehensive unit tests for all core classes and methods.
    * [ ] Develop benchmark tests against analytical solutions in simplified limits.
    * [ ] Perform and document convergence studies with respect to $\Delta t, \Delta x, \Delta E$.
* [ ] **Comprehensive Documentation:**
    * [ ] Create detailed user documentation (physical models, numerical methods, parameters, config file structure, examples).
    * [ ] Maintain and improve API documentation (docstrings).