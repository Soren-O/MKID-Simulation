# Project Roadmap for Quasiparticle Dynamics Simulator

This document outlines planned features, improvements, and long-term goals for the simulator.

---

## Physics Testing:

- [ ] **Comparison of diffusion to theoretical solutions:**
    - [ ] Turn off all physics except the diffusion. Certain initial conditions have analytic solutions for the diffusion equation. The code should accurately reproduce these
- [ ] **Comparison of scattering to theoretical solutions:**
    - [ ] Turn off all physics except the scattering. Verify that the decay is exponential with the expected lifetime
- [ ] **Comparison of recombination to theoretical solutions:**
    - [ ] Turn off all physics except the recombination. Verify that the decay is as expected (second-order decay)

## Additional Physics Modeling:

- [ ] **Self-Consistent Gap Suppression and Time-Dependent Gap Dynamics:**
    - [ ] Implement dynamic feedback where the gap `Δ(x, t)` is suppressed by the local QP density `n(x, t)`.
    - [ ] Implement dynamic recalculation of all `Δ`-dependent matrices (`D`, `ρ`, `K_s`, `K_r`) during the simulation loop.
    - [ ] Investigate the need for an adiabatic "lifting term" that changes the quasiparticle energies due to the changing gap Δ(x, t)
- [ ] **Spatially-Dependent Gap Dynamics:**
    - [ ] Implement an effective force term (`-∂Δ/∂x`) that pushes quasiparticles out of regions with a higher gap.
- [ ] **Phonon Physics:**
    - [ ] Full treatment of phonon physics and for non-equilibrium phonons (currently the code is only for phonons at a bath temperature)
- [ ] **Alternative Diffusion Boundary Conditions:**
    - [ ] Extend `DiffusionSolver` to support user-selectable Dirichlet (fixed value) and Robin (mixed) boundary conditions.
- [ ] **2D/3D Physics:**
    - [ ] Expand the model to 2 or 3 spatial dimensions
- [ ] **Additional Physics:**
    - [ ] (Long-term goal) Add terms for elastic scattering by impurities.
    - [ ] (Long-term goal) Investigate and add non-local effects for the collision integral.

## Numerical Methods and Performance:

- [ ] **Numerical Stability Checks:**
    - [ ] Analyze more completely conditions or numerical stability. Even if it is numerically stable, there can still be non-physical oscillations that are the result of the numerical method
- [ ] **Implicit Solver for Collision Integral:**
    - [ ] Replace the current explicit Euler method in `ScatteringSolver` with a stiff ODE solver (e.g., from `scipy.integrate.solve_ivp` with `method='Radau'` or `BDF`) to allow for larger, stable time steps. This is a high-priority item for enabling long-timescale simulations.
- [ ] **Adaptive Time Stepping:**
    - [ ] Explore and implement an adaptive `Δt` adjustment based on the rate of change of the solution, allowing the simulation to take large steps during slow evolution and small steps during rapid changes.
- [ ] **Performance Profiling and Optimization:**
    - [ ] Profile the code to identify computational bottlenecks.
    - [ ] Apply `numba.jit` or further vectorization to critical loops.
    - [ ] Investigate parallelization/GPU acceleration for the linear algebra and grid operations.

## Usability and Expanded Modeling:

- [ ] **Command-Line Interface (CLI):**
    - [ ] Develop a CLI using `argparse` or `Click` to run simulations from a terminal using configuration files (e.g., YAML), facilitating batch jobs and parameter sweeps.
- [ ] **Enhanced Configuration and Validation:**
    - [ ] Augment configuration loading with schema validation (e.g., using Pydantic) to ensure config files are well-formed.
    - [ ] Allow loading custom `Δ(x)` profiles from data files.
