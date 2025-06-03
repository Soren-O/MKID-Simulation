# Example: Using the Improved Quasiparticle Dynamics Simulator

This notebook demonstrates how to use the improved quasiparticle dynamics simulator with its new features.

## Important: Units in the Density Representation

The simulator now uses a **consistent density representation** throughout:

| Quantity | Symbol | Units |
|----------|--------|-------|
| QP density | n(x,E,t) | QPs/eV/μm |
| Injection rate | r_inj | QPs/μm/eV/ns |
| Total QPs | N_total | QPs (dimensionless) |
| Scattering matrix | G^s | 1/ns |
| Recombination matrix | G^r | μm·eV/QPs/ns |

**Key relationships:**
- Total QPs in system: N_total = ΣᵢΣⱼ n[i,j] × Δx × ΔE
- Total QPs at position x: N(x) = Σⱼ n[x,j] × ΔE
- Both G^s and G^r include ΔE factors from integral discretization

## Key Improvements Implemented

### 1. **Robust Numerical Handling**
- Proper error handling for singular matrices and division by zero
- CFL condition checking with warnings
- Numerical stability monitoring

### 2. **Object-Oriented Architecture**
- Clean separation of concerns with dedicated classes
- No global variables - all parameters passed properly
- Modular design for easy extension

### 3. **Conservation Monitoring**
- Tracks total quasiparticle number
- Monitors total energy
- Detects numerical instabilities (NaN/Inf)
- Provides real-time feedback on simulation health

### 4. **Parameter Validation**
- Input validation with meaningful error messages
- Physical parameter consistency checks
- Warning system for potentially problematic configurations

### 5. **Enhanced Usability**
- Structured parameter classes using dataclasses
- Clear logging and progress reporting
- Improved plotting with gap profile overlay

### 6. **Consistent Density Representation**
- All quantities now use density representation: n(x,E) with units [QPs/eV/μm]
- Eliminates confusion about ΔE factors
- Conservation: Total QPs = Σᵢⱼ n[i,j] × Δx × ΔE
- Injection rates in density units: QPs/μm/eV/ns
- Both G^s and G^r matrices include ΔE consistently

## Example Usage

```python
# Import the improved simulator (assuming it's in qp_simulator.py)
from qp_simulator import (
    QuasiparticleSimulator, 
    MaterialParameters, 
    SimulationParameters,
    InjectionParameters,
    constant_gap, step_gap, well_gap
)

# 1. Define material parameters for aluminum
material = MaterialParameters(
    tau_s=400,    # Scattering time (ns)
    tau_r=400,    # Recombination time (ns)
    D_0=6.0,      # Diffusion coefficient (μm²/ns)
    T_c=1.2,      # Critical temperature (K)
    gamma=1e-7,   # Dynes broadening (eV)
    N_0=1.7e10    # Single-spin DOS (eV^-1 μm^-3)
)

# 2. Define simulation parameters
sim_params = SimulationParameters(
    nx=100,       # Spatial cells
    ne=100,       # Energy bins
    nt=1000,      # Time steps
    L=100.0,      # System length (μm)
    T=100.0,      # Simulation time (ns)
    E_min=1.7e-4, # Min energy (eV)
    E_max=8.5e-4  # Max energy (eV)
)

# 3. Create simulator with constant gap
simulator = QuasiparticleSimulator(
    material=material,
    sim_params=sim_params,
    gap_function=constant_gap,
    T_bath=0.1  # Bath temperature (K)
)

# 4. Initialize with thermal distribution
# Now using density representation: n(x,E) in units of QPs/eV/μm
simulator.initialize(init_type='thermal')

# 5. Run simulation with monitoring
simulator.run(plot_interval=200)
```

## Example with Injection

```python
# Define injection parameters
injection = InjectionParameters(
    location=50.0,    # Center of device (μm)
    energy=5.1e-4,    # 3×Δ₀ (eV)
    rate=100,         # QPs per μm per eV per ns (density rate)
    type='pulse',     # Pulse injection
    pulse_duration=1.0 # 1 ns pulse
)

# Run with injection
simulator.run(injection=injection, plot_interval=100)
```

## Example with Step Gap Profile

```python
# Define a step gap function
def my_step_gap(x):
    return step_gap(x, L=100.0)

# Create simulator with step gap
simulator_step = QuasiparticleSimulator(
    material=material,  # Uses the material defined earlier with N_0
    sim_params=sim_params,
    gap_function=my_step_gap,
    T_bath=0.1
)

simulator_step.initialize(init_type='uniform', uniform_density=1e5)  # QPs/eV/μm
simulator_step.run(plot_interval=200)
```

## Accessing Conservation History

```python
# After simulation, access conservation monitoring data
history = simulator.monitor.history

import matplotlib.pyplot as plt

# Plot total quasiparticle number vs time
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['time'], history['total_qp'])
plt.xlabel('Time (ns)')
plt.ylabel('Total QP Number')
plt.title('Quasiparticle Number Evolution')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(history['time'][1:], history['numerical_error'][1:])  # Skip first point
plt.xlabel('Time (ns)')
plt.ylabel('Numerical Error (QPs/ns)')
plt.title('Numerical Stability')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Advanced: Custom Gap Profile

```python
# Define a custom gap profile
def gaussian_gap(x):
    """Gaussian dip in the gap"""
    x0 = 50.0  # Center
    sigma = 10.0  # Width
    delta_min = 1.0e-4
    delta_max = 2.0e-4
    
    gaussian = np.exp(-(x - x0)**2 / (2 * sigma**2))
    return delta_max - (delta_max - delta_min) * gaussian

# Use custom profile
simulator_custom = QuasiparticleSimulator(
    material=material,
    sim_params=sim_params,
    gap_function=gaussian_gap,
    T_bath=0.1
)

simulator_custom.initialize()
simulator_custom.run(plot_interval=100)
```

## Error Handling Example

```python
# The simulator now validates parameters
try:
    bad_material = MaterialParameters(
        tau_s=-400,   # Invalid negative time!
        tau_r=400,
        D_0=6.0,
        T_c=1.2,
        gamma=0,
        N_0=1.7e10
    )
except ValueError as e:
    print(f"Caught error: {e}")

# CFL warnings
fast_sim = SimulationParameters(
    nx=50,      # Coarse grid
    ne=100,
    nt=10000,   # Many time steps
    L=100.0,
    T=10.0,     # Short time → large dt
    E_min=1.7e-4,
    E_max=8.5e-4
)
# This will trigger a CFL warning
```

## Benefits of the Improved Design

1. **Reliability**: Robust error handling prevents crashes and provides meaningful feedback
2. **Modularity**: Easy to extend with new physics or numerical methods
3. **Transparency**: Conservation monitoring shows exactly what's happening
4. **Maintainability**: Clean code structure makes debugging and modifications easier
5. **Performance**: While maintaining clarity, the code is still efficiently vectorized
6. **Physical Clarity**: Consistent density representation eliminates confusion about units and factors

### Density Representation Benefits

Using density (QPs/eV/μm) throughout provides:
- **Mesh independence**: Results don't change when you refine the grid
- **Clear physics**: Directly maps to the continuous diffusion equation
- **Unambiguous units**: No confusion about where ΔE or Δx factors belong
- **Easy conservation checks**: Total = Σ density × volume element

The improved simulator provides a solid foundation for studying quasiparticle dynamics with confidence in the numerical results and physical interpretation.
