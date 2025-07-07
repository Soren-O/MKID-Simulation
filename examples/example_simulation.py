#!/usr/bin/env python3
"""
Example simulation script demonstrating basic usage of the quasiparticle simulator v1.1.0.

This can be run from the command line:
    python example_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from qp_simulator import (
    MaterialParameters, SimulationParameters,
    QuasiparticleSimulator, InjectionParameters
)
import qp_validation as qpv


def main():
    """Run a simple quasiparticle simulation with v1.1.0 notation."""
    
    print("Quasiparticle Dynamics Simulation Example (v1.1.0)")
    print("=" * 50)
    
    # 1. Define material parameters (Aluminum)
    print("\n1. Setting up material parameters...")
    material = MaterialParameters(
        tau_s=400.0,    # Electron-phonon scattering time (ns)
        tau_r=400.0,    # Electron-phonon recombination time (ns)
        D_0=6.0,        # Normal state diffusion coefficient (μm²/ns)
        T_c=1.2,        # Critical temperature (K)
        gamma=0.0,      # Dynes broadening parameter (eV)
        N_0=2.1e9       # Single-spin DOS at Fermi level (eV^-1 μm^-3)
    )
    
    # 2. Define simulation parameters
    print("2. Setting up simulation parameters...")
    sim_params = SimulationParameters(
        nx=50,          # Number of spatial cells
        ne=30,          # Number of energy bins
        nt=500,         # Number of time steps
        L=100.0,        # System length (μm)
        T=2000.0,       # Simulation duration (ns)
        E_min=0.0001,   # Minimum energy: 100 μeV
        E_max=0.001,    # Maximum energy: 1000 μeV
        verbose=True    # Show progress bars
    )
    
    # 3. Define gap profile (constant for simplicity)
    gap_value = 0.00018  # 180 μeV
    gap_function = lambda x: gap_value
    
    # 4. Create simulator
    print("3. Creating simulator...")
    bath_temperature = 0.1  # K
    sim = QuasiparticleSimulator(material, sim_params, gap_function, bath_temperature)
    
    # 5. Initialize with thermal distribution
    print("4. Initializing thermal distribution...")
    sim.initialize('thermal')
    
    # Note: Now using dimensionless density internally
    # Convert to physical units for display
    initial_qps = 4 * material.N_0 * np.sum(sim.n_density) * \
                  sim_params.dx * sim_params.dE
    print(f"   Initial QP count: {initial_qps:.2e}")
    print("   Note: Using dimensionless density n(E,x) internally")
    
    # 6. Test thermal equilibrium
    print("\n5. Testing thermal equilibrium...")
    equilibrium_ok = qpv.validate_thermal_equilibrium(sim, duration_ns=200.0, tolerance=1e-5)
    if equilibrium_ok:
        print("   ✓ Thermal equilibrium maintained")
    else:
        print("   ✗ Thermal equilibrium violated!")
    
    # 7. Test pulse injection and decay
    print("\n6. Testing pulse injection and decay...")
    sim.initialize('thermal')  # Reset
    
    injection_energy = 0.0003   # 300 μeV
    injection_location = 50.0   # Center of system
    pulse_qps = 1e5            # Number of QPs to inject
    
    times, totals = qpv.validate_pulse_decay(
        sim,
        injection_energy=injection_energy,
        injection_location=injection_location,
        pulse_qps=pulse_qps,
        duration_ns=1000.0
    )
    
    # 8. Plot results
    print("\n7. Plotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # QP decay plot
    ax1.plot(times, totals, 'b-', linewidth=2)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Total QPs')
    ax1.set_title('Quasiparticle Decay')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Final spatial distribution
    # Convert dimensionless to physical units
    spatial_density = 4 * material.N_0 * np.sum(sim.n_density, axis=1) * sim_params.dE
    ax2.plot(sim.geometry.x_centers, spatial_density, 'r-', linewidth=2)
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('QP density (QPs/μm)')
    ax2.set_title('Final Spatial Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to simulation_results.png")
    
    # 9. Check Pauli exclusion
    print("\n8. Checking Pauli exclusion principle...")
    pauli_ok, max_f, _ = qpv.check_pauli_exclusion(sim)
    print(f"   Maximum occupation factor: {max_f:.3f}")
    if pauli_ok:
        print("   ✓ Pauli exclusion satisfied (f ≤ 1)")
    else:
        print("   ✗ Pauli exclusion violated!")
    
    # 10. Display kernel information
    print("\n9. Kernel array information...")
    print(f"   Scattering kernel (Ks) shape: {sim.scattering._Ks_array.shape}")
    print(f"   Recombination kernel (Kr) shape: {sim.scattering._Kr_array.shape}")
    print("   Kernel units: [1/ns/eV] (no pre-multiplied dE)")
    
    # 11. Export data
    print("\n10. Exporting data...")
    qpv.export_for_analysis(sim, filename='simulation_data.npz')
    print("   Saved data to simulation_data.npz")
    print("   Data uses dimensionless density representation")
    
    print("\n" + "=" * 50)
    print("Simulation complete!")
    print("\nKey v1.1.0 changes demonstrated:")
    print("- Kernels renamed: Gs→Ks, Gr→Kr")
    print("- Dimensionless density: n(E,x) = ρ(E,x)f(E,x)")
    print("- Physical QPs = 4N₀ × ∫∫ n dx dE")
    print("- Consistent discretization handling")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass


if __name__ == '__main__':
    main()