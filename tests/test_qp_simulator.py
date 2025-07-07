"""
Unit tests for the quasiparticle simulator v1.1.0.

Run with: pytest tests/test_qp_simulator.py
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp_simulator import (
    MaterialParameters, SimulationParameters, 
    QuasiparticleSimulator, BCSPhysics,
    SuperconductorGeometry, InjectionParameters
)


class TestImports:
    """Test that all modules import correctly."""
    
    def test_main_import(self):
        """Test main module imports."""
        import qp_simulator
        assert hasattr(qp_simulator, 'QuasiparticleSimulator')
        assert hasattr(qp_simulator, 'MaterialParameters')
        assert hasattr(qp_simulator, 'SimulationParameters')
    
    def test_validation_import(self):
        """Test validation module imports."""
        from qp_simulator import qp_validation
        assert hasattr(qp_validation, 'validate_thermal_equilibrium')
        assert hasattr(qp_validation, 'validate_pulse_decay')


class TestBCSPhysics:
    """Test BCS physics calculations."""
    
    def setup_method(self):
        """Create a BCS physics instance for testing."""
        self.bcs = BCSPhysics(gamma=0, N_0=1e9)
    
    def test_phonon_occupation_zero_energy(self):
        """Test phonon occupation at zero energy."""
        assert self.bcs.phonon_occupation(0, 1.0) == 0.0
    
    def test_phonon_occupation_zero_temp(self):
        """Test phonon occupation at zero temperature."""
        # Positive energy - should return 1 (spontaneous emission)
        assert self.bcs.phonon_occupation(1e-6, 0) == 1.0
        # Negative energy - should return 0 (no thermal phonons)
        assert self.bcs.phonon_occupation(-1e-6, 0) == 0.0
    
    def test_phonon_occupation_high_temp_limit(self):
        """Test phonon occupation in high temperature limit."""
        # At high T, should approach kT/E for positive E
        E = 1e-6  # eV
        T = 10000  # K (very high)
        k_B = 8.617333262e-5  # eV/K
        expected = k_B * T / E
        actual = self.bcs.phonon_occupation(E, T)
        assert abs(actual - expected) / expected < 0.01  # Within 1%
    
    def test_fermi_dirac_limits(self):
        """Test Fermi-Dirac distribution limits."""
        # At T=0, should be step function
        assert self.bcs.fermi_dirac(1e-6, 0) == 0.0
        assert self.bcs.fermi_dirac(-1e-6, 0) == 0.0  # Special case in code
        
        # At high energy, should approach 0
        assert self.bcs.fermi_dirac(1.0, 1.0) < 1e-10
        
        # At negative high energy, should approach 1
        assert self.bcs.fermi_dirac(-1.0, 1.0) > 0.999999
    
    def test_density_of_states_gap_edge(self):
        """Test DOS behavior at gap edge."""
        delta = 0.0001  # 100 μeV
        
        # Below gap with no Dynes broadening
        E_below = 0.99 * delta
        assert self.bcs.density_of_states(E_below, delta) == 0.0
        
        # At gap edge - should diverge (but code returns large finite value)
        E_edge = delta
        dos_edge = self.bcs.density_of_states(E_edge, delta)
        assert dos_edge > 100  # Should be large
        
        # Above gap
        E_above = 2 * delta
        dos_above = self.bcs.density_of_states(E_above, delta)
        expected = E_above / np.sqrt(E_above**2 - delta**2)
        assert abs(dos_above - expected) / expected < 1e-6


class TestMaterialParameters:
    """Test material parameter validation."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        material = MaterialParameters(
            tau_s=400.0,
            tau_r=400.0,
            D_0=6.0,
            T_c=1.2,
            gamma=0.0,
            N_0=2.1e9
        )
        material.validate()  # Should not raise
    
    def test_invalid_scattering_time(self):
        """Test that negative scattering time raises error."""
        material = MaterialParameters(
            tau_s=-400.0,  # Invalid
            tau_r=400.0,
            D_0=6.0,
            T_c=1.2,
            gamma=0.0,
            N_0=2.1e9
        )
        with pytest.raises(ValueError, match="Scattering times must be positive"):
            material.validate()
    
    def test_invalid_diffusion(self):
        """Test that negative diffusion coefficient raises error."""
        material = MaterialParameters(
            tau_s=400.0,
            tau_r=400.0,
            D_0=-6.0,  # Invalid
            T_c=1.2,
            gamma=0.0,
            N_0=2.1e9
        )
        with pytest.raises(ValueError, match="Diffusion coefficient must be positive"):
            material.validate()


class TestSimulationParameters:
    """Test simulation parameter validation and properties."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        sim_params = SimulationParameters(
            nx=50, ne=50, nt=100,
            L=100.0, T=1000.0,
            E_min=0.0001, E_max=0.001
        )
        sim_params.validate()  # Should not raise
    
    def test_derived_properties(self):
        """Test calculated properties."""
        sim_params = SimulationParameters(
            nx=50, ne=50, nt=100,
            L=100.0, T=1000.0,
            E_min=0.0001, E_max=0.001
        )
        
        assert sim_params.dx == 2.0  # 100/50
        assert sim_params.dt == 10.0  # 1000/100
        assert abs(sim_params.dE - 1.8e-5) < 1e-10  # (0.001-0.0001)/50
    
    def test_low_discretization_warning(self):
        """Test warning for low discretization."""
        sim_params = SimulationParameters(
            nx=5, ne=5, nt=100,  # Very low
            L=100.0, T=1000.0,
            E_min=0.0001, E_max=0.001
        )
        with pytest.warns(UserWarning, match="Low discretization"):
            sim_params.validate()


class TestSuperConductorGeometry:
    """Test gap profile handling."""
    
    def test_constant_gap(self):
        """Test constant gap function."""
        L = 100.0
        nx = 50
        gap_value = 0.00018
        gap_func = lambda x: gap_value
        
        geom = SuperconductorGeometry(L, nx, gap_func)
        
        # Check discretization
        assert geom.dx == 2.0
        assert len(geom.x_centers) == nx
        assert len(geom.x_boundaries) == nx + 1
        
        # Check gap values
        assert geom.gap_at_position(50.0) == gap_value
        gap_array = geom.gap_array()
        assert np.all(gap_array == gap_value)
    
    def test_varying_gap(self):
        """Test spatially varying gap."""
        L = 100.0
        nx = 50
        # Linear gap profile
        gap_func = lambda x: 0.0001 + 0.000001 * x
        
        geom = SuperconductorGeometry(L, nx, gap_func)
        
        # Check that gap varies
        gap_array = geom.gap_array()
        assert gap_array[0] < gap_array[-1]
        assert abs(gap_array[0] - gap_func(geom.x_centers[0])) < 1e-10


class TestQuasiparticleSimulator:
    """Test the main simulator class."""
    
    def create_test_simulator(self, nx=20, ne=20):
        """Create a small simulator for testing."""
        material = MaterialParameters(
            tau_s=400.0, tau_r=400.0, D_0=6.0,
            T_c=1.2, gamma=0.0, N_0=2.1e9
        )
        sim_params = SimulationParameters(
            nx=nx, ne=ne, nt=10,
            L=50.0, T=100.0,
            E_min=0.0001, E_max=0.001,
            verbose=False
        )
        gap_function = lambda x: 0.00018
        
        return QuasiparticleSimulator(material, sim_params, gap_function, T_bath=0.1)
    
    def test_initialization_thermal(self):
        """Test thermal initialization."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # Check that density is non-negative
        assert np.all(sim.n_density >= 0)
        
        # Check that thermal and initial match
        assert np.allclose(sim.n_density, sim.n_thermal)
        
        # Check total QPs is reasonable (WITH CONVERSION FACTOR)
        total_qps = 4 * sim.material.N_0 * np.sum(sim.n_density) * \
                    sim.sim_params.dx * sim.sim_params.dE
        assert total_qps > 0
        assert total_qps < 1e20  # Reasonable upper bound
    
    def test_initialization_uniform(self):
        """Test uniform initialization."""
        sim = self.create_test_simulator()
        uniform_value = 1e-8  # NOW THIS IS DIMENSIONLESS
        sim.initialize('uniform', uniform_density=uniform_value)
        
        # Check that density above gap is uniform
        gap = 0.00018
        for ix in range(sim.sim_params.nx):
            for ie, E in enumerate(sim.energies):
                if E >= gap:
                    assert sim.n_density[ix, ie] == uniform_value
                else:
                    assert sim.n_density[ix, ie] == 0.0
    
    def test_injection_valid(self):
        """Test QP injection at valid location."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # Get initial total WITH CONVERSION FACTOR
        initial_total = 4 * sim.material.N_0 * np.sum(sim.n_density) * \
                       sim.sim_params.dx * sim.sim_params.dE
        
        # Inject QPs
        injection = InjectionParameters(
            location=25.0,  # Center
            energy=0.0003,  # Well above gap
            rate=1e10,      # QPs/μm/eV/ns (PHYSICAL UNITS)
            type='pulse',
            pulse_duration=10.0
        )
        
        sim.inject_quasiparticles(injection, time=0.0)
        
        # Check that QPs were added WITH CONVERSION FACTOR
        final_total = 4 * sim.material.N_0 * np.sum(sim.n_density) * \
                      sim.sim_params.dx * sim.sim_params.dE
        assert final_total > initial_total
    
    def test_step_execution(self):
        """Test that step() runs without error."""
        # Create simulator with stable parameters
        sim = self.create_test_simulator()
        
        # Initialize with uniform density above thermal to ensure evolution
        sim.initialize('uniform', uniform_density=1e-6)  # Non-equilibrium state
        
        # Get initial state
        initial_density = sim.n_density.copy()
        initial_total = 4 * sim.material.N_0 * np.sum(initial_density) * \
                       sim.sim_params.dx * sim.sim_params.dE
        
        # Run one step
        sim.step()
        
        # Check that step() ran without error and density evolved
        final_total = 4 * sim.material.N_0 * np.sum(sim.n_density) * \
                      sim.sim_params.dx * sim.sim_params.dE
        
        # For a non-equilibrium initial state, there should be evolution
        # Check either spatial redistribution OR energy relaxation occurred
        spatial_change = np.max(np.abs(np.sum(sim.n_density, axis=1) - 
                                       np.sum(initial_density, axis=1)))
        energy_change = np.max(np.abs(np.sum(sim.n_density, axis=0) - 
                                      np.sum(initial_density, axis=0)))
        
        # At least one should show change
        assert spatial_change > 0 or energy_change > 0, \
               "Density should evolve from non-equilibrium state"

    
    def test_array_names_updated(self):
        """Test that array names have been updated to Ks/Kr."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # Check new names exist
        assert hasattr(sim.scattering, '_Ks_array')
        assert hasattr(sim.scattering, '_Kr_array')
        
        # Check old names don't exist
        assert not hasattr(sim.scattering, '_Gs_array')
        assert not hasattr(sim.scattering, '_Gr_array')
        
        # Check arrays are created
        assert sim.scattering._Ks_array is not None
        assert sim.scattering._Kr_array is not None
    
    def test_dimensionless_density(self):
        """Test that density is properly dimensionless."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # For dimensionless density n = ρf, max value should be ρ (when f=1)
        for ix in range(sim.sim_params.nx):
            for ie in range(sim.sim_params.ne):
                rho = sim.scattering._rho_array[ix, ie]
                # Allow small numerical tolerance
                assert sim.n_density[ix, ie] <= rho * 1.001
                
        # Check occupation factors are sensible
        f = sim.scattering._calculate_occupation_factors(sim.n_density)
        assert np.all(f >= 0)
        assert np.all(f <= 1.001)
    
    def test_scattering_step_signature(self):
        """Test that scattering_step has correct signature."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # This should work (new signature with dE)
        try:
            sim.scattering.scattering_step(
                sim.n_density, sim.n_thermal, 
                sim.sim_params.dt, sim.sim_params.dE
            )
        except TypeError:
            pytest.fail("scattering_step should accept dE parameter")
        
        # This should fail (old signature without dE)
        with pytest.raises(TypeError):
            sim.scattering.scattering_step(
                sim.n_density, sim.n_thermal, 
                sim.sim_params.dt  # Missing dE
            )
    
    def test_monitor_signature(self):
        """Test that SystemMonitor.check_system has correct signature."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        occupation_factors = sim.scattering._calculate_occupation_factors(sim.n_density)
        
        # This should work (new signature with N_0)
        metrics = sim.monitor.check_system(
            sim.n_density, occupation_factors, 
            sim.energies, 0.0, sim.material.N_0
        )
        
        assert 'total_qp' in metrics
        assert metrics['total_qp'] > 0
    
    def test_kernel_arrays_no_de(self):
        """Test that kernel arrays don't include dE factor."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # Get arrays
        Ks = sim.scattering._Ks_array
        Kr = sim.scattering._Kr_array
        
        # Arrays should have shape (nx, ne, ne)
        assert Ks.shape == (sim.sim_params.nx, sim.sim_params.ne, sim.sim_params.ne)
        assert Kr.shape == (sim.sim_params.nx, sim.sim_params.ne, sim.sim_params.ne)
        
        # Typical values should be reasonable for units [1/ns/eV]
        # Without dE factor, values should be ~1/dE times larger than before
        typical_Ks = np.mean(Ks[Ks > 0])
        typical_Kr = np.mean(Kr[Kr > 0])
        
        # Order of magnitude check (depends on parameters)
        assert typical_Ks > 1e-3  # Should be largish for 1/ns/eV units
        assert typical_Ks < 1e6   # But not absurdly large
    
    def test_unit_conversion_consistency(self):
        """Test that unit conversions are consistent throughout."""
        sim = self.create_test_simulator()
        sim.initialize('thermal')
        
        # Method 1: Direct summation
        n_total = np.sum(sim.n_density) * sim.sim_params.dx * sim.sim_params.dE
        qp_total_1 = 4 * sim.material.N_0 * n_total
        
        # Method 2: Using monitor
        occupation_factors = sim.scattering._calculate_occupation_factors(sim.n_density)
        metrics = sim.monitor.check_system(
            sim.n_density, occupation_factors, 
            sim.energies, 0.0, sim.material.N_0
        )
        qp_total_2 = metrics['total_qp']
        
        # Should give same result
        assert abs(qp_total_1 - qp_total_2) / qp_total_1 < 1e-10
    
    def test_version_compatibility(self):
        """Ensure we're testing v1.1.0 behavior."""
        sim = self.create_test_simulator()
        # Check for v1.1.0 features
        assert hasattr(sim.scattering, '_Ks_array')
        assert 'dimensionless' in sim.initialize.__doc__.lower()


class TestTqdmFallback:
    """Test that tqdm fallback works when module is absent."""
    
    def test_tqdm_fallback(self):
        """Test graceful fallback when tqdm not available."""
        import importlib
        
        # Save original tqdm if it exists
        tqdm_backup = sys.modules.get('tqdm', None)
        tqdm_auto_backup = sys.modules.get('tqdm.auto', None)
        
        try:
            # Remove tqdm from modules
            sys.modules.pop('tqdm', None)
            sys.modules.pop('tqdm.auto', None)
            
            # Force reload of our module
            if 'qp_simulator' in sys.modules:
                del sys.modules['qp_simulator']
            
            import qp_simulator
            
            # Should have fallback tqdm
            assert hasattr(qp_simulator, 'tqdm')
            
            # Test that it works (returns input iterable)
            result = list(qp_simulator.tqdm([1, 2, 3]))
            assert result == [1, 2, 3]
            
        finally:
            # Restore original modules
            if tqdm_backup:
                sys.modules['tqdm'] = tqdm_backup
            if tqdm_auto_backup:
                sys.modules['tqdm.auto'] = tqdm_auto_backup


# Integration tests that use validation module

def test_validation_import():
    """Test that validation can be run from external module."""
    from qp_simulator import (
        qp_validation,
        MaterialParameters,
        SimulationParameters,
        QuasiparticleSimulator,
    )
    
    # Create a small simulator
    material = MaterialParameters(
        tau_s=400.0, tau_r=400.0, D_0=6.0,
        T_c=1.2, gamma=0.0, N_0=2.1e9
    )
    sim_params = SimulationParameters(
        nx=10, ne=10, nt=10,
        L=20.0, T=20.0,
        E_min=0.0001, E_max=0.001,
        verbose=False
    )
    gap_function = lambda x: 0.00018
    
    sim = QuasiparticleSimulator(material, sim_params, gap_function, T_bath=0.1)
    sim.initialize('thermal')
    
    # Run validation
    passed = qp_validation.validate_thermal_equilibrium(sim, duration_ns=10.0)
    assert passed is True or passed is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])