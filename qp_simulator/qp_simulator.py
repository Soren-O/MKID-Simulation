"""
Quasiparticle Dynamics Simulator

This module implements a numerical solver for non-equilibrium quasiparticle dynamics in superconductors,
solving the diffusion equation with collision integrals, and with treatment of:
- BCS density of states with optional Dynes broadening
- Pauli blocking for scattering processes
- Quasiparticle-phonon scattering 
- Quasiparticle-phonon recombination
- Quasiparticle-phonon generation
- Energy dependent diffusion coefficient 
- Spatially varying diffusion coefficient due to a spatially varying gap-parameter

This module does not have treatment of:
- Non-equilibrium phonons; all phonon mediated dynamics assume the phonons are at steady bath temperature
- Effective force due to the spatially varying gap-parameter
- Self-consistent gap equation changing the gap-parameter due to changes in quasiparticle density

Note: This implementation internally uses dimensionless density n(E,x) = ρ(E,x)f(E,x). Physical quantities (total QPs, etc.) are obtained by multiplying
by 4N₀ where needed. 

For validation and analysis tools, see qp_validation.py
For usage examples, see the notebooks/examples folder

Author: Soren Ormseth
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from scipy.constants import physical_constants as pyc
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import warnings
import logging

# Optional progress bar with graceful fallback
try:
    from tqdm.auto import tqdm #tdqm.auto picks the progress-bar best for your environmet (Jupyter notebook vs terminal)
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable #if tqdm is unavailable then when you call it this will just give you back the original iterable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants
EPS_ENERGY = 1e-30  # eV² - prevents division by zero in energy products

# ============================================================================
# Data Classes for Parameters
# ============================================================================

@dataclass
class MaterialParameters:
    """Physical parameters of the superconductor material"""
    tau_s: float  # Electron-phonon scattering time (ns)
    tau_r: float  # Electron-phonon recombination time (ns)
    D_0: float    # Normal state diffusion coefficient (μm²/ns)
    T_c: float    # Critical temperature (K)
    gamma: float  # Dynes broadening parameter (eV)
    N_0: float    # Single-spin DOS at Fermi level (eV^-1 μm^-3)
    
    def validate(self):
        """Validate physical parameters"""
        if self.tau_s <= 0 or self.tau_r <= 0:
            raise ValueError("Scattering times must be positive")
        if self.D_0 <= 0:
            raise ValueError("Diffusion coefficient must be positive")
        if self.T_c <= 0:
            raise ValueError("Critical temperature must be positive")
        if self.gamma < 0:
            raise ValueError("Dynes parameter must be non-negative")
        if self.N_0 <= 0:
            raise ValueError("Density of states must be positive")

@dataclass
class SimulationParameters:
    """Numerical simulation parameters"""
    nx: int      # Number of spatial cells
    ne: int      # Number of energy bins
    nt: int      # Number of time steps
    L: float     # System length (μm)
    T: float     # Simulation duration (ns)
    E_min: float # Minimum energy (eV)
    E_max: float # Maximum energy (eV)
    verbose: bool = False  # Enable progress bars
    rtol: float = 1e-8     # Relative tolerance for implicit solver
    atol: float = 1e-12    # Absolute tolerance for implicit solver
    
    def validate(self):
        """Validate simulation parameters"""
        if self.nx < 10 or self.ne < 10:
            warnings.warn("Low discretization may lead to inaccurate results")
        if self.E_min <= 0:
            raise ValueError("E_min must be positive")
        if self.E_max <= self.E_min:
            raise ValueError("E_max must be greater than E_min")
    
    @property
    def dx(self) -> float:
        """Spatial step size"""
        return self.L / self.nx
        
    @property
    def dt(self) -> float:
        """Time step size"""
        return self.T / self.nt
        
    @property
    def dE(self) -> float:
        """Energy step size"""
        return (self.E_max - self.E_min) / self.ne

    @property
    def energy_grid(self) -> np.ndarray:
        """Center points of the energy bins."""
        return self.E_min + (np.arange(self.ne) + 0.5) * self.dE

@dataclass
class InjectionParameters:
    """Quasiparticle injection parameters. 
    If you have a quasiparticle-current in units of per ns then you'll need to do simulation_rate = your_qp_current / (dx * dE)"""
    location: float      # Injection location (μm)
    energy: float        # Injection energy (eV)
    rate: float          # Injection rate density (QPs/μm/eV/ns).
    type: str = 'continuous'  # 'continuous' or 'pulse'
    pulse_duration: float = 0.0  # Pulse duration for pulse type (ns)

# ============================================================================
# Core Physics Classes
# ============================================================================

class SuperconductorGeometry:
    """
    Handles spatial discretization and gap profile.
    This version pre-calculates gap arrays for improved performance.
    """

    def __init__(self, L: float, nx: int, gap_function: Callable[[float], float]):
        self.L = L
        self.nx = nx
        self.dx = L / nx
        self.gap_function = gap_function

        # --- Pre-calculate coordinate arrays ---
        self.x_centers = (np.arange(nx) + 0.5) * self.dx
        self.x_boundaries = np.arange(nx + 1) * self.dx

        # --- Pre-calculate and cache gap arrays ---
        # Create a vectorized version of the gap function once.
        vec_gap = np.vectorize(self.gap_function)

        # Calculate the gap arrays and store them as private attributes.
        self._gap_array_centers = vec_gap(self.x_centers)
        self._gap_array_boundaries = vec_gap(self.x_boundaries[1:-1])

    def gap_at_position(self, x: float) -> float:
        """
        Get the gap Δ at an arbitrary position x.
        """
        return self.gap_function(x)

    @property
    def gap_at_center(self) -> np.ndarray:
        """Gets the pre-calculated gap values at all cell centers."""
        return self._gap_array_centers

    @property
    def delta_at_inner_boundaries(self) -> np.ndarray:
        """Gets the pre-calculated gap values at each interior boundary."""
        return self._gap_array_boundaries


class BCSPhysics:
    """
    Vectorized BCS theory calculations with Dynes broadening.
    All public methods now operate on NumPy arrays.
    """

    def __init__(self, gamma: float, N_0: float):
        self.gamma = gamma
        self.N_0 = N_0

    def density_of_states(self, E: np.ndarray, delta: float) -> np.ndarray:
        """
        Calculate normalized superconducting density of states with Dynes broadening.
        This is a vectorized version of the Dynes DOS formula.
        """
        # Handle the ideal case (gamma=0) where DOS is zero below the gap
        if self.gamma == 0:
            return np.where(E >= delta, E / np.sqrt(E**2 - delta**2), 0.0)

        # Vectorized complex arithmetic for Dynes broadening
        E_complex = E - 1j * self.gamma

        arg = E_complex**2 - delta**2
        sqrt_arg = np.sqrt(arg)

        # The original code's logic to ensure the correct branch cut, vectorized
        fix_mask = sqrt_arg.real < 0
        sqrt_arg[fix_mask] = -sqrt_arg[fix_mask]

        # Calculate result and ensure it's non-negative
        result = np.real(E_complex / sqrt_arg)
        return np.maximum(0.0, result)

    def fermi_dirac(self, E: np.ndarray, T: float) -> np.ndarray:
        """Vectorized Fermi-Dirac distribution."""
        if T == 0:
            return np.where(E > 0, 0.0, 1.0) # Step function at T=0

        k_B = pyc['Boltzmann constant in eV/K'][0]
        arg = E / (k_B * T)

        # Use np.exp directly, which is naturally vectorized.
        # np.exp handles large arguments by returning inf.
        return 1.0 / (np.exp(arg) + 1.0)

    def phonon_occupation(self, E: np.ndarray, T: float) -> np.ndarray:
        """
        Vectorized phonon occupation number N_p(E,T).
        Handles phonon emission (E > 0) and absorption (E < 0).
        """
        # Handle zero temperature case first
        if T == 0:
            return np.where(E > 0, 1.0, 0.0)

        k_B = pyc['Boltzmann constant in eV/K'][0]
        arg = -E / (k_B * T)

        # Calculate the exponential term
        exp_term = np.exp(arg)
        
        # Calculate the denominator, avoiding division by zero
        denominator = np.abs(exp_term - 1.0)
        
        # Where E is very close to 0, the denominator is near zero.
        # The result of the division would be inf, which is physically correct
        # for the Bose-Einstein distribution, but we set it to 0 as in the
        # original code since self-scattering is ignored.
        result = np.divide(1.0, denominator, where=denominator!=0, 
                           out=np.full_like(E, 0.0))
        
        return np.where(E == 0, 0.0, result)

class DiffusionSolver:
    """
    Crank-Nicolson solver for diffusion equation
    
    This solver takes a SuperconductorGeometry instance and pre-computes the Thomas algorithm factors for efficiency,
    avoiding redundant calculations during the simulation.
    """
    
    def __init__(self, geometry: SuperconductorGeometry, D_0: float):
        self.geometry = geometry
        self.D_0 = D_0
        self._thomas_factors = None
        self._diff_array = None
        
    def create_diffusion_array(self, energies: np.ndarray) -> np.ndarray:
        """
        Creation of the diffusion coefficient array D[boundary_idx, energy_idx].
        """
        # Get gap values at all interior boundaries, shape (nx-1)
        delta_vals = self.geometry.delta_at_inner_boundaries()

        # Use broadcasting to compute D for all boundaries and energies simultaneously
        # energies shape -> (1, ne)
        # delta_vals shape -> (nx-1, 1)
        E = energies[None, :]
        delta = delta_vals[:, None]

        arg = 1.0 - (delta / E)**2

        # Use np.where to apply conditions element-wise
        # This correctly sets D=0 where E <= delta
        D_array = np.where(E > delta,
                           self.D_0 * np.sqrt(np.maximum(0.0, arg)),
                           0.0)

        self._diff_array = D_array
        return D_array
        
    def create_thomas_factors(self, alpha: float) -> np.ndarray:
        """
        Pre-compute Thomas algorithm factors for efficiency
        """
        if self._diff_array is None:
            raise ValueError("Must create diffusion array first")
            
        nx = self.geometry.nx
        ne = self._diff_array.shape[1]
        c_prime = np.zeros((nx - 1, ne))
        
        for j in range(ne):
            D_E = self._diff_array[:, j]
            
            # First element
            denom = alpha + D_E[0]
            if abs(denom) < 1e-15:
                logger.warning(f"Near-singular matrix at energy index {j}")
                c_prime[0, j] = 0.0
            else:
                c_prime[0, j] = -D_E[0] / denom
                
            # Remaining elements
            for i in range(1, nx - 1):
                denom = (alpha + D_E[i-1] + D_E[i]) + D_E[i-1] * c_prime[i-1, j]
                if abs(denom) < 1e-15:
                    logger.warning(f"Near-singular matrix at ({i}, {j})")
                    c_prime[i, j] = 0.0
                else:
                    c_prime[i, j] = -D_E[i] / denom
                    
        self._thomas_factors = c_prime
        return c_prime
        
    def diffusion_step(self, n_density: np.ndarray, dt: float) -> None:
        """
        Perform one diffusion step using Crank-Nicolson method
        Updates n_density in place
        
        Parameters:
        -----------
        n_density : np.ndarray
            Dimensionless quasiparticle density n(E,x) with shape (nx, ne)
        dt : float
            Time step (ns)
        """
        nx, ne = n_density.shape
        alpha = 2 * self.geometry.dx ** 2 / dt
        
        if self._thomas_factors is None:
            raise ValueError("Must create Thomas factors first")
            
        # Process each energy slice
        for j in range(ne):
            n_j = n_density[:, j]
            D_j = self._diff_array[:, j]
            c_prime_j = self._thomas_factors[:, j]
            
            # Solve matrix equation A*n(E,t+1) = B*n(E,t)
            d = self._multiply_B_matrix(n_j, D_j, alpha)
            n_density[:, j] = self._thomas_solve(d, D_j, c_prime_j, alpha)
            
    def _multiply_B_matrix(self, n: np.ndarray, D: np.ndarray, alpha: float) -> np.ndarray:
        """Multiply by B matrix for Crank-Nicolson RHS"""
        d = np.zeros_like(n)
        
        # Boundary conditions
        d[0] = (alpha - D[0]) * n[0] + D[0] * n[1]
        d[-1] = D[-1] * n[-2] + (alpha - D[-1]) * n[-1]
        
        # Interior points (vectorized)
        d[1:-1] = D[:-1] * n[:-2] + (alpha - D[:-1] - D[1:]) * n[1:-1] + D[1:] * n[2:]
        
        return d
        
    def _thomas_solve(self, d: np.ndarray, D: np.ndarray, c_prime: np.ndarray, 
                     alpha: float) -> np.ndarray:
        nx = len(d)
        d_prime = np.zeros_like(d)
        x = np.zeros_like(d)
        
        # Forward elimination
        d_prime[0] = d[0] / (alpha + D[0])
        
        for i in range(1, nx-1):
            denom = alpha + D[i] + D[i-1] * (1 + c_prime[i-1])
            d_prime[i] = (d[i] + D[i-1] * d_prime[i-1]) / denom
        
        denom = alpha + D[nx-2] * (1 + c_prime[nx-2])
        d_prime[nx-1] = (d[nx-1] + D[nx-2] * d_prime[nx-2]) / denom
        
        # Back substitution
        x[nx-1] = d_prime[nx-1]
        for i in range(nx-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
            
        return x

# Assume MaterialParameters, SuperconductorGeometry, BCSPhysics, and logger are defined
class ScatteringSolver:
    """
    Handles energy relaxation through scattering and recombination.
    This version is fully vectorized for high performance.
    """

    def __init__(self, material: MaterialParameters, geometry: SuperconductorGeometry,
                 bcs: BCSPhysics, T_bath: float):
        self.material = material
        self.geometry = geometry
        self.bcs = bcs 
        self.T_bath = T_bath
        self._Ks_array = None
        self._Kr_array = None
        self._rho_array = None

    def create_scattering_arrays(self, energies: np.ndarray) -> None:
        """
        Create scattering and recombination kernel arrays using vectorized operations.

        NOTE: This implementation assumes the geometry object can provide the entire
        gap profile as a single array, e.g., `self.geometry.gap_profile`.
        """
        nx = self.geometry.nx
        ne = len(energies)
        k_B = pyc['Boltzmann constant in eV/K'][0]

        # Material constants for the kernels
        Ks_const = 1.0 / (self.material.tau_s * (k_B * self.material.T_c) ** 3)
        Kr_const = 1.0 / (self.material.tau_r * (k_B * self.material.T_c) ** 3)

        # Vectorized Kernel Calculation 

        # 1. Pre-compute energy matrices, shape (ne, ne)
        E_diff = np.subtract.outer(energies, energies)
        E_sum = np.add.outer(energies, energies)
        E_prod = np.multiply.outer(energies, energies)
        E_prod_safe = np.where(np.abs(E_prod) < 1e-15, 1e-15, E_prod)

        # 2. Get gap profile and calculate density of states, shape (nx, ne)
        deltas = self.geometry.gap_at_center 
        self._rho_array = self.bcs.density_of_states(energies[None, :], deltas[:, None])

        # 3. Calculate position-independent Phonon Occupation matrices, shape (ne, ne)
        N_diff = self.bcs.phonon_occupation(E_diff, self.T_bath)
        N_sum = self.bcs.phonon_occupation(E_sum, self.T_bath)


        # 4. Use broadcasting to compute kernels for all spatial points at once
        delta_sq = deltas[:, None, None]**2
        
        coherence_s = 1.0 - delta_sq / E_prod_safe
        self._Ks_array = Ks_const * E_diff**2 * coherence_s * N_diff

        coherence_r = 1.0 + delta_sq / E_prod_safe
        self._Kr_array = Kr_const * E_sum**2 * coherence_r * N_sum

        # 5. Check, report, and correct the final kernel arrays
        for kernel, name in [(self._Ks_array, "Ks"), (self._Kr_array, "Kr")]:
            # Find any negative values using a boolean mask
            neg_mask = kernel < 0
            if np.any(neg_mask):
                # Find the most negative value to report its magnitude
                min_val = np.min(kernel[neg_mask])

                # Decide if the error is significant enough for a major warning
                if min_val < -1e-9:  # Threshold for a significant numerical error
                    logger.warning(
                        f"Significant negative values found in {name} kernel (min = {min_val:.2e}). "
                        "This may indicate a numerical instability. Clipping to zero."
                    )
                else: # Otherwise, it's likely trivial floating point noise
                    logger.info(
                        f"Trivial negative values clipped in {name} kernel (min = {min_val:.2e})."
                    )

                # Correct the array by setting only the negative values to zero
                kernel[neg_mask] = 0.0

        # 6. Set self-scattering to zero after all other checks
        e_idx = np.arange(ne)
        self._Ks_array[:, e_idx, e_idx] = 0.0

    def _calculate_occupation_factors(self, n_density: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of occupation factors f = n/ρ.
        """
        # Use np.divide for safe, vectorized division to get f = n/ρ
        f = np.divide(n_density, self._rho_array,
                      out=np.zeros_like(n_density),
                      where=self._rho_array > 1e-15)

        # Warning if any occupation is high (simulation may be dubious)
        if np.any(f > 0.5):
            logger.warning("""High occupation factors (f > 0.5) detected.
        The simulation assumes a static gap, but high densities can suppress it,
        making the current results potentially unphysical.""")

        # Halt if the occupation factor approaches the physical limit of 1.0
        if np.any(f > 0.99):
            raise ValueError("""Occupation factor is approaching or exceeding 1.
        The physical assumptions of the BCS model are breaking down.
        This may indicate the superconductor is being driven into its normal state.""")
        return f

    def scattering_step(self, n_density: np.ndarray, n_thermal: np.ndarray,
                          dt: float, dE: float) -> np.ndarray:
        """
        Update dimensionless density n due to scattering and recombination.
        """

        f = self._calculate_occupation_factors(n_density)
        pauli = 1.0 - f

        n = n_density[:, :, None]
        n_th = n_thermal[:, :, None]
        rho = self._rho_array
        Ks = self._Ks_array
        Kr = self._Kr_array

        scatter_in = rho[:, :, None] * (Ks.transpose(0, 2, 1) @ n) * pauli[:, :, None]

        scatter_out_rates = np.einsum('xjk,xk,xk->xj', Ks, rho, pauli)
        scatter_out = scatter_out_rates[:, :, None] * n

        recomb_loss = 2 * (Kr @ n) * n
        thermal_gen = 2 * (Kr @ n_th) * n_th

        dn = dt * dE * (scatter_in - scatter_out - recomb_loss + thermal_gen)
        n_new = (n + dn).squeeze(2)

        if np.any(n_new < 0):
            bad = np.argwhere(n_new < 0)[0]
            ix, ie = bad[0], bad[1]
            raise ValueError(f"""Negative density at x-index {ix}, energy-index {ie}. This may be caused by too large of a time step being used.""")

        return n_new

class SystemMonitor:
    """Monitor system diagnostics, physical parameters, and detect unphysical states"""
    
    def __init__(self, dx: float, dE: float):
        """
        Parameters:
        -----------
        dx : float
            Spatial step size (μm)
        dE : float
            Energy step size (eV)
        """
        self.dx = dx
        self.dE = dE
        self.history = {
            'time': [],
            'total_qp': [],
            'total_energy': [],
            'max_density': [],
            'stable': [],
            'max_occupation': [],
            'avg_occupation': []
        }
        
    def check_system(self, n_density: np.ndarray, occupation_factors: np.ndarray,
                    energies: np.ndarray, time: float, N_0: float) -> dict:
        """
        Check system state and calculate metrics
        
        Parameters:
        -----------
        n_density : np.ndarray
            Dimensionless quasiparticle density n(E,x) with shape (nx, ne)
        occupation_factors : np.ndarray
            Pre-calculated occupation factors with shape (nx, ne)
        energies : np.ndarray
            Energy values (eV)
        time : float
            Current simulation time (ns)
        N_0 : float
            Single-spin DOS at Fermi level for unit conversion
            
        Returns:
        --------
        dict
            System metrics including conservation checks
        """
        # Basic quantities - convert to physical units
        total_qp = 4 * N_0 * np.sum(n_density) * self.dx * self.dE
        E_grid = energies.reshape(1, -1)
        total_energy = 4 * N_0 * np.sum(n_density * E_grid) * self.dx * self.dE
        max_density = np.max(n_density)  # Dimensionless
        
        # Occupation statistics from pre-calculated values
        max_occupation = np.max(occupation_factors)
        # Average only over non-zero values to avoid skewing
        nonzero_mask = occupation_factors > 1e-15
        avg_occupation = np.mean(occupation_factors[nonzero_mask]) if np.any(nonzero_mask) else 0.0
        
        # Stability checks
        has_nan = np.any(np.isnan(n_density))
        has_inf = np.any(np.isinf(n_density))

        stable = not (has_nan or has_inf)
        
        # Store history
        self.history['time'].append(time)
        self.history['total_qp'].append(total_qp)
        self.history['total_energy'].append(total_energy)
        self.history['max_density'].append(max_density)
        self.history['stable'].append(stable)
        self.history['max_occupation'].append(max_occupation)
        self.history['avg_occupation'].append(avg_occupation)
        
        # Issue warnings
        if has_nan or has_inf:
            logger.error("NaN or Inf detected in distribution!")
        if max_density > 1e3:  # You can still have a large DOS but small f near the gap edge
            logger.warning(f"Very large density: max(n) = {max_density:.2e} (dimensionless)")
            
        return {
            'total_qp': total_qp,
            'total_energy': total_energy,
            'max_density': max_density,
            'stable': stable,
            'max_occupation': max_occupation,
            'avg_occupation': avg_occupation
        }

# ============================================================================
# Main Simulator Class
# ============================================================================

class QuasiparticleSimulator:
    """
    Main orchestrator for quasiparticle dynamics simulation
    
    Solves the dimensionless reaction-diffusion equation:
    ∂n/∂t = ∇·[D∇n] + I_coll[n]
    
    where n(E,x,t) = ρ(E,x)f(E,x,t) is the dimensionless spectral density.
    """
    
    def __init__(self, material: MaterialParameters, sim_params: SimulationParameters,
                 gap_function: Callable[[float], float], T_bath: float):
        
        # Validate parameters
        material.validate()
        sim_params.validate()
        
        # Store parameters
        self.material = material
        self.sim_params = sim_params
        self.T_bath = T_bath
        
        # Create components
        self.geometry = SuperconductorGeometry(sim_params.L, sim_params.nx, gap_function)
        self.bcs = BCSPhysics(material.gamma, N_0=material.N_0)
        self.diffusion = DiffusionSolver(self.geometry, material.D_0)
        self.scattering = ScatteringSolver(material, self.geometry, self.bcs, T_bath)
        
        # Pass solver parameters to ScatteringSolver for use in implicit solver
        self.scattering.verbose = sim_params.verbose
        self.scattering.rtol = sim_params.rtol
        self.scattering.atol = sim_params.atol
        
        self.monitor = SystemMonitor(sim_params.dx, sim_params.dE)
    
        # Energy grid
        self.energies = self.sim_params.energy_grid
        
        # Initialize arrays (dimensionless density representation)
        self.n_density = None  # Dimensionless QP density n(E,x) = ρ(E,x)f(E,x)
        self.n_thermal = None  # Thermal equilibrium density
        
        # Check CFL condition
        self._check_cfl_condition()
        
    def _check_cfl_condition(self):
        """Check and warn about CFL stability condition"""
        D_max = self.material.D_0
        dx = self.sim_params.dx
        dt = self.sim_params.dt

        cfl_limit = 0.5 * dx ** 2 / D_max

        if dt > cfl_limit:
            warnings.warn(f"""Time step ({dt:.3e} ns) exceeds the CFL stability limit ({cfl_limit:.3e} ns). The Crank-Nicolson algorithm ensures stability, but unphysical oscillations may stilloccur. Consider reducing the time step or increasing spatial resolution.""")
            
    def initialize(self, init_type: str = 'thermal', uniform_density: float = 1e-10):
        """
        Initialization of the quasiparticle distribution.
        """
        nx = self.sim_params.nx
        ne = self.sim_params.ne

        # Get the gap profile and energy grid
        deltas = self.geometry.gap_at_center 
        energies = self.energies # shape (ne,)

        # Use broadcasting to calculate thermal density n = ρ*f for the entire grid
        # energies[None, :] -> shape (1, ne)
        # deltas[:, None]   -> shape (nx, 1)
        rho_grid = self.bcs.density_of_states(energies[None, :], deltas[:, None])
        f_thermal = self.bcs.fermi_dirac(energies[None, :], self.T_bath)
        self.n_thermal = rho_grid * f_thermal

        # Create a mask for states where density can exist (E >= Δ or Dynes is on)
        valid_states_mask = (energies[None, :] >= deltas[:, None]) | (self.material.gamma > 0)
        self.n_thermal *= valid_states_mask

        # Set the initial distribution based on type
        if init_type == 'thermal':
            self.n_density = self.n_thermal.copy()
        elif init_type == 'uniform':
            self.n_density = np.full((nx, ne), uniform_density)
            self.n_density *= valid_states_mask
        else:
            self.n_density = np.zeros((nx, ne))

        # Prepare solver arrays 
        self.diffusion.create_diffusion_array(self.energies)
        alpha = 2 * self.sim_params.dx**2 / self.sim_params.dt
        self.diffusion.create_thomas_factors(alpha)
        self.scattering.create_scattering_arrays(self.energies)

        # --- Logging and info ---
        total_qps = 4 * self.material.N_0 * np.sum(self.n_density) * \
                    self.sim_params.dx * self.sim_params.dE
        logger.info(f"Vectorized initialization with '{init_type}' distribution complete.")
        logger.info(f"Total QPs = {total_qps:.2e}")
        
    def inject_quasiparticles(self, injection, time: float):
        """Inject quasiparticles at a specified location and energy.
        This function converts the provided physical injection rate into the
        internal dimensionless units used by the simulation.
        """
        inject_quasiparticles_safe(self, injection, time)
        if injection.rate <= 0:
            return
            
        # Find closest spatial index
        ix = np.argmin(np.abs(self.geometry.x_centers - injection.location))
        
        # Find closest energy index
        ie = np.argmin(np.abs(self.energies - injection.energy))
        
        # Check if injection is valid at this location
        delta = self.geometry.gap_at_position(self.geometry.x_centers[ix])
        if self.energies[ie] < delta and self.material.gamma == 0:
            logger.warning(f"Cannot inject below gap at x={injection.location:.1f} μm")
            return
            
        # Determine if we should inject
        inject_now = False
        if injection.type == 'continuous':
            inject_now = True
        elif injection.type == 'pulse':
            inject_now = time <= injection.pulse_duration
            
        if inject_now:
            # Convert injection rate to dimensionless density rate
            # injection.rate is in QPs/μm/eV/ns
            # Need to convert to dimensionless: divide by 4N₀
            rho = self.scattering._rho_array[ix, ie]
            if rho > 1e-15:
                dimensionless_rate = injection.rate / (4 * self.material.N_0)
                density_to_add = dimensionless_rate * self.sim_params.dt
                self.n_density[ix, ie] += density_to_add
            
    def step(self):
        """Perform one complete time step"""
        dt = self.sim_params.dt
        dE = self.sim_params.dE
        
        # Diffusion step (updates in place)
        self.diffusion.diffusion_step(self.n_density, dt)
        
        # Scattering step (this method returns a new array)
        self.n_density = self.scattering.scattering_step(self.n_density, self.n_thermal, dt, dE)
        
    def run(self, injection: Optional[InjectionParameters] = None,
            plot_interval: int = 100, save_data: bool = False):
        """Run the complete simulation"""
        
        logger.info("Starting simulation with Pauli blocking...")
        logger.info(f"Using dimensionless density representation: n(x,E)")
        
        for it in range(self.sim_params.nt + 1):
            time = it * self.sim_params.dt
            
            # Monitor conservation with occupation factor diagnostics
            if it % plot_interval == 0:
                # Calculate occupation factors for monitoring
                occupation_factors = self.scattering._calculate_occupation_factors(self.n_density)
                
                metrics = self.monitor.check_system(
                    self.n_density, occupation_factors, self.energies, 
                    time, self.material.N_0
                )
                
                if not metrics['stable']:
                    logger.error("Simulation became unstable!")
                    break
                    
                logger.info(
                    f"t={time:.1f} ns: Total QP={metrics['total_qp']:.2e}, "
                    f"E_total={metrics['total_energy']:.2e} eV, "
                    f"Max n={metrics['max_density']:.2e} (dimensionless), "
                    f"Max occupation f={metrics['max_occupation']:.3f}"
                )
                
                # Plot if needed
                if plot_interval > 0:
                    self.plot_distribution(time, occupation_factors)
                    
            # Save data if requested
            if save_data and it % plot_interval == 0:
                self.save_snapshot(time)
                
            # Stop here on last iteration
            if it == self.sim_params.nt:
                break
                
            # Injection
            if injection:
                self.inject_quasiparticles(injection, time)
                
            # Time evolution
            self.step()
            
        logger.info("Simulation completed!")
        
    def plot_distribution(self, time: float, occupation_factors: np.ndarray = None):
        """Plot the current distribution with occupation factor diagnostics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 2D distribution plot - show log of density
        log_n = np.log10(self.n_density.T[::-1] + 1e-40)
        
        extent = [0, self.sim_params.L, self.sim_params.E_min, self.sim_params.E_max]
        im1 = axes[0,0].imshow(log_n, aspect='auto', extent=extent,
                               vmin=-10, vmax=5, cmap='viridis')
        
        axes[0,0].set_xlabel('Position (μm)')
        axes[0,0].set_ylabel('Energy (eV)')
        axes[0,0].set_title(f'log₁₀(n) at t={time:.1f} ns')
        
        # Gap profile
        gap_profile = self.geometry.gap_array()
        axes[0,0].plot(self.geometry.x_centers, gap_profile, 'w--', 
                       label='Gap Δ(x)', linewidth=2)
        axes[0,0].legend()
        
        cbar1 = plt.colorbar(im1, ax=axes[0,0])
        cbar1.set_label('log₁₀(n) [dimensionless]')
        
        # Integrated density vs position
        n_integrated = np.sum(self.n_density, axis=1) * self.sim_params.dE  # This gives xqp(x)
        # Convert to physical QPs/μm
        n_physical = 4 * self.material.N_0 * n_integrated
        axes[0,1].plot(self.geometry.x_centers, n_physical)
        axes[0,1].set_xlabel('Position (μm)')
        axes[0,1].set_ylabel('QP density (QPs/μm)')
        
        # Total number
        total_qps = 4 * self.material.N_0 * np.sum(self.n_density) * \
                    self.sim_params.dx * self.sim_params.dE
        axes[0,1].set_title(f'Total: {total_qps:.2e} QPs')
        axes[0,1].grid(True, alpha=0.3)
        
        # Occupation factors plot
        if occupation_factors is not None:
            log_f = np.log10(occupation_factors.T[::-1] + 1e-40)
            im2 = axes[1,0].imshow(log_f, aspect='auto', extent=extent,
                                   vmin=-6, vmax=0, cmap='plasma')
            
            axes[1,0].set_xlabel('Position (μm)')
            axes[1,0].set_ylabel('Energy (eV)')
            axes[1,0].set_title(f'log₁₀(occupation f) at t={time:.1f} ns')
            
            # Gap profile
            axes[1,0].plot(self.geometry.x_centers, gap_profile, 'w--', 
                           label='Gap Δ(x)', linewidth=2)
            axes[1,0].legend()
            
            cbar2 = plt.colorbar(im2, ax=axes[1,0])
            cbar2.set_label('log₁₀(occupation factor)')
            
            # Occupation factor statistics
            axes[1,1].plot(self.monitor.history['time'], self.monitor.history['max_occupation'], 
                           'r-', label='Max occupation')
            axes[1,1].plot(self.monitor.history['time'], self.monitor.history['avg_occupation'], 
                           'b-', label='Avg occupation')
            axes[1,1].set_xlabel('Time (ns)')
            axes[1,1].set_ylabel('Occupation factor')
            axes[1,1].set_title('Pauli blocking evolution')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_ylim(0, 1)
        else:
            axes[1,0].text(0.5, 0.5, 'Occupation factors\nnot available', 
                           ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,1].text(0.5, 0.5, 'Occupation statistics\nnot available', 
                           ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        clear_output(wait=True)
        plt.show()
        
    def save_snapshot(self, time: float):
        """Save current state to file including occupation factors"""
        # Calculate occupation factors for saving
        occupation_factors = self.scattering._calculate_occupation_factors(self.n_density)
        
        filename = f"qp_density_pauli_t{time:.1f}ns.npz"
        np.savez(filename,
                 n_density=self.n_density,  # Dimensionless
                 n_thermal=self.n_thermal,  # Dimensionless
                 occupation_factors=occupation_factors,
                 time=time,
                 x_centers=self.geometry.x_centers,
                 energies=self.energies,
                 gap_profile=self.geometry.gap_array(),
                 N_0=self.material.N_0,  # For unit conversion
                 units="dimensionless n(E,x) = ρ(E,x)f(E,x)")
        logger.info(f"Saved snapshot with Pauli blocking data to {filename}")


class PauliViolationError(Exception):
    """
    Raised when injection would violate Pauli exclusion and drive the 
    superconductor into the normal state.
    """
    pass

def inject_quasiparticles_safe(simulator, injection, time: float):
    """
    Safe injection that checks Pauli exclusion before proceeding.
    
    If injection would drive f > 1.0, raises PauliViolationError with 
    detailed physics explanation and suggestions.
    
    Parameters:
    -----------
    simulator : QuasiparticleSimulator
        The simulator instance
    injection : InjectionParameters
        Injection parameters
    time : float
        Current time (ns)
        
    Raises:
    -------
    PauliViolationError
        If injection would violate Pauli exclusion
    """
    if injection.rate <= 0:
        return
        
    # Check if injection is valid at this time
    inject_now = False
    if injection.type == 'continuous':
        inject_now = True
    elif injection.type == 'pulse':
        inject_now = time <= injection.pulse_duration
        
    if not inject_now:
        return
    
    # Find injection cell
    ix = np.argmin(np.abs(simulator.geometry.x_centers - injection.location))
    ie = np.argmin(np.abs(simulator.energies - injection.energy))
    
    # Check if injection energy is above gap
    delta = simulator.geometry.gap_at_position(simulator.geometry.x_centers[ix])
    if simulator.energies[ie] < delta and simulator.material.gamma == 0:
        logger.warning(f"Cannot inject below gap at x={injection.location:.1f} μm")
        return
    
    # Calculate injection amount
    dimensionless_rate = injection.rate / (4 * simulator.material.N_0)
    density_to_add = dimensionless_rate * simulator.sim_params.dt
    
    # Check current state
    current_density = simulator.n_density[ix, ie]
    rho = simulator.scattering._rho_array[ix, ie] if simulator.scattering._rho_array is not None else 1.0
    
    if rho <= 1e-15:
        logger.warning(f"Cannot inject at energy below gap: E={simulator.energies[ie]*1e6:.1f} μeV")
        return
    
    # Calculate target occupation factor
    target_density = current_density + density_to_add
    target_occupation = target_density / rho
    
    # Check Pauli violation
    if target_occupation > 1.0:
        # Calculate available space
        available_density = rho - current_density
        max_injectable_physical = available_density * 4 * simulator.material.N_0 * \
                                 simulator.sim_params.dx * simulator.sim_params.dE
        
        # Calculate what was attempted
        attempted_physical = density_to_add * 4 * simulator.material.N_0 * \
                            simulator.sim_params.dx * simulator.sim_params.dE
        
        raise PauliViolationError(
            f"Injection would violate Pauli exclusion and drive superconductor to normal state!\n\n"
            f"PHYSICS VIOLATION DETAILS:\n"
            f"  Location: x = {injection.location:.1f} μm (cell {ix})\n"
            f"  Energy: E = {injection.energy*1e6:.1f} μeV (cell {ie})\n"
            f"  Current occupation: f = {current_density/rho:.3f}\n"
            f"  Target occupation: f = {target_occupation:.3f} > 1.0 ❌\n\n"
            f"INJECTION ANALYSIS:\n"
            f"  Attempted injection: {attempted_physical:.2e} QPs\n"
            f"  Available space: {max_injectable_physical:.2e} QPs\n"
            f"  Excess: {attempted_physical - max_injectable_physical:.2e} QPs\n"
            f"  Injection efficiency: {max_injectable_physical/attempted_physical*100:.1f}%\n\n"
            f"PHYSICAL CONSEQUENCE:\n"
            f"  f > 1.0 would break Cooper pairs faster than they can form\n"
            f"  → Superconducting gap collapse → Normal state\n"
            f"  → BCS theory no longer valid → Simulation invalid\n\n"
            f"SOLUTIONS:\n"
            f"  1. Reduce injection rate: rate < {injection.rate * max_injectable_physical/attempted_physical:.2e} QPs/μm/eV/ns\n"
            f"  2. Spread injection: Use multiple locations or energies\n"
            f"  3. Lower energy: Inject closer to gap (higher density of states)\n"
            f"  4. Longer pulse: Spread over more time steps\n"
            f"  5. Accept partial injection: {max_injectable_physical:.2e} QPs maximum\n\n"
            f"Use calculate_safe_injection_rate() for automatic rate calculation."
        )
    
    # Safe injection - proceed normally
    simulator.n_density[ix, ie] += density_to_add
    logger.debug(f"Safe injection: added {density_to_add:.2e} (dimensionless) at ({ix},{ie}), f = {target_occupation:.3f}")

def calculate_safe_injection_rate(simulator, injection_energy: float, injection_location: float,
                                 target_qps: float, safety_factor: float = 0.9) -> float:
    """
    Calculate maximum safe injection rate that avoids Pauli violation.
    
    Parameters:
    -----------
    simulator : QuasiparticleSimulator
        The simulator instance
    injection_energy : float
        Injection energy (eV)
    injection_location : float
        Injection location (μm)
    target_qps : float
        Desired number of QPs to inject
    safety_factor : float
        Safety margin (0.9 = stay 10% below Pauli limit)
        
    Returns:
    --------
    safe_rate : float
        Maximum safe injection rate (QPs/μm/eV/ns)
    """
    # Find injection cell
    ix = np.argmin(np.abs(simulator.geometry.x_centers - injection_location))
    ie = np.argmin(np.abs(simulator.energies - injection_energy))
    
    # Get current state
    current_density = simulator.n_density[ix, ie]
    rho = simulator.scattering._rho_array[ix, ie] if simulator.scattering._rho_array is not None else \
          simulator.bcs.density_of_states(injection_energy, simulator.geometry.gap_at_position(simulator.geometry.x_centers[ix]))
    
    if rho <= 1e-15:
        raise ValueError(f"Cannot inject at energy {injection_energy*1e6:.1f} μeV below gap")
    
    # Calculate available space with safety factor
    max_safe_density = rho * safety_factor
    available_density = max_safe_density - current_density
    
    if available_density <= 0:
        raise PauliViolationError(f"Injection cell already at {current_density/rho:.3f} occupation - no space available")
    
    # Convert to physical units
    max_safe_qps = available_density * 4 * simulator.material.N_0 * \
                   simulator.sim_params.dx * simulator.sim_params.dE
    
    # Calculate rate for desired QPs
    if target_qps <= max_safe_qps:
        # Can inject all QPs safely
        safe_rate = target_qps / (simulator.sim_params.dx * simulator.sim_params.dE * simulator.sim_params.dt)
    else:
        # Limited by Pauli exclusion
        safe_rate = max_safe_qps / (simulator.sim_params.dx * simulator.sim_params.dE * simulator.sim_params.dt)
        logger.warning(f"Requested {target_qps:.2e} QPs, but only {max_safe_qps:.2e} QPs can be safely injected")
    
    logger.info(f"Safe injection rate calculated:")
    logger.info(f"  Available space: f = {current_density/rho:.3f} → {max_safe_density/rho:.3f}")
    logger.info(f"  Max safe QPs: {max_safe_qps:.2e}")
    logger.info(f"  Safe rate: {safe_rate:.2e} QPs/μm/eV/ns")
    
    return safe_rate

def validate_injection_parameters(simulator, injection) -> dict:
    """
    Validate injection parameters before running simulation.
    
    Returns detailed analysis of injection feasibility.
    """
    from qp_simulator import InjectionParameters
    
    # Find injection cell
    ix = np.argmin(np.abs(simulator.geometry.x_centers - injection.location))
    ie = np.argmin(np.abs(simulator.energies - injection.energy))
    
    results = {
        'safe': False,
        'current_occupation': 0.0,
        'target_occupation': 0.0,
        'max_safe_qps': 0.0,
        'attempted_qps': 0.0,
        'efficiency': 0.0,
        'recommendations': []
    }
    
    # Check energy vs gap
    delta = simulator.geometry.gap_at_position(simulator.geometry.x_centers[ix])
    if simulator.energies[ie] < delta and simulator.material.gamma == 0:
        results['recommendations'].append(f"Increase energy above gap: E > {delta*1e6:.1f} μeV")
        return results
    
    # Get current state
    current_density = simulator.n_density[ix, ie]
    rho = simulator.scattering._rho_array[ix, ie] if simulator.scattering._rho_array is not None else \
          simulator.bcs.density_of_states(injection.energy, delta)
    
    if rho <= 1e-15:
        results['recommendations'].append("Cannot inject below gap")
        return results
    
    # Calculate injection amount
    dimensionless_rate = injection.rate / (4 * simulator.material.N_0)
    density_to_add = dimensionless_rate * simulator.sim_params.dt
    target_density = current_density + density_to_add
    
    # Calculate occupations
    current_occupation = current_density / rho
    target_occupation = target_density / rho
    
    # Calculate limits
    max_safe_density = rho * 0.9  # 90% of Pauli limit
    available_density = max_safe_density - current_density
    
    max_safe_qps = available_density * 4 * simulator.material.N_0 * \
                   simulator.sim_params.dx * simulator.sim_params.dE
    
    attempted_qps = density_to_add * 4 * simulator.material.N_0 * \
                    simulator.sim_params.dx * simulator.sim_params.dE
    
    results.update({
        'safe': target_occupation < 1.0,
        'current_occupation': current_occupation,
        'target_occupation': target_occupation,
        'max_safe_qps': max_safe_qps,
        'attempted_qps': attempted_qps,
        'efficiency': min(max_safe_qps / attempted_qps, 1.0) if attempted_qps > 0 else 0
    })
    
    # Generate recommendations
    if target_occupation > 1.0:
        results['recommendations'].extend([
            f"Reduce injection rate by factor of {attempted_qps/max_safe_qps:.1f}",
            f"Spread injection over multiple cells",
            f"Use pulse injection over {attempted_qps/max_safe_qps:.1f}× longer duration",
            f"Inject at higher energy (more available states)"
        ])
    elif target_occupation > 0.9:
        results['recommendations'].append("Close to Pauli limit - consider reducing injection")
    else:
        results['recommendations'].append("Injection parameters are safe")
    
    return results

# Enhanced QuasiparticleSimulator method (to replace existing inject_quasiparticles)
def enhanced_inject_quasiparticles(self, injection, time: float, check_pauli: bool = True):
    """
    Enhanced injection with optional Pauli checking.
    
    Parameters:
    -----------
    injection : InjectionParameters
        Injection parameters
    time : float
        Current time (ns)
    check_pauli : bool
        If True, performs Pauli exclusion checking (default: True)
        If False, uses original behavior (for legacy compatibility)
    """
    if check_pauli:
        inject_quasiparticles_safe(self, injection, time)
    else:
        # Original method (for comparison/testing)
        self._inject_quasiparticles_original(injection, time)


# ============================================================================
# End of qp_simulator.py
# ============================================================================
