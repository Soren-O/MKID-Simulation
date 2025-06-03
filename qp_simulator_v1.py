import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from scipy.constants import physical_constants as pyc
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    N_0: float = 1.7e10  # Single-spin DOS at Fermi level (eV^-1 μm^-3)
    
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

@dataclass
class InjectionParameters:
    """Quasiparticle injection parameters"""
    location: float      # Injection location (μm)
    energy: float        # Injection energy (eV)
    rate: float          # Injection rate density (QPs/μm/eV/ns)
    type: str = 'continuous'  # 'continuous' or 'pulse'
    pulse_duration: float = 0.0  # Pulse duration for pulse type (ns)

# ============================================================================
# Core Physics Classes
# ============================================================================

class SuperconductorGeometry:
    """Handles spatial discretization and gap profile"""
    
    def __init__(self, L: float, nx: int, gap_function: Callable[[float], float]):
        self.L = L
        self.nx = nx
        self.dx = L / nx
        self.gap_function = gap_function
        self.x_centers = (np.arange(nx) + 0.5) * self.dx
        self.x_boundaries = (np.arange(nx + 1)) * self.dx
        
    def gap_at_position(self, x: float) -> float:
        """Get gap value at position x"""
        return self.gap_function(x)
        
    def gap_array(self) -> np.ndarray:
        """Get gap values at all cell centers"""
        return np.array([self.gap_function(x) for x in self.x_centers])

class BCSPhysics:
    """BCS theory calculations with Dynes broadening"""
    
    def __init__(self, gamma: float = 0.0, N_0: float = 1.7e10):
        """
        Initialize BCS physics calculator
        
        Parameters:
        -----------
        gamma : float
            Dynes broadening parameter (eV)
        N_0 : float
            Single-spin normal state density of states at Fermi level (eV^-1 μm^-3)
            Default value for aluminum: 1.7e10 eV^-1 μm^-3
        """
        self.gamma = gamma
        self.N_0 = N_0  # Single-spin DOS
        
    def density_of_states(self, E: float, delta: float) -> float:
        """
        Calculate normalized superconducting density of states
        with Dynes broadening
        """
        if E < delta and self.gamma == 0:
            return 0.0
            
        # Use complex arithmetic for Dynes broadening
        E_complex = complex(E, -self.gamma)
        
        # Avoid numerical issues
        delta_sq = delta * delta
        arg = E_complex * E_complex - delta_sq
        
        # Handle branch cut properly
        if arg == 0:
            return 1e10  # Large but finite value
            
        sqrt_arg = np.sqrt(arg)
        
        # Ensure we're on the correct branch
        if sqrt_arg.real < 0:
            sqrt_arg = -sqrt_arg
            
        result = np.real(E_complex / sqrt_arg)
        
        # Ensure non-negative
        return max(0.0, result)
        
    def fermi_dirac(self, E: float, T: float) -> float:
        """Fermi-Dirac distribution"""
        k_B = pyc['Boltzmann constant in eV/K'][0]
        
        if T == 0:
            return 0.0
            
        arg = E / (k_B * T)
        
        # Avoid overflow
        if arg > 100:
            return 0.0
        elif arg < -100:
            return 1.0
            
        return 1.0 / (np.exp(arg) + 1.0)
        
    def bose_einstein(self, E: float, T: float) -> float:
        """Bose-Einstein distribution for phonons"""
        if E <= 0:
            return 0.0
            
        k_B = pyc['Boltzmann constant in eV/K'][0]
        
        if T == 0:
            return 0.0
            
        arg = E / (k_B * T)
        
        # Avoid numerical issues
        if arg > 100:
            return 0.0
        elif arg < 0.01:
            # Use Taylor expansion for small arguments
            return k_B * T / E
            
        return 1.0 / (np.exp(arg) - 1.0)

class DiffusionSolver:
    """Crank-Nicolson solver for diffusion equation"""
    
    def __init__(self, geometry: SuperconductorGeometry, D_0: float):
        self.geometry = geometry
        self.D_0 = D_0
        self._thomas_factors = None
        self._diff_array = None
        
    def create_diffusion_array(self, energies: np.ndarray) -> np.ndarray:
        """
        Create diffusion coefficient array D[boundary_idx, energy_idx]
        """
        nx = self.geometry.nx
        ne = len(energies)
        D_array = np.zeros((nx - 1, ne))
        
        # Boundary positions
        for i in range(nx - 1):
            x_boundary = (i + 1) * self.geometry.dx
            delta = self.geometry.gap_at_position(x_boundary)
            
            for j, E in enumerate(energies):
                if E <= delta:
                    D_array[i, j] = 0.0
                else:
                    # Avoid numerical issues with sqrt
                    arg = 1.0 - (delta / E) ** 2
                    D_array[i, j] = self.D_0 * np.sqrt(max(0.0, arg))
                    
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
            Quasiparticle density [QPs/eV/μm] with shape (nx, ne)
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
            
            # Compute RHS = B * n_j
            d = self._multiply_B_matrix(n_j, D_j, alpha)
            
            # Solve A * n_new = d using Thomas algorithm
            n_density[:, j] = self._thomas_solve(d, D_j, c_prime_j, alpha)
            
    def _multiply_B_matrix(self, n: np.ndarray, D: np.ndarray, alpha: float) -> np.ndarray:
        """Multiply by B matrix for Crank-Nicolson RHS"""
        nx = len(n)
        d = np.zeros_like(n)
        
        # First row (boundary condition)
        d[0] = (alpha - D[0]) * n[0] + D[0] * n[1]
        
        # Interior rows
        for i in range(1, nx - 1):
            d[i] = D[i-1] * n[i-1] + (alpha - D[i-1] - D[i]) * n[i] + D[i] * n[i+1]
            
        # Last row (boundary condition)
        d[nx-1] = D[nx-2] * n[nx-2] + (alpha - D[nx-2]) * n[nx-1]
        
        return d
        
    def _thomas_solve(self, d: np.ndarray, D: np.ndarray, c_prime: np.ndarray, 
                      alpha: float) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm"""
        nx = len(d)
        d_prime = np.zeros_like(d)
        x = np.zeros_like(d)
        
        # Forward elimination
        denom = alpha + D[0]
        if abs(denom) < 1e-15:
            d_prime[0] = 0.0
        else:
            d_prime[0] = d[0] / denom
            
        for i in range(1, nx):
            if i < nx - 1:
                b_coeff = alpha + D[i-1] + D[i]
            else:
                b_coeff = alpha + D[i-1]
                
            a_coeff = -D[i-1]
            
            if i < nx - 1:
                denom = b_coeff - a_coeff * c_prime[i-1]
            else:
                denom = b_coeff - a_coeff * c_prime[i-2]
                
            if abs(denom) < 1e-15:
                d_prime[i] = 0.0
            else:
                d_prime[i] = (d[i] - a_coeff * d_prime[i-1]) / denom
                
        # Back substitution
        x[nx-1] = d_prime[nx-1]
        for i in range(nx-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
            
        return x

class ScatteringSolver:
    """Handles energy relaxation through scattering and recombination"""
    
    def __init__(self, material: MaterialParameters, geometry: SuperconductorGeometry,
                 bcs: BCSPhysics, T_bath: float):
        self.material = material
        self.geometry = geometry
        self.bcs = bcs
        self.T_bath = T_bath
        self._Gs_array = None
        self._Gr_array = None
        
    def create_scattering_arrays(self, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create scattering and recombination rate arrays for density representation
        Returns (Gs, Gr) where each has shape [nx, ne, ne]
        
        Units:
        - Gs[i,j,k]: [1/ns] - rate from energy j to k
        - Gr[i,j,k]: [μm·eV/QPs/ns] - recombination coefficient
        """
        nx = self.geometry.nx
        ne = len(energies)
        dE = energies[1] - energies[0]  # Energy bin width
        k_B = pyc['Boltzmann constant in eV/K'][0]
        
        # Material constants for the kernels K^s and K^r
        # K^s and K^r have units [1/eV²/ns]
        Ks_const = 1.0 / (self.material.tau_s * (k_B * self.material.T_c) ** 3)
        Kr_const = 1.0 / (self.material.tau_r * (k_B * self.material.T_c) ** 3)
        
        Gs = np.zeros((nx, ne, ne))
        Gr = np.zeros((nx, ne, ne))
        
        # Pre-compute energy matrices
        E_diff = np.subtract.outer(energies, energies)
        E_sum = np.add.outer(energies, energies)
        E_prod = np.multiply.outer(energies, energies)
        
        # Avoid division by zero
        E_prod_safe = np.where(E_prod == 0, 1e-30, E_prod)
        
        for ix in range(nx):
            x = self.geometry.x_centers[ix]
            delta = self.geometry.gap_at_position(x)
            delta_sq = delta * delta
            
            # Density of states at this position
            rho_vec = np.array([self.bcs.density_of_states(E, delta) for E in energies])
            
            # Phonon occupation numbers
            N_diff = np.array([[self.bcs.bose_einstein(abs(E_diff[i, j]), self.T_bath) 
                               for j in range(ne)] for i in range(ne)])
            N_sum = np.array([[self.bcs.bose_einstein(E_sum[i, j], self.T_bath) 
                              for j in range(ne)] for i in range(ne)])
            
            # Scattering matrix: Gs[j,k] = K^s[j,k] * rho[k] * [1-f[k]] * ΔE
            # For now, assuming [1-f] ≈ 1 (low density approximation)
            coherence_s = 1.0 - delta_sq / E_prod_safe
            Gs[ix] = Ks_const * E_diff ** 2 * coherence_s * N_diff * rho_vec * dE
            
            # Recombination matrix: Gr[j,k] = K^r[j,k] * ΔE
            coherence_r = 1.0 + delta_sq / E_prod_safe
            Gr[ix] = Kr_const * E_sum ** 2 * coherence_r * N_sum * dE
            
            # Ensure non-negative and set diagonal to zero for scattering
            Gs[ix] = np.maximum(0, Gs[ix])
            Gr[ix] = np.maximum(0, Gr[ix])
            np.fill_diagonal(Gs[ix], 0)
            
        self._Gs_array = Gs
        self._Gr_array = Gr
        
        return Gs, Gr
        
    def scattering_step(self, n_density: np.ndarray, n_thermal: np.ndarray, dt: float) -> None:
        """
        Update density n (QPs/eV/μm) due to scattering and recombination
        Includes thermal generation term
        """
        nx, ne = n_density.shape
        
        for ix in range(nx):
            n = n_density[ix, :].reshape(-1, 1)
            n_th = n_thermal[ix, :].reshape(-1, 1)
            
            Gs = self._Gs_array[ix]
            Gr = self._Gr_array[ix]
            
            # Scattering terms
            scatter_in = Gs.T @ n
            scatter_out = np.sum(Gs, axis=1, keepdims=True) * n
            
            # Recombination and generation terms
            recomb_loss = 2 * (Gr @ n) * n
            thermal_gen = 2 * (Gr @ n_th) * n_th
            
            # Update
            dn = dt * (scatter_in - scatter_out - recomb_loss + thermal_gen)
            n_density[ix, :] = np.maximum(0, (n + dn).flatten())

class ConservationMonitor:
    """Monitor conservation laws and numerical stability"""
    
    def __init__(self, dx: float, dE: float):
        """
        Initialize conservation monitor
        
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
            'numerical_error': []
        }
        
    def check_conservation(self, n_density: np.ndarray, energies: np.ndarray, 
                          time: float, injection_rate: float = 0) -> dict:
        """
        Check conservation and stability metrics
        
        Parameters:
        -----------
        n_density : np.ndarray
            Quasiparticle density [QPs/eV/μm] with shape (nx, ne)
        energies : np.ndarray
            Energy values (eV)
        time : float
            Current simulation time (ns)
        injection_rate : float
            Injection rate (QPs/μm/eV/ns)
        """
        
        # Total quasiparticle number = ∫∫ n(x,E) dx dE
        # Discrete: Σᵢⱼ n[i,j] × Δx × ΔE
        total_qp = np.sum(n_density) * self.dx * self.dE
        
        # Total energy = ∫∫ E × n(x,E) dx dE
        E_grid = energies.reshape(1, -1)
        total_energy = np.sum(n_density * E_grid) * self.dx * self.dE
        
        # Maximum density (for stability check)
        max_density = np.max(n_density)
        
        # Check for NaN or Inf
        has_nan = np.any(np.isnan(n_density))
        has_inf = np.any(np.isinf(n_density))
        
        # Numerical error estimate (change rate without physics)
        numerical_error = 0.0
        if len(self.history['total_qp']) > 0 and injection_rate == 0:
            dt = time - self.history['time'][-1]
            if dt > 0:
                dN_dt = (total_qp - self.history['total_qp'][-1]) / dt
                # In equilibrium, this should approach zero
                numerical_error = abs(dN_dt)
                
        # Store history
        self.history['time'].append(time)
        self.history['total_qp'].append(total_qp)
        self.history['total_energy'].append(total_energy)
        self.history['max_density'].append(max_density)
        self.history['numerical_error'].append(numerical_error)
        
        # Warnings
        if has_nan or has_inf:
            logger.error("NaN or Inf detected in distribution!")
        if max_density > 1e15:  # Reasonable limit for density in QPs/eV/μm
            logger.warning(f"Very large density detected: max(n) = {max_density:.2e} QPs/eV/μm")
            
        return {
            'total_qp': total_qp,
            'total_energy': total_energy,
            'max_density': max_density,
            'stable': not (has_nan or has_inf)
        }

# ============================================================================
# Main Simulator Class
# ============================================================================

class QuasiparticleSimulator:
    """Main orchestrator for quasiparticle dynamics simulation"""
    
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
        self.monitor = ConservationMonitor(sim_params.dx, sim_params.dE)
        
        # Energy grid
        self.energies = sim_params.E_min + (np.arange(sim_params.ne) + 0.5) * sim_params.dE
        
        # Initialize arrays (density representation)
        self.n_density = None  # QP density [QPs/eV/μm]
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
            warnings.warn(
                f"Time step ({dt:.3e} ns) exceeds CFL limit ({cfl_limit:.3e} ns). "
                "Simulation may be unstable. Consider reducing time step or increasing spatial resolution."
            )
            
    def initialize(self, init_type: str = 'thermal', uniform_density: float = 1e-10):
        """
        Initialize quasiparticle distribution
        
        Parameters:
        -----------
        init_type : str
            'thermal' for Fermi-Dirac or 'uniform' for constant density
        uniform_density : float
            Density value for uniform initialization [QPs/eV/μm]
        """
        nx = self.sim_params.nx
        ne = self.sim_params.ne
        
        self.n_density = np.zeros((nx, ne))
        self.n_thermal = np.zeros((nx, ne))
        
        # Calculate thermal distribution
        for ix in range(nx):
            x = self.geometry.x_centers[ix]
            delta = self.geometry.gap_at_position(x)
            
            for ie, E in enumerate(self.energies):
                # Thermal density = ρ(E) × f_FD(E) × 2N(0)
                # Note: ρ is normalized, so multiply by 2N(0) to get actual DOS
                if E >= delta or self.material.gamma > 0:
                    rho = self.bcs.density_of_states(E, delta)
                    f_thermal = self.bcs.fermi_dirac(E, self.T_bath)
                    # Density = DOS × occupation × 2 (for two spins)
                    self.n_thermal[ix, ie] = 2 * self.bcs.N_0 * rho * f_thermal
                    
                # Initial distribution
                if init_type == 'thermal':
                    self.n_density[ix, ie] = self.n_thermal[ix, ie]
                elif init_type == 'uniform':
                    if E >= delta or self.material.gamma > 0:
                        self.n_density[ix, ie] = uniform_density
                        
        # Prepare solver arrays
        self.diffusion.create_diffusion_array(self.energies)
        alpha = 2 * self.sim_params.dx ** 2 / self.sim_params.dt
        self.diffusion.create_thomas_factors(alpha)
        self.scattering.create_scattering_arrays(self.energies)
        
        logger.info(f"Initialized with {init_type} distribution")
        logger.info(f"Total QPs = {np.sum(self.n_density) * self.sim_params.dx * self.sim_params.dE:.2e}")
        
    def inject_quasiparticles(self, injection: InjectionParameters, time: float):
        """
        Inject quasiparticles at specified location and energy
        
        Injection rate is in units of QPs/μm/eV/ns (density rate)
        """
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
            # Add density at the injection point
            density_to_add = injection.rate * self.sim_params.dt
            self.n_density[ix, ie] += density_to_add
            
    def step(self):
        """Perform one complete time step"""
        dt = self.sim_params.dt
        
        # Diffusion step (updates in place)
        self.diffusion.diffusion_step(self.n_density, dt)
        
        # Scattering step (updates in place)
        self.scattering.scattering_step(self.n_density, self.n_thermal, dt)
        
    def run(self, injection: Optional[InjectionParameters] = None,
            plot_interval: int = 100, save_data: bool = False):
        """Run the complete simulation"""
        
        logger.info("Starting simulation...")
        logger.info(f"Using density representation: n(x,E) in units of QPs/eV/μm")
        
        for it in range(self.sim_params.nt + 1):
            time = it * self.sim_params.dt
            
            # Monitor conservation
            if it % plot_interval == 0:
                metrics = self.monitor.check_conservation(
                    self.n_density, self.energies, time, 
                    injection.rate if injection else 0
                )
                
                if not metrics['stable']:
                    logger.error("Simulation became unstable!")
                    break
                    
                logger.info(
                    f"t={time:.1f} ns: Total QP={metrics['total_qp']:.2e}, "
                    f"E_total={metrics['total_energy']:.2e} eV, "
                    f"Max density={metrics['max_density']:.2e} QPs/eV/μm"
                )
                
                # Plot if needed
                if plot_interval > 0:
                    self.plot_distribution(time)
                    
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
        
    def plot_distribution(self, time: float):
        """Plot the current distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D distribution plot - show log of density
        log_n = np.log10(self.n_density.T[::-1] + 1e-40)
        
        extent = [0, self.sim_params.L, self.sim_params.E_min, self.sim_params.E_max]
        im = ax1.imshow(log_n, aspect='auto', extent=extent,
                       vmin=-10, vmax=5, cmap='viridis')
        
        ax1.set_xlabel('Position (μm)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'log₁₀(n) at t={time:.1f} ns')
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('log₁₀(QP density) [QPs/eV/μm]')
        
        # Gap profile
        gap_profile = self.geometry.gap_array()
        ax1.plot(self.geometry.x_centers, gap_profile, 'w--', 
                label='Gap Δ(x)', linewidth=2)
        ax1.legend()
        
        # Integrated density vs position
        # n_total(x) = ∫ n(x,E) dE
        n_integrated = np.sum(self.n_density, axis=1) * self.sim_params.dE  # QPs/μm
        ax2.plot(self.geometry.x_centers, n_integrated)
        ax2.set_xlabel('Position (μm)')
        ax2.set_ylabel('Integrated QP density (QPs/μm)')
        
        # Total number
        total_qps = np.sum(self.n_density) * self.sim_params.dx * self.sim_params.dE
        ax2.set_title(f'Total: {total_qps:.2e} QPs')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        clear_output(wait=True)
        plt.show()
        
    def save_snapshot(self, time: float):
        """Save current state to file"""
        filename = f"qp_density_t{time:.1f}ns.npz"
        np.savez(filename,
                 n_density=self.n_density,
                 n_thermal=self.n_thermal,
                 time=time,
                 x_centers=self.geometry.x_centers,
                 energies=self.energies,
                 gap_profile=self.geometry.gap_array(),
                 units="QPs/eV/μm")
        logger.info(f"Saved snapshot to {filename}")

# ============================================================================
# Example Usage Functions
# ============================================================================

def constant_gap(x: float) -> float:
    """Constant gap profile"""
    return 1.7e-4  # 0.17 meV

def step_gap(x: float, L: float) -> float:
    """Step function gap profile"""
    return 1.7e-4 if x < L/2 else 2.5e-4

def well_gap(x: float, L: float) -> float:
    """Potential well gap profile"""
    if 0.3*L <= x <= 0.7*L:
        return 1.0e-4  # Lower gap in center
    else:
        return 2.0e-4  # Higher gap at edges