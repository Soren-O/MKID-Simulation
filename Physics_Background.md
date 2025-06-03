# Physics Background for MKID Quasiparticle Dynamics Simulator

This document outlines the core physics principles and mathematical formulations used in the MKID Quasiparticle Dynamics Simulator. The simulator models the spatiotemporal evolution of quasiparticles in a one-dimensional superconductor.

## 1. Fundamental Concepts

* **Primary Variable:** The simulation tracks $n(x,E,t)$, the quasiparticle (QP) density.
    * **Units:** QPs / (eV $\cdot$ µm). "QP" is treated as a dimensionless quantity representing a number.
    * This means $n(x,E,t)$ is the number of quasiparticles per unit energy interval per unit length along the 1D superconductor.
* **BCS Framework:** The underlying physics is based on the Bardeen-Cooper-Schrieffer (BCS) theory of superconductivity. This theory predicts the formation of an energy gap $\Delta$ at the Fermi level.
* **Spatially Varying Energy Gap $\Delta(x)$:** The simulator allows the superconducting energy gap $\Delta$ to be a function of position $x$. This is crucial for modeling inhomogeneous materials or engineered device structures.
* **Dynes Broadening:** To account for lifetime-limiting effects and allow for a non-zero density of states (DOS) within the energy gap (as often observed experimentally), Dynes broadening is incorporated. The normalized superconducting DOS, $\rho(E, \Gamma, \Delta)$, is calculated using a complex energy $E - i\Gamma$:
    $$ \rho(E, \Gamma, \Delta) = \text{Re}\left\{\frac{E-i\Gamma}{\sqrt{(E-i\Gamma)^2 - \Delta^2}}\right\} $$
    where $\Gamma$ is the Dynes broadening parameter (eV). If $\Gamma=0$ and $E < \Delta$, $\rho(E)=0$.
* **Thermal Equilibrium Density $n_{thermal}(x,E)$:** The baseline quasiparticle density in thermal equilibrium at a bath temperature $T_{bath}$ is given by:
    $$ n_{thermal}(x,E) = 2 N_0 \rho(E, \Gamma, \Delta(x)) f_{FD}(E, T_{bath}) $$
    where:
    * $N_0$ is the single-spin normal state density of states at the Fermi level (units: $\text{eV}^{-1} \mu\text{m}^{-3}$, effectively $1/(\text{eV} \cdot \mu\text{m}^3)$ as QP is dimensionless). The factor of 2 accounts for both spin directions.
    * $f_{FD}(E, T_{bath}) = (e^{E/(k_B T_{bath})} + 1)^{-1}$ is the Fermi-Dirac distribution.

## 2. Simulated Physical Processes

The evolution of $n(x,E,t)$ is governed by diffusion, scattering, and recombination/generation processes.

### 2.1. Diffusion

* **Equation:** Quasiparticles diffuse spatially according to the 1D diffusion equation:
    $$ \frac{\partial n(x,E,t)}{\partial t} = \frac{\partial}{\partial x} \left[ D(x,E) \frac{\partial n(x,E,t)}{\partial x} \right] $$
* **Diffusion Coefficient $D(x,E)$:** The diffusion coefficient is energy-dependent and varies with position through the local energy gap $\Delta(x)$:
    $$ D(E,x) = D_0 \sqrt{1 - \left(\frac{\Delta(x)}{E}\right)^2} \quad \text{for } E > \Delta(x) $$
    For $E \le \Delta(x)$, $D(E,x) = 0$. $D_0$ is the normal-state diffusion coefficient (units: $\mu\text{m}^2/\text{ns}$).
* **Numerical Method:** The diffusion equation is solved numerically using the Crank-Nicolson method, which is implicit and unconditionally stable. The resulting tridiagonal system of linear equations for each energy slice is solved using the Thomas algorithm.
* **Boundary Conditions:** The simulation currently implements insulated (zero-flux) boundary conditions at the ends of the 1D spatial domain.

### 2.2. Scattering, Recombination, and Generation

These processes describe how quasiparticles change their energy or are created/destroyed. The rate equation for $n_k \equiv n(x, E_k, t)$ at a fixed position $x$ is:
$$ \frac{dn_k}{dt} = \left(\frac{dn_k}{dt}\right)_{\text{scatter-in}} - \left(\frac{dn_k}{dt}\right)_{\text{scatter-out}} - \left(\frac{dn_k}{dt}\right)_{\text{recomb-loss}} + \left(\frac{dn_k}{dt}\right)_{\text{thermal-gen}} $$

* **Scattering Matrix ($\mathcal{G}^s_{jk}(x)$):**
    * This represents the transition rate probability per unit time (units: $\text{ns}^{-1}$) for a quasiparticle at position $x$ to scatter from an initial energy $E_j$ to a final energy $E_k$.
    * It is calculated as:
        $$ \mathcal{G}^s_{jk}(x) = \frac{1}{\tau_s (k_B T_c)^3} (E_j - E_k)^2 \left(1-\frac{\Delta(x)^2}{E_j E_k}\right) N_p(|E_j - E_k|, T_{bath}) \rho(E_k, \Gamma, \Delta(x)) \Delta E_k $$
        where $\tau_s$ is the characteristic electron-phonon scattering time, $T_c$ is the critical temperature, $N_p$ is the Bose-Einstein phonon occupation factor, and $\Delta E_k$ is the width of the energy bin for state $k$.
    * **Pauli Blocking:** The current calculation of $\mathcal{G}^s_{jk}$ implicitly assumes the final state $E_k$ is mostly empty (i.e., the $(1-f_k)$ factor is approximated as 1).
    * **Scattering Terms:**
        * Scatter-in to $E_k$: $\left(\frac{dn_k}{dt}\right)_{\text{scatter-in}} = \sum_j \mathcal{G}^s_{jk} n_j$
        * Scatter-out from $E_k$: $\left(\frac{dn_k}{dt}\right)_{\text{scatter-out}} = n_k \sum_j \mathcal{G}^s_{kj}$

* **Recombination Matrix ($\mathcal{G}^r_{jk}(x)$):**
    * This matrix element (units: $\text{ns}^{-1}$) is part of the term describing the recombination of two quasiparticles (one at $E_j$, one at $E_k$) into a Cooper pair.
    * It is calculated as:
        $$ \mathcal{G}^r_{jk}(x) = \frac{1}{\tau_r (k_B T_c)^3} (E_j + E_k)^2 \left(1+\frac{\Delta(x)^2}{E_j E_k}\right) N_p(E_j + E_k, T_{bath}) \Delta E_k $$
        where $\tau_r$ is the characteristic recombination time. (Note: Symmetrically, $\mathcal{G}^r_{kj}$ involves $\Delta E_j$).
    * **Recombination Loss Term:** The rate of loss of quasiparticles at $E_k$ due to recombination with quasiparticles at any $E_j$ is:
        $$ \left(\frac{dn_k}{dt}\right)_{\text{recomb-loss}} = 2 n_k \sum_j \mathcal{G}^r_{kj} (n_j \Delta E_j \Delta x_j) $$
        Here, $(n_j \Delta E_j \Delta x_j)$ represents the dimensionless *number* of quasiparticles in the cell $(x, E_j)$, acting as partners for recombination. This formulation ensures dimensional consistency.

* **Thermal Generation Term:**
    * This term balances the recombination loss at thermal equilibrium. It is formulated analogously to the recombination loss, using the thermal equilibrium densities $n_{thermal}$:
        $$ \left(\frac{dn_k}{dt}\right)_{\text{thermal-gen}} = 2 n_{thermal,k} \sum_j \mathcal{G}^r_{kj} (n_{thermal,j} \Delta E_j \Delta x_j) $$

This physics background allows for the simulation of how an initial quasiparticle distribution relaxes towards equilibrium or responds to external stimuli like injection.