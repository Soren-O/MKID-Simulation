# Quasiparticle Dynamics Simulator

A Python‑based numerical solver for quasiparticle (QP) dynamics in superconductors, based on the theoretical framework described in the accompanying paper, *“Quasiparticle Density and Dynamics in Superconductors”* (June 2025).

This simulator is designed to be a flexible tool for researchers studying non‑equilibrium phenomena in superconducting devices like MKIDs, qubits, and other cryogenic detectors.

## Features

* **BCS Physics Engine** – Implements the full BCS density of states, with an optional Dynes broadening parameter for modeling lifetime effects.
* **Reaction‑Diffusion Dynamics** – Solves the coupled equations for spatial diffusion (transport) and energy relaxation (collisions).
* **Flexible Configuration** – Supports spatially varying superconducting gap profiles `Δ(x)` and various injection scenarios (pulse or continuous).
* **Built‑in Validation** – Includes a suite of tools to test for physical realism, such as thermal‑equilibrium stability and pulse‑decay characteristics.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/BoltzPhlow.git
   cd qp-simulator
   ```

2. **Install dependencies** (recommended: within a virtual environment)

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the test suite**

   ```bash
   pytest tests/
   ```

## Project Structure

```text
qp-simulator/
├── qp_simulator/              # Installable Python package
│   ├── __init__.py            # Defines public API; may re-export key classes/functions
│   ├── qp_simulator.py        # Core physics engine: geometry, diffusion, reaction kernels, time-stepper
│   ├── qp_validation.py       # Regression tests & analysis helpers
│   └── cli.py                 # Not created yet, console entry point
│
├── pyproject.toml             # Build metadata: name, version, deps, extras, classifiers
├── README.md                  # Project overview, install guide, quick-start commands
├── ROADMAP.md                 # Planned features
├── LICENSE                    # MIT licence text
├── Quasiparticle_Master_Equation.pdf
│
├── examples/                  # Self-contained runnable demos
│   └── example_simulation.py  # Simple example
│
├── tests/                     # Pytest suite (excluded from final wheel)
│   └── test_qp_simulator.py   # Unit & integration tests for solver and validation layers
│
├── notebooks/                 # Interactive tutorials / exploratory analyses
│   └── example_notebook.ipynb # Hands-on walk-through with rich visualisation
│
└── .github/                   # GitHub-specific configuration
    └── workflows/
        └── ci.yml            # CI pipeline: lint → style check → tests → coverage upload
```


## Physics Implementation

The simulator solves the spatially resolved reaction‑diffusion equation for the quasiparticle spectral density $n(E,x,t)$ \[$\text{QPs}/\mathrm{eV}/\mu\text{m}$]:

$$
\frac{\partial n}{\partial t} \;=\; \nabla \!\cdot\! \bigl[D(E,x)\,\nabla n\bigr] \;+\; \mathcal{I}_{\mathrm{coll}}[n].
$$

* **Diffusion term**
  The spatial transport term is solved using a numerically stable Crank–Nicolson scheme. The energy‑dependent diffusion coefficient

  $$
  D(E) \;=\; D_0\sqrt{1 - \frac{\Delta^2}{E^2}}
  $$

  naturally includes quasiparticle trapping for $E < \Delta$.

* **Collision integral**
  $I_{\mathrm{coll}}[n]$ includes QP–phonon scattering, two‑body recombination, and thermal generation. It is derived from a master‑equation formalism and correctly includes Pauli‑blocking factors.

## Known Limitations

* **Explicit time stepping** – The collision integral presently uses an explicit Euler method, necessitating small time steps for stability.
* **Static gap** – The superconducting gap $\Delta(x)$ is fixed in time and does not respond to changes in $n$.
* **Hot phonons** – The phonon bath is held at $T_\text{bath}$ and is not dynamically coupled to the QP population.
* **Others** – See ROADMAP for more plans to improve.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.

2. Create a feature branch:

   ```bash
   git checkout -b feature/your-amazing-feature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add your amazing feature"
   ```

4. Push the branch:

   ```bash
   git push origin feature/your-amazing-feature
   ```

5. Open a pull request.

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=qp_simulator --cov=qp_validation
```

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## References

* Martinis *et al.*, “Energy Decay in Josephson Qubits from Non‑equilibrium Quasiparticles” (2009), arXiv:0904.2171.
* Lenander *et al.*, “Measurement of energy decay in superconducting qubits from nonequilibrium quasiparticles” (2011), DOI:10.1103/PhysRevB.84.024501.
* *Soren Ormseth*: “(Unpublished, the PDF in this repo: Quasiparticle Density and Dynamics in Superconductors” (2025).

## Authors & Acknowledgments

**Soren Ormseth** – Initial implementation.

Special thanks to Dr. Dave Harrison for providing the initial seed code; my advisor Dr. Peter Timbie for his guidance; Dr. Robert McDermott for his support; and Drs. Thomas Stevenson, Emily Barrentine, and Cary Volpert for their reviews. This work was supported by a NASA grant.

```python

```
