# Quantum Molecular Simulation Project

## Overview
This project provides a framework for simulating quantum molecular systems using the Variational Quantum Eigensolver (VQE). It allows for molecule initialization, quantum ansatz preparation, optimization of molecular geometries, and visualization of results.

## Environment Setup
This project is developed and tested with Python 3.12. Follow the steps below to set up your environment:

### Prerequisites
Ensure you have the following installed:
- **Python 3.12**
- **Git**

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/a1eb4th/Quantum_Simulations
   cd Quantum_Simulations
   ```
2. Create a virtual environment using Python 3.12:
   ```bash
   python -m venv env
   ```
3. Activate the virtual environment:
   - **Windows:**
     ```bash
     env\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source env/bin/activate
     ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Verify the installation:
   ```bash
   python --version
   ```

### Running the Project
To execute a simulation:
1. Define your molecule in `molecules.json` or pass it via command-line arguments.
2. Run the main script with desired parameters:
   ```bash
   python main.py --molecule H2 --optimizer Adam --stepsize 0.1
   ```
3. Results will be saved in the `temp_results_autograd/` directory.

## Project Structure
```plaintext
quantum_simulation_project/
├── config/                     # Configuration files
├── modules/                    # Core functionalities
│   ├── ansatz_preparer.py      # Quantum ansatz preparation
│   ├── hamiltonian_builder.py  # Molecular Hamiltonian construction
│   ├── molecule_manager.py     # Molecule initialization
│   ├── opt_mol.py              # Main optimization workflow
│   ├── optimizer.py            # Optimization logic
│   ├── visualizer.py           # Visualization utilities
├── temp_results_autograd/      # Temporary results
├── main.py                     # Entry point for simulations
├── requirements.txt            # Dependencies
```

## Features
- **Molecule Initialization**: Define molecules with symbols, coordinates, charge, and spin multiplicity.
- **Hamiltonian Construction**: Build molecular Hamiltonians based on quantum chemistry methods.
- **Quantum Ansatz**: Supports `uccsd` and hardware-efficient ansatz types.
- **Optimization**: Uses various optimizers like Adam, RMSProp, and Gradient Descent.
- **Visualization**: Generate energy evolution plots and final geometries.

## Inputs
Supported command-line arguments:
- `--molecule`: Name of the molecule(s) to simulate (e.g., H2O, NH3, H2, LiH).
- `--optimizer`: Optimization algorithm (e.g., Adam, RMSProp).
- `--stepsize`: Step size for the optimizer.
- `--basis_set`: Currently fixed to `sto-3g`.

## Outputs
Generated files include:
- **Energy Plots**: `energy_evolution_linear.png`, `energy_evolution_log_offset.png`
- **Optimization Reports**: `profile_output_autograd.txt`, `filtered_report_autograd.txt`
- **Final Geometry Visualization**: `final_geometries_3D.png`

## License
This project is licensed under the MIT License.
