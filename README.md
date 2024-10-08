
---

# Quantum Simulation Project

## Overview

The **Quantum Simulation Project** is a suite of Python scripts designed to perform quantum simulations of molecules and chemical reactions using the Variational Quantum Eigensolver (VQE) algorithm. Leveraging the power of [PennyLane](https://pennylane.ai/), [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/), and other scientific libraries, this project enables users to:

- **Simulate individual molecules** to find their ground state energies.
- **Simulate chemical reactions** by calculating the reaction energy based on reactants and products.
- **Visualize molecular structures** and **energy convergence** during simulations.
- **Automate testing** of simulations with various configurations to ensure reliability and robustness.

## Features

### `quantum_simulation.py`

- **Molecular Simulation:** Perform quantum simulations of molecules using VQE.
- **Chemical Reaction Simulation:** Calculate reaction energies by simulating reactants and products.
- **Visualization:** Visualize molecular structures and energy convergence plots.
- **Result Management:** Save and load simulation results for future reference or reuse.
- **Error Handling:** Gracefully handle undefined molecules and invalid optimizer choices.
- **User Interaction:** Easily add new molecules via command-line prompts.

### `test_quantum_simulation.py`

- **Automated Testing:** Run a series of predefined tests to validate the functionality of `quantum_simulation.py`.
- **Test Configurations:** Execute both fast and full test suites with various simulation parameters.
- **Error Testing:** Ensure proper handling of errors like undefined molecules and invalid optimizers.
- **Result Verification:** Check if results are correctly saved and loaded.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from the [official website](https://www.python.org/downloads/).

- **Git**: Required for version control and retrieving commit hashes. Download from [Git Downloads](https://git-scm.com/downloads).

### Clone the Repository

```bash
git clone https://github.com/your_username/quantum_simulation_project.git
cd quantum_simulation_project
```

### Install Dependencies

Use `pip` to install the necessary Python packages:

```bash
pip install pennylane matplotlib tqdm pandas ase
```

Additionally, for animation features, install **ImageMagick**:

- **Ubuntu/Debian:**

  ```bash
  sudo apt-get install imagemagick
  ```

- **macOS (using Homebrew):**

  ```bash
  brew install imagemagick
  ```

- **Windows:**

  Download and install ImageMagick from the [official website](https://imagemagick.org/script/download.php).

### Verify Installation

Ensure all packages are installed correctly by running:

```bash
python -c "import pennylane; import ase; import matplotlib; import tqdm; import pandas"
```

If no errors are returned, the installation was successful.

## Usage

### `quantum_simulation.py`

This script allows you to perform quantum simulations of molecules and chemical reactions.

#### Adding a New Molecule

Before simulating a molecule, it must be defined in the `molecules.json` file. You can add a new molecule interactively using the `--add_molecule` flag.

**Command:**

```bash
python quantum_simulation.py --add_molecule
```

**Steps:**

1. **Enter Molecule Name:** Provide a unique identifier for the molecule (e.g., `H2`, `LiH`).
2. **Enter Atomic Symbols:** Input atomic symbols separated by commas (e.g., `H,H,O` for water).
3. **Enter Coordinates:** Provide the x, y, z coordinates for each atom.
4. **Enter Active Electrons:** Specify the number of active electrons.
5. **Enter Active Orbitals:** Specify the number of active orbitals.

The molecule details will be saved to `molecules.json`.

#### Simulating a Molecule

To simulate a molecule and find its ground state energy:

**Basic Command:**

```bash
python quantum_simulation.py --molecule MoleculeName
```

**Options:**

- `--optimizer`: Choose an optimizer (`GradientDescent`, `Adam`, `NesterovMomentum`, `RMSProp`). Default is `GradientDescent`.
- `--max_iterations`: Set the maximum number of optimization iterations. Default is `50`.
- `--conv_tol`: Define the convergence tolerance. Default is `1e-6`.
- `--stepsize`: Specify the step size for the optimizer. Default is `0.4`.
- `--basis_set`: Choose the basis set (e.g., `sto-3g`, `6-31G`, `cc-pVDZ`). Default is `sto-3g`.
- `--save`: Save the simulation results.
- `--save_dir`: Specify the directory to save results. If not provided, a default directory based on Git version and timestamp is used.
- `--plot`: Display the energy convergence plot.
- `--use_cached_results`: Use cached results if available.

**Example:**

```bash
python quantum_simulation.py --molecule H2 --optimizer Adam --max_iterations 100 --save --save_dir results/H2_adam_sim --plot
```

#### Simulating a Chemical Reaction

To simulate a chemical reaction and calculate its energy:

**Command:**

```bash
python quantum_simulation.py --reaction "Reactant1+Reactant2->Product1+Product2"
```

**Options:**

- All options available for molecule simulation also apply here.
- Ensure all reactants and products are defined in `molecules.json`.

**Example:**

```bash
python quantum_simulation.py --reaction "H2+O2->H2O2" --optimizer Adam --max_iterations 100 --save --save_dir results/H2O2_reaction --plot
```

#### Viewing Help

For a full list of options and usage instructions:

```bash
python quantum_simulation.py --help
```

### `test_quantum_simulation.py`

This script automates the testing of `quantum_simulation.py` by running a series of predefined tests.

#### Running Tests

**Basic Command:**

```bash
python test_quantum_simulation.py
```

**Options:**

- `--test_type`: Choose the type of tests to run (`fast` or `full`). Default is `fast`.
- `--show_plots`: Display plots during the tests.

**Examples:**

- **Run Fast Tests:**

  ```bash
  python test_quantum_simulation.py --test_type fast
  ```

- **Run Full Tests Without Plots:**

  ```bash
  python test_quantum_simulation.py --test_type full
  ```

- **Run Fast Tests and Show Plots:**

  ```bash
  python test_quantum_simulation.py --show_plots
  ```

- **Run Full Tests and Show Plots:**

  ```bash
  python test_quantum_simulation.py --test_type full --show_plots
  ```

#### Test Scenarios

##### Fast Tests

1. **Simulate H₂ with GradientDescent and Show Plot:**
   ```bash
   python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot
   ```
2. **Simulate LiH with Adam and Save Results:**
   ```bash
   python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 50 --save --save_dir results/results_lih_fast
   ```
3. **Test Error Handling with Undefined Molecule:**
   ```bash
   python quantum_simulation.py --molecule XYZ --optimizer Adam
   ```
4. **Test Error Handling with Invalid Optimizer:**
   ```bash
   python quantum_simulation.py --molecule H2 --optimizer UnknownOptimizer
   ```

##### Full Tests

1. **Simulate H₂ with GradientDescent and Show Plot:**
   ```bash
   python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot
   ```
2. **Simulate LiH with Adam, Increase Iterations, and Save Results:**
   ```bash
   python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 100 --save --save_dir results/results_lih
   ```
3. **Simulate H₂O with RMSProp, Change Stepsize, and Show Plot:**
   ```bash
   python quantum_simulation.py --molecule H2O --optimizer RMSProp --stepsize 0.1 --plot
   ```
4. **Simulate H₂ with Basis '6-31G' and Save Results:**
   ```bash
   python quantum_simulation.py --molecule H2 --basis_set 6-31G --save --save_dir results/results_h2_631G
   ```
5. **Simulate LiH with NesterovMomentum, Change conv_tol, and Show Plot:**
   ```bash
   python quantum_simulation.py --molecule LiH --optimizer NesterovMomentum --conv_tol 1e-8 --plot
   ```
6. **Simulate CH₄ and Show Plot:**
   ```bash
   python quantum_simulation.py --molecule CH4 --optimizer Adam --max_iterations 100 --plot
   ```
7. **Test Error Handling with Undefined Molecule:**
   ```bash
   python quantum_simulation.py --molecule XYZ --optimizer Adam
   ```
8. **Test Error Handling with Invalid Optimizer:**
   ```bash
   python quantum_simulation.py --molecule H2 --optimizer UnknownOptimizer
   ```
9. **Simulate H₂ with Small Stepsize and High Tolerance:**
   ```bash
   python quantum_simulation.py --molecule H2 --optimizer GradientDescent --stepsize 0.01 --conv_tol 1e-4 --max_iterations 200 --plot
   ```
10. **Simulate H₂O with Basis cc-pVDZ and Show Plot:**
    ```bash
    python quantum_simulation.py --molecule H2O --basis_set cc-pVDZ --optimizer Adam --max_iterations 100 --plot
    ```
11. **Test Saving and Loading Results:**
    ```bash
    python quantum_simulation.py --molecule H2 --optimizer Adam --save --save_dir results/results_h2_adam
    ```

#### Adding Missing Molecules

Ensure that all molecules used in the tests (e.g., `H2`, `LiH`, `H2O`, `CH4`) are defined in `molecules.json`. If not, add them using the `--add_molecule` flag as described in the **Adding a New Molecule** section above.

## molecules.json

The `molecules.json` file stores the definitions of all molecules that can be simulated. Each molecule entry includes:

- **symbols:** List of atomic symbols (e.g., `["H", "O"]`).
- **coordinates:** Flattened list of x, y, z coordinates for each atom.
- **active_electrons:** Number of active electrons.
- **active_orbitals:** Number of active orbitals.

**Example Entry:**

```json
{
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": [0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
        "active_electrons": 2,
        "active_orbitals": 2
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": [0.0, 0.0, 0.0, 0.0, 0.0, 1.6],
        "active_electrons": 4,
        "active_orbitals": 4
    }
}
```

**Managing Molecules:**

- **Adding Molecules:** Use the `--add_molecule` flag with `quantum_simulation.py` to add new molecules interactively.
- **Editing Molecules:** Manually edit `molecules.json` with caution to ensure correct formatting.
- **Removing Molecules:** Delete the corresponding entry from `molecules.json`.

## Dependencies

Ensure all the following Python packages are installed:

- [PennyLane](https://pennylane.ai/) (for quantum simulations)
- [Matplotlib](https://matplotlib.org/) (for plotting and visualization)
- [Tqdm](https://tqdm.github.io/) (for progress bars)
- [Pandas](https://pandas.pydata.org/) (for data management)
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) (for molecular structures)

**Install via pip:**

```bash
pip install pennylane matplotlib tqdm pandas ase
```

**Additional Requirements:**

- **ImageMagick:** Required for saving animations as GIFs.

  - **Ubuntu/Debian:**

    ```bash
    sudo apt-get install imagemagick
    ```

  - **macOS (using Homebrew):**

    ```bash
    brew install imagemagick
    ```

  - **Windows:**

    Download and install ImageMagick from the [official website](https://imagemagick.org/script/download.php).

## Examples

### Simulating a Molecule

Simulate a Hydrogen molecule (H₂) using the Gradient Descent optimizer and display the energy convergence plot.

```bash
python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot
```

### Simulating a Chemical Reaction

Simulate the reaction of Hydrogen and Oxygen to form Hydrogen Peroxide (H₂ + O₂ → H₂O₂), using the Adam optimizer, saving the results, and displaying the plot.

```bash
python quantum_simulation.py --reaction "H2+O2->H2O2" --optimizer Adam --max_iterations 100 --save --save_dir results/H2O2_reaction --plot
```

### Running Automated Tests

Run the full suite of tests without displaying plots:

```bash
python test_quantum_simulation.py --test_type full
```

Run fast tests and display plots:

```bash
python test_quantum_simulation.py --test_type fast --show_plots
```

## Notes

- **Git Integration:** The scripts attempt to retrieve the current Git commit hash to version the results directories. Ensure that your project is initialized with Git, and you have committed your changes. If Git is not available, the version will be marked as `'unknown'`.

- **Result Directories:** When using the `--save` flag without specifying `--save_dir`, results are saved in a directory structured as `results/version_<git_hash>/<molecule_or_reaction>_<timestamp>/`.

- **Error Handling:** The scripts include error handling for undefined molecules and invalid optimizer choices. Review the console output to identify and resolve any issues.

- **Temporary Files:** Molecular visualizations create temporary image files that are securely handled and removed after use to prevent clutter.

- **Performance Considerations:** Quantum simulations can be computationally intensive. Ensure your system has adequate resources, especially when running full test suites or simulations with a large number of iterations.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or support, please contact [albert.lopez.escudero@estudiantat.upc.edu](mailto:albert.lopez.escudero@estudiantat.upc.edu).

---
