import os
import sys
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tempfile
import matplotlib.image as mpimg

from molecule_simulation import QuantumSimulation, QuantumSimulationError
import re
from ase import Atoms
from ase.io import write
import subprocess
import datetime


class ChemicalReaction:
    """
    Class to simulate a chemical reaction.
    """

    def __init__(self, reaction_str, basis_set='sto-3g'):
        """
        Initializes the chemical reaction.

        Args:
            reaction_str (str): Chemical reaction in the format 'Reactant1+Reactant2->Product1+Product2'.
            basis_set (str): Basis set to use.
        """
        self.reaction_str = reaction_str
        self.basis_set = basis_set
        self.reactants = []
        self.products = []
        self.parse_reaction()
        self.molecules_data = QuantumSimulation.load_molecules()
        self.check_molecules_defined()
        self.simulations = []
        self.reaction_energy = None
        # Add a name derived from the reaction string
        self.name = self.reaction_str.replace('->', '_to_').replace('+', '_plus_').replace(' ', '')

    # ... [rest of your existing methods]

    @staticmethod
    def get_git_version():
        """
        Retrieves the current project version from Git (commit hash).

        Returns:
            version (str): Current commit hash or 'unknown' if not available.
        """
        try:
            version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            return version
        except Exception as e:
            print(f"Warning: Could not retrieve Git version: {e}")
            return 'unknown'

    def get_results_directory(self):
        """
        Retrieves the directory where results will be saved, based on Git version and timestamp.

        Returns:
            results_dir (str): Path to the results directory.
        """
        git_version = self.get_git_version()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join('results', f'version_{git_version}', f'{self.name}_{timestamp}')
        return results_dir

    def parse_reaction(self):
        """
        Parses the reaction string to extract reactants and products with their coefficients.
        """
        try:
            reactants_str, products_str = self.reaction_str.split('->')
            self.reactants = self._parse_side(reactants_str)
            self.products = self._parse_side(products_str)
        except ValueError:
            print("Error: The reaction must be in the format 'Reactant1+Reactant2->Product1+Product2'.")
            sys.exit(1)

    def _parse_side(self, side_str):
        """
        Parses one side of the reaction (reactants or products).

        Args:
            side_str (str): String representing one side of the reaction.

        Returns:
            list of tuples: Each tuple contains (coefficient, species).
        """
        species = side_str.split('+')
        parsed_species = []
        for s in species:
            s = s.strip()
            # Match optional coefficient followed by species name
            match = re.match(r'^(\d*)\s*([A-Za-z0-9]+)$', s)
            if match:
                coeff_str, species_name = match.groups()
                coeff = int(coeff_str) if coeff_str else 1
                if coeff <= 0:
                    print(f"Error: Coefficient must be a positive integer in species '{s}'.")
                    sys.exit(1)
                parsed_species.append((coeff, species_name))
            else:
                print(f"Error: Unable to parse species '{s}'. Ensure it follows the format '[coeff]Species'.")
                sys.exit(1)
        return parsed_species

    def check_molecules_defined(self):
        """
        Checks that all molecules in the reaction are defined.
        """
        # Extract only species names from reactants and products
        all_molecules = [species for coeff, species in self.reactants + self.products]
        undefined_molecules = [m for m in all_molecules if m not in self.molecules_data]
        if undefined_molecules:
            print(f"The following molecules are not defined: {', '.join(undefined_molecules)}")
            print("Use '--add_molecule' to add them.")
            sys.exit(1)

    def simulate_reaction(self, optimizer_choice='GradientDescent', max_iterations=50, conv_tol=1e-6, stepsize=0.4, use_cached_results=False):
        self.reactant_energies = []
        self.product_energies = []
        self.reactant_coeffs = []
        self.product_coeffs = []

        # Simulate reactants
        print("Simulating reactants...")
        for coeff, molecule_name in self.reactants:
            molecule_data = self.molecules_data[molecule_name]
            simulation = QuantumSimulation(
                symbols=molecule_data['symbols'],
                coordinates=np.array(molecule_data['coordinates']),
                active_electrons=molecule_data['active_electrons'],
                active_orbitals=molecule_data['active_orbitals'],
                multiplicity=molecule_data['multiplicity'],
                basis_set=self.basis_set,
                name=molecule_name
            )
            try:
                simulation.generate_molecular_hamiltonian()
                simulation.run_vqe(
                    optimizer_choice=optimizer_choice,
                    max_iterations=max_iterations,
                    conv_tol=conv_tol,
                    stepsize=stepsize,
                    use_cached_results=use_cached_results
                )
                self.reactant_energies.append(simulation.ground_state_energy * coeff)
                self.reactant_coeffs.append(coeff)
                self.simulations.append(simulation)
            except QuantumSimulationError as e:
                print(f"Error simulating reactant '{molecule_name}': {e}")
                sys.exit(1)

        # Simulate products
        print("Simulating products...")
        for coeff, molecule_name in self.products:
            molecule_data = self.molecules_data[molecule_name]
            simulation = QuantumSimulation(
                symbols=molecule_data['symbols'],
                coordinates=np.array(molecule_data['coordinates']),
                active_electrons=molecule_data['active_electrons'],
                active_orbitals=molecule_data['active_orbitals'],
                multiplicity=molecule_data['multiplicity'],
                basis_set=self.basis_set,
                name=molecule_name
            )
            try:
                simulation.generate_molecular_hamiltonian()
                simulation.run_vqe(
                    optimizer_choice=optimizer_choice,
                    max_iterations=max_iterations,
                    conv_tol=conv_tol,
                    stepsize=stepsize,
                    use_cached_results=use_cached_results
                )
                self.product_energies.append(simulation.ground_state_energy * coeff)
                self.product_coeffs.append(coeff)
                self.simulations.append(simulation)
            except QuantumSimulationError as e:
                print(f"Error simulating product '{molecule_name}': {e}")
                sys.exit(1)

        # Calculate reaction energy
        self.total_reactant_energy = sum(self.reactant_energies)
        self.total_product_energy = sum(self.product_energies)
        self.reaction_energy = self.total_product_energy - self.total_reactant_energy
        print(f"\nReaction energy: {self.reaction_energy:.8f} Ha")

    def plot_reaction_energy(self, save_path=None, show_plot=True):
        """
        Plots the energies of reactants and products with their molecular structures.

        Args:
            save_path (str): File path where the plot will be saved.
            show_plot (bool): Indicates whether to display the plot.
        """
        if not self.simulations:
            print("No simulations to plot.")
            return

        molecules = [species for coeff, species in self.reactants + self.products]
        energies = [energy for energy in self.reactant_energies + self.product_energies]

        fig = plt.figure(figsize=(4 * len(molecules), 6))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(molecules)), axes_pad=0.5)

        for i, molecule_name in enumerate(molecules):
            molecule_data = self.molecules_data[molecule_name]
            symbols = molecule_data['symbols']
            coordinates = np.array(molecule_data['coordinates']).reshape(-1, 3)
            atoms = Atoms(symbols=symbols, positions=coordinates)

            # Save temporary image securely
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    write(tmp_file.name, atoms, format='png')
                    image = mpimg.imread(tmp_file.name)
                    grid[i].imshow(image)
                    grid[i].axis('off')
                    grid[i].set_title(f"{molecule_name}\nEnergy: {energies[i]:.4f} Ha")
                except Exception as e:
                    print(f"Error generating image for {molecule_name}: {e}")
                finally:
                    tmp_file_name = tmp_file.name

            # Remove temporary image
            try:
                os.remove(tmp_file_name)
            except Exception as e:
                print(f"Error removing temporary file {tmp_file_name}: {e}")

        plt.suptitle(f'Reaction Energy: {self.reaction_energy:.6f} Ha', fontsize=16)

        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Reaction plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving reaction plot: {e}")
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_results(self, directory, optimizer_choice, max_iterations, conv_tol, stepsize):
        """
        Saves the reaction results in the specified directory.

        Args:
            directory (str): Directory where results will be saved.
        """
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save individual molecule results
        for simulation in self.simulations:
            molecule_dir = os.path.join(directory, simulation.name)
            simulation.save_results(molecule_dir, optimizer_choice, max_iterations, conv_tol, stepsize)

        # Save reaction information
        reaction_file = os.path.join(directory, 'reaction_results.txt')
        try:
            with open(reaction_file, 'w') as f:
                f.write(f"Reaction: {self.reaction_str}\n")
                f.write(f"Reaction Energy: {self.reaction_energy:.8f} Ha\n")
                f.write(f"Reactant Energies:\n")
                for (coeff, name), energy in zip(self.reactants, self.reactant_energies):
                    f.write(f"  {coeff} x {name}: {energy:.8f} Ha\n")
                f.write(f"Product Energies:\n")
                for (coeff, name), energy in zip(self.products, self.product_energies):
                    f.write(f"  {coeff} x {name}: {energy:.8f} Ha\n")
            print(f"Reaction results saved to {reaction_file}")
        except Exception as e:
            print(f"Error saving reaction results: {e}")

        # Save the reaction plot if necessary
        plot_file = os.path.join(directory, 'reaction_energy.png')
        self.plot_reaction_energy(save_path=plot_file, show_plot=False)

