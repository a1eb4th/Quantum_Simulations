#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import json
import datetime
import hashlib
import tempfile

import pennylane as qml
from pennylane import numpy as np  # Import pennylane.numpy for autograd support
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm  # For progress bar
import pandas as pd  # For data organization and visualization

from ase import Atoms
from ase.visualize import view
from ase.io import write


class QuantumSimulationError(Exception):
    """Custom exception for QuantumSimulation."""
    pass


class QuantumSimulation:
    """
    Class to perform quantum simulations of molecules using VQE.
    """

    def __init__(self, symbols, coordinates, active_electrons, active_orbitals, basis_set='sto-3g', name='molecule'):
        self.symbols = symbols
        self.coordinates = coordinates
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.basis_set = basis_set
        self.name = name  # Molecule name
        self.n_spin_orbitals = 2 * active_orbitals  # Total number of spin orbitals
        self.hf_state = None
        self.H = None
        self.dev = None
        self.weights = None
        self.energy_history = []
        self.optimized_weights = None
        self.ground_state_energy = None

    def generate_molecular_hamiltonian(self):
        """
        Generates the molecular Hamiltonian and Hartree-Fock state for the molecule.
        """
        try:
            self.H, qubits = qml.qchem.molecular_hamiltonian(
                self.symbols,
                self.coordinates,
                name=self.name,
                basis=self.basis_set,
                active_electrons=self.active_electrons,
                active_orbitals=self.active_orbitals,
            )

            self.hf_state = np.array(qml.qchem.hf_state(self.active_electrons, self.n_spin_orbitals), dtype=int)
            print(f"Molecular Hamiltonian and Hartree-Fock state successfully generated for {self.name}.")
        except Exception as e:
            raise QuantumSimulationError(f"Error generating molecular Hamiltonian: {e}")

    def get_simulation_identifier(self, optimizer_choice, max_iterations, conv_tol, stepsize):
        """
        Generates a unique identifier for the simulation based on its parameters.
        """
        identifier_str = f"{self.name}_{self.basis_set}_{self.active_electrons}_{self.active_orbitals}"
        identifier_str += f"_{optimizer_choice}_{max_iterations}_{conv_tol}_{stepsize}"
        return hashlib.md5(identifier_str.encode()).hexdigest()

    def save_results(self, directory, optimizer_choice, max_iterations, conv_tol, stepsize):
        """
        Saves the simulation results to a file within the specified directory.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Use the unique identifier in the file name
        simulation_id = self.get_simulation_identifier(optimizer_choice, max_iterations, conv_tol, stepsize)
        results_file = os.path.join(directory, f'results_{self.name}_{simulation_id}.npz')

        # Save results to a .npz file
        results = {
            'symbols': self.symbols,
            'coordinates': self.coordinates,
            'active_electrons': self.active_electrons,
            'active_orbitals': self.active_orbitals,
            'basis_set': self.basis_set,
            'energy_history': [float(e) for e in self.energy_history],
            'optimized_weights': self.optimized_weights,
            'ground_state_energy': float(self.ground_state_energy),
            'optimizer_choice': optimizer_choice,
            'max_iterations': max_iterations,
            'conv_tol': conv_tol,
            'stepsize': stepsize
        }

        try:
            np.savez(results_file, **results)
            print(f"Results saved to {results_file}")
        except Exception as e:
            raise QuantumSimulationError(f"Error saving results: {e}")

    def load_results(self, directory, optimizer_choice, max_iterations, conv_tol, stepsize):
        """
        Loads simulation results if they exist.
        """
        simulation_id = self.get_simulation_identifier(optimizer_choice, max_iterations, conv_tol, stepsize)
        results_file = os.path.join(directory, f'results_{self.name}_{simulation_id}.npz')
        if os.path.exists(results_file):
            data = np.load(results_file, allow_pickle=True)
            self.symbols = data['symbols']
            self.coordinates = data['coordinates']
            self.active_electrons = data['active_electrons'].item()
            self.active_orbitals = data['active_orbitals'].item()
            self.basis_set = data['basis_set'].item()
            self.energy_history = data['energy_history']
            self.optimized_weights = data['optimized_weights']
            self.ground_state_energy = data['ground_state_energy'].item()
            print(f"Previous results loaded from {results_file}")
            return True
        else:
            return False

    def run_vqe(self, optimizer_choice='GradientDescent', max_iterations=50, conv_tol=1e-6, stepsize=0.4, use_cached_results=False, results_dir=None):
        """
        Runs the VQE algorithm to find the minimum energy of the molecular Hamiltonian.

        Args:
            optimizer_choice (str): Optimizer to use ('GradientDescent', 'Adam', etc.).
            max_iterations (int): Maximum number of optimization iterations.
            conv_tol (float): Convergence tolerance.
            stepsize (float): Step size for the optimizer.
            use_cached_results (bool): Indicates whether to use cached results if available.
            results_dir (str): Directory where results will be saved or loaded.
        """
        if use_cached_results and results_dir:
            if self.load_results(results_dir, optimizer_choice, max_iterations, conv_tol, stepsize):
                # Results loaded, skip optimization
                return

        # Generate excitations
        singles, doubles = qml.qchem.excitations(self.active_electrons, self.n_spin_orbitals)

        # Map excitations to wires
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles, wires=range(self.n_spin_orbitals))

        # Save s_wires and d_wires for later use
        self.s_wires = s_wires
        self.d_wires = d_wires

        # Set up the quantum device
        self.dev = qml.device('default.qubit', wires=self.n_spin_orbitals, shots=None)

        # Define the UCCSD ansatz
        def circuit(weights):
            qml.templates.UCCSD(
                weights,
                wires=range(self.n_spin_orbitals),
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=self.hf_state,
            )

        # Define the cost function
        @qml.qnode(self.dev)
        def cost_fn(weights):
            circuit(weights)
            return qml.expval(self.H)

        # Initialize parameters and optimizer
        num_params = len(singles) + len(doubles)
        self.weights = np.zeros(num_params, requires_grad=True)

        # Select the optimizer
        optimizers = {
            'GradientDescent': qml.GradientDescentOptimizer,
            'Adam': qml.AdamOptimizer,
            'NesterovMomentum': qml.NesterovMomentumOptimizer,
            'RMSProp': qml.RMSPropOptimizer,
        }

        optimizer_class = optimizers.get(optimizer_choice, qml.GradientDescentOptimizer)
        optimizer = optimizer_class(stepsize=stepsize)

        # Run the optimization with a progress bar
        self.energy_history = []
        print(f"Running VQE with {optimizer_choice} optimizer for {self.name}...")
        with tqdm(total=max_iterations, desc=f"Optimization {self.name}") as pbar:
            for n in range(max_iterations):
                self.weights, prev_energy = optimizer.step_and_cost(cost_fn, self.weights)
                current_energy = cost_fn(self.weights)
                self.energy_history.append(current_energy)
                conv = np.abs(current_energy - prev_energy)

                pbar.set_postfix({'Energy': f"{current_energy:.6f} Ha", 'Convergence': f"{conv:.2e} Ha"})
                pbar.update(1)

                if conv <= conv_tol:
                    print(f"\nConvergence reached at iteration {n+1}.")
                    break

        self.optimized_weights = self.weights
        self.ground_state_energy = self.energy_history[-1]
        print(f"\nMinimum energy found for {self.name}: {self.ground_state_energy:.8f} Ha")

    def get_final_state(self):
        """
        Obtains the final quantum state after optimization.
        """
        @qml.qnode(self.dev)
        def final_state_circuit():
            qml.templates.UCCSD(
                self.optimized_weights,
                wires=range(self.n_spin_orbitals),
                s_wires=self.s_wires,
                d_wires=self.d_wires,
                init_state=self.hf_state,
            )
            return qml.state()

        state = final_state_circuit()
        return state

    def visualize_molecule(self, save_path=None, show_plot=True):
        """
        Visualizes the molecular structure.

        Args:
            save_path (str): File path where the image will be saved.
            show_plot (bool): Indicates whether to display the plot on screen.
        """
        atoms = Atoms(symbols=self.symbols, positions=self.coordinates.reshape(-1, 3))
        if save_path:
            write(save_path, atoms, format='png')
            print(f"Molecular structure saved to {save_path}")
        if show_plot:
            view(atoms)

    def animate_energy_convergence(self, save_path=None, show_plot=True):
        """
        Creates an animation of energy convergence over iterations.

        Args:
            save_path (str): File path where the animation will be saved.
            show_plot (bool): Indicates whether to display the animation.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        ax.set_title(f'Energy Convergence of {self.name}')
        line, = ax.plot([], [], 'b-')

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            x = list(range(1, frame + 2))
            y = self.energy_history[:frame + 1]
            line.set_data(x, y)
            ax.set_xlim(1, len(self.energy_history))
            ax.set_ylim(
                min(self.energy_history) - 0.1 * abs(min(self.energy_history)),
                max(self.energy_history) + 0.1 * abs(max(self.energy_history))
            )
            return line,

        ani = FuncAnimation(fig, update, frames=len(self.energy_history), init_func=init, blit=True)

        if save_path:
            try:
                ani.save(save_path, writer='imagemagick')
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
        if show_plot:
            plt.show()

    def plot_energy_convergence(self, save_path=None, show_plot=True):
        """
        Plots the energy convergence during optimization.
        If save_path is provided, saves the plot to the specified file.

        Args:
            save_path (str): File path where the plot will be saved.
            show_plot (bool): Indicates whether to display the plot on screen.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.energy_history) + 1), self.energy_history, 'o-', color='blue', label='Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Ha)')
        plt.title(f'Energy Convergence during Optimization of {self.name}')
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            print(f"Convergence plot saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

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

    @staticmethod
    def load_molecules():
        """
        Loads defined molecules from the molecules.json file.

        Returns:
            molecules (dict): Dictionary of molecules.
        """
        molecules_file = 'molecules.json'
        if not os.path.exists(molecules_file):
            print(f"The file '{molecules_file}' does not exist. Creating an empty file.")
            with open(molecules_file, 'w') as f:
                json.dump({}, f)
        with open(molecules_file, 'r') as f:
            molecules = json.load(f)
        return molecules

    @staticmethod
    def add_new_molecule():
        """
        Adds a new molecule by interacting with the user.
        """
        print("Add a new molecule to the system.")
        name = input("Molecule name (unique identifier): ").strip()
        symbols = input("Atomic symbols separated by commas (e.g., H,H,O): ").strip().split(',')
        symbols = [s.strip() for s in symbols]
        num_atoms = len(symbols)
        print(f"Enter coordinates for {num_atoms} atoms (x, y, z for each atom).")
        coordinates = []
        for i in range(num_atoms):
            coords_input = input(f"Coordinates of atom {i+1} ({symbols[i]}), separated by commas (x,y,z): ").strip().split(',')
            try:
                coords = [float(c.strip()) for c in coords_input]
                if len(coords) != 3:
                    print("Error: You must enter three values for x, y, z.")
                    sys.exit(1)
                coordinates.extend(coords)
            except ValueError:
                print("Error: Coordinates must be numbers.")
                sys.exit(1)
        try:
            active_electrons = int(input("Number of active electrons: ").strip())
            active_orbitals = int(input("Number of active orbitals: ").strip())
        except ValueError:
            print("Error: You must enter integer numbers for active electrons and orbitals.")
            sys.exit(1)

        # Create the molecule dictionary
        molecule = {
            'symbols': symbols,
            'coordinates': coordinates,
            'active_electrons': active_electrons,
            'active_orbitals': active_orbitals,
        }

        # Load existing molecules
        molecules = QuantumSimulation.load_molecules()

        # Check if the molecule already exists
        if name in molecules:
            print(f"The molecule '{name}' already exists. Do you want to overwrite it? (y/n)")
            overwrite = input().strip().lower()
            if overwrite != 'y':
                print("Operation cancelled.")
                sys.exit(0)

        # Add the new molecule
        molecules[name] = molecule

        # Save the updated molecules
        with open('molecules.json', 'w') as f:
            json.dump(molecules, f, indent=4)

        print(f"Molecule '{name}' successfully added.")


    @staticmethod
    def from_user_input():
        """
        Creates an instance of QuantumSimulation or ChemicalReaction from user input.

        Returns:
            (object, Namespace): An instance of ChemicalReaction or QuantumSimulation along with the arguments.
        """
        parser = argparse.ArgumentParser(description='Quantum simulation of molecules and chemical reactions using VQE.')
        parser.add_argument('--molecule', type=str, help='Molecule to simulate.')
        parser.add_argument('--reaction', type=str, help='Chemical reaction to simulate (reactants->products).')
        parser.add_argument('--basis_set', type=str, default='sto-3g', help='Basis set to use.')
        parser.add_argument('--optimizer', type=str, default='GradientDescent', help='Optimizer to use.')
        parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of optimization iterations.')
        parser.add_argument('--conv_tol', type=float, default=1e-6, help='Convergence tolerance.')
        parser.add_argument('--stepsize', type=float, default=0.4, help='Step size for the optimizer.')
        parser.add_argument('--save', action='store_true', help='Save results to the results folder.')
        parser.add_argument('--save_dir', type=str, help='Name of the directory where results will be saved.')
        parser.add_argument('--plot', action='store_true', help='Show the energy convergence plot.')
        parser.add_argument('--add_molecule', action='store_true', help='Add a new molecule to the system.')
        parser.add_argument('--use_cached_results', action='store_true', help='Use cached results if available.')
        args = parser.parse_args()

        # If the user wants to add a molecule
        if args.add_molecule:
            QuantumSimulation.add_new_molecule()
            sys.exit(0)

        # If a chemical reaction is specified
        if args.reaction:
            return 'reaction', args

        # Load molecules from molecules.json
        molecules = QuantumSimulation.load_molecules()

        if not args.molecule:
            print("Error: You must specify a molecule with '--molecule' or a reaction with '--reaction'.")
            sys.exit(1)

        if args.molecule not in molecules:
            print(f"Molecule '{args.molecule}' is not defined. Use '--add_molecule' to add it.")
            sys.exit(1)

        molecule_data = molecules[args.molecule]

        simulation = QuantumSimulation(
            symbols=molecule_data['symbols'],
            coordinates=np.array(molecule_data['coordinates']),
            active_electrons=molecule_data['active_electrons'],
            active_orbitals=molecule_data['active_orbitals'],
            basis_set=args.basis_set,
            name=args.molecule
        )
        return simulation, args


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

    def parse_reaction(self):
        """
        Parses the reaction string to extract reactants and products.
        """
        try:
            reactants_str, products_str = self.reaction_str.split('->')
            self.reactants = [s.strip() for s in reactants_str.split('+')]
            self.products = [s.strip() for s in products_str.split('+')]
        except ValueError:
            print("Error: The reaction must be in the format 'Reactant1+Reactant2->Product1+Product2'.")
            sys.exit(1)

    def check_molecules_defined(self):
        """
        Checks that all molecules in the reaction are defined.
        """
        all_molecules = self.reactants + self.products
        undefined_molecules = [m for m in all_molecules if m not in self.molecules_data]
        if undefined_molecules:
            print(f"The following molecules are not defined: {', '.join(undefined_molecules)}")
            print("Use '--add_molecule' to add them.")
            sys.exit(1)

    def get_results_directory(self):
        """
        Retrieves the directory where reaction results will be saved, based on Git version and timestamp.

        Returns:
            results_dir (str): Path to the results directory.
        """
        git_version = self.get_git_version()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        reaction_str_formatted = self.reaction_str.replace(' ', '').replace('+', '_').replace('->', '_to_')
        results_dir = os.path.join('results', f'version_{git_version}', f'reaction_{reaction_str_formatted}_{timestamp}')
        return results_dir

    def simulate_reaction(self, optimizer_choice='GradientDescent', max_iterations=50, conv_tol=1e-6, stepsize=0.4, use_cached_results=False):
        """
        Simulates the chemical reaction.

        Args:
            optimizer_choice (str): Optimizer to use.
            max_iterations (int): Maximum number of iterations.
            conv_tol (float): Convergence tolerance.
            stepsize (float): Step size.
            use_cached_results (bool): Indicates whether to use cached results if available.
        """
        self.reactant_energies = []
        self.product_energies = []

        # Simulate reactants
        print("Simulating reactants...")
        for molecule_name in self.reactants:
            molecule_data = self.molecules_data[molecule_name]
            simulation = QuantumSimulation(
                symbols=molecule_data['symbols'],
                coordinates=np.array(molecule_data['coordinates']),
                active_electrons=molecule_data['active_electrons'],
                active_orbitals=molecule_data['active_orbitals'],
                basis_set=self.basis_set,
                name=molecule_name
            )
            simulation.generate_molecular_hamiltonian()
            simulation.run_vqe(
                optimizer_choice=optimizer_choice,
                max_iterations=max_iterations,
                conv_tol=conv_tol,
                stepsize=stepsize,
                use_cached_results=use_cached_results
            )
            self.reactant_energies.append(simulation.ground_state_energy)
            self.simulations.append(simulation)

        # Simulate products
        print("Simulating products...")
        for molecule_name in self.products:
            molecule_data = self.molecules_data[molecule_name]
            simulation = QuantumSimulation(
                symbols=molecule_data['symbols'],
                coordinates=np.array(molecule_data['coordinates']),
                active_electrons=molecule_data['active_electrons'],
                active_orbitals=molecule_data['active_orbitals'],
                basis_set=self.basis_set,
                name=molecule_name
            )
            simulation.generate_molecular_hamiltonian()
            simulation.run_vqe(
                optimizer_choice=optimizer_choice,
                max_iterations=max_iterations,
                conv_tol=conv_tol,
                stepsize=stepsize,
                use_cached_results=use_cached_results
            )
            self.product_energies.append(simulation.ground_state_energy)
            self.simulations.append(simulation)

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
        molecules = self.reactants + self.products
        energies = self.reactant_energies + self.product_energies

        fig = plt.figure(figsize=(4 * len(molecules), 6))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(molecules)), axes_pad=0.5)

        for i, molecule_name in enumerate(molecules):
            molecule_data = self.molecules_data[molecule_name]
            symbols = molecule_data['symbols']
            coordinates = np.array(molecule_data['coordinates']).reshape(-1, 3)
            atoms = Atoms(symbols=symbols, positions=coordinates)

            # Save temporary image securely
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                write(tmp_file.name, atoms, format='png')
                image = mpimg.imread(tmp_file.name)
            grid[i].imshow(image)
            grid[i].axis('off')
            grid[i].set_title(f"{molecule_name}\nEnergy: {energies[i]:.4f} Ha")

            # Remove temporary image
            os.remove(tmp_file.name)

        plt.suptitle(f'Reaction Energy: {self.reaction_energy:.6f} Ha', fontsize=16)

        if save_path:
            plt.savefig(save_path)
            print(f"Reaction plot saved to {save_path}")
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
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save individual molecule results
        for simulation in self.simulations:
            molecule_dir = os.path.join(directory, simulation.name)
            simulation.save_results(molecule_dir, optimizer_choice, max_iterations, conv_tol, stepsize)

        # Save reaction information
        reaction_file = os.path.join(directory, 'reaction_results.txt')
        with open(reaction_file, 'w') as f:
            f.write(f"Reaction: {self.reaction_str}\n")
            f.write(f"Reaction Energy: {self.reaction_energy:.8f} Ha\n")
            f.write(f"Reactant Energies:\n")
            for name, energy in zip(self.reactants, self.reactant_energies):
                f.write(f"  {name}: {energy:.8f} Ha\n")
            f.write(f"Product Energies:\n")
            for name, energy in zip(self.products, self.product_energies):
                f.write(f"  {name}: {energy:.8f} Ha\n")
        print(f"Reaction results saved to {reaction_file}")

        # Save the reaction plot if necessary
        plot_file = os.path.join(directory, 'reaction_energy.png')
        self.plot_reaction_energy(save_path=plot_file, show_plot=False)

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


def main():
    # Create an instance of QuantumSimulation or ChemicalReaction from user input
    simulation_or_reaction, args = QuantumSimulation.from_user_input()

    if simulation_or_reaction == 'reaction':
        # It's a chemical reaction
        reaction = ChemicalReaction(
            reaction_str=args.reaction,
            basis_set=args.basis_set
        )
        reaction.simulate_reaction(
            optimizer_choice=args.optimizer,
            max_iterations=args.max_iterations,
            conv_tol=args.conv_tol,
            stepsize=args.stepsize,
            use_cached_results=args.use_cached_results
        )

        if args.save:
            if args.save_dir:
                results_dir = args.save_dir
            else:
                results_dir = reaction.get_results_directory()
            reaction.save_results(results_dir, args.optimizer, args.max_iterations, args.conv_tol, args.stepsize)
        if args.plot:
            reaction.plot_reaction_energy()
    else:
        # It's an individual molecule simulation
        simulation = simulation_or_reaction

        # Generate Hamiltonian
        simulation.generate_molecular_hamiltonian()

        # Run VQE
        simulation.run_vqe(
            optimizer_choice=args.optimizer,
            max_iterations=args.max_iterations,
            conv_tol=args.conv_tol,
            stepsize=args.stepsize,
            use_cached_results=args.use_cached_results,
            results_dir=args.save_dir  # Pass the results directory if specified
        )

        # Get the results directory
        if args.save:
            if args.save_dir:
                results_dir = args.save_dir
            else:
                results_dir = simulation.get_results_directory()
            # Save results
            simulation.save_results(results_dir, args.optimizer, args.max_iterations, args.conv_tol, args.stepsize)
            # Save convergence plot
            plot_file = os.path.join(results_dir, f'energy_convergence_{simulation.name}.png')
            simulation.plot_energy_convergence(save_path=plot_file, show_plot=False)
            # Visualize the molecule
            molecule_image_path = os.path.join(results_dir, f'molecule_{simulation.name}.png')
            simulation.visualize_molecule(save_path=molecule_image_path, show_plot=False)
            # Create an animation of energy convergence
            animation_path = os.path.join(results_dir, f'energy_animation_{simulation.name}.gif')
            simulation.animate_energy_convergence(save_path=animation_path, show_plot=False)
        else:
            # Show convergence plot if requested
            if args.plot:
                simulation.plot_energy_convergence()
                # Visualize the molecule
                simulation.visualize_molecule()


if __name__ == "__main__":
    main()
