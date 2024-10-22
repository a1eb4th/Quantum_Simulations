import os
import sys
import subprocess
from pennylane import numpy as np
import pennylane as qml
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import tempfile
import datetime
import hashlib
import argparse
import json
import re

from exceptions import QuantumSimulationError


class QuantumSimulation:
    def __init__(self, symbols, coordinates, active_electrons, active_orbitals, multiplicity=1, basis_set='sto-3g', name=''):
        self.symbols = symbols
        self.coordinates = coordinates
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.multiplicity = multiplicity
        self.basis_set = basis_set
        self.name = name
        self.H = None
        self.n_spin_orbitals = None
        self.hf_state = None
        self.energy_history = []
        self.ground_state_energy = None
        self.optimized_weights = None
        self.s_wires = None
        self.d_wires = None
        self.dev = None
        self.weights = None
        # Initialize other necessary attributes as needed

    def generate_molecular_hamiltonian(self):
        """
        Generates the molecular Hamiltonian and Hartree-Fock state for the molecule.
        Dynamically adjusts active orbitals if necessary to ensure at least one virtual orbital.
        """
        try:
            method = 'openfermion' if self.multiplicity > 1 else 'dhf'

            print("Generating molecular Hamiltonian with the following parameters:")
            print(f"Symbols: {self.symbols}")
            print(f"Coordinates: {self.coordinates}")
            print(f"Name: {self.name}")
            print(f"Basis Set: {self.basis_set}")
            print(f"Active Electrons: {self.active_electrons}")
            print(f"Active Orbitals: {self.active_orbitals}")
            print(f"Multiplicity: {self.multiplicity}")
            print(f"Method: {method}")

            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                try:
                    # Attempt to generate the Hamiltonian
                    mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=self.coordinates,
                        mult=self.multiplicity,
                        name=self.name,
                        basis_name=self.basis_set
                    )
                    self.H, qubits = qml.qchem.molecular_hamiltonian(
                        molecule=mol,
                        active_electrons=self.active_electrons,
                        active_orbitals=self.active_orbitals,
                        method=method,
                    )

                    # Successfully generated Hamiltonian
                    self.n_spin_orbitals = qubits
                    self.hf_state = np.array(qml.qchem.hf_state(self.active_electrons, self.n_spin_orbitals), dtype=int)
                    print(f"Molecular Hamiltonian and Hartree-Fock state successfully generated for {self.name}.")
                    break  # Exit the loop upon success

                except ValueError as e:
                    if "no virtual orbitals" in str(e).lower():
                        print(f"No virtual orbitals available for {self.name}. Using known energy value.")
                        self.n_spin_orbitals = self.active_orbitals * 2

                        # Manually set the Hartree-Fock state
                        hf_state = [0] * self.n_spin_orbitals
                        for i in range(self.active_electrons):
                            hf_state[i] = 1
                        self.hf_state = np.array(hf_state, dtype=int)

                        # Manually set the energy for hydrogen atom or other atoms
                        if self.name == 'H':
                            self.ground_state_energy = -0.5  # Known Hartree-Fock energy for H atom
                            print(f"Hartree-Fock energy for {self.name} set manually to {self.ground_state_energy:.8f} Ha")
                        elif self.name == 'He':
                            self.ground_state_energy = -2.86168  # Known Hartree-Fock energy for He atom
                            print(f"Hartree-Fock energy for {self.name} set manually to {self.ground_state_energy:.8f} Ha")
                        else:
                            raise QuantumSimulationError(
                                f"Cannot calculate energy for system '{self.name}' without virtual orbitals."
                            )
                        break  # Exit the loop
                    else:
                        raise QuantumSimulationError(f"Error generating molecular Hamiltonian: {e}")

                attempts += 1

            if attempts == max_attempts:
                raise QuantumSimulationError(f"Exceeded attempts to configure active space for {self.name}.")

        except Exception as e:
            raise QuantumSimulationError(f"Failed to generate Hamiltonian: {e}")

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
        os.makedirs(directory, exist_ok=True)

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
            self.active_electrons = int(data['active_electrons'])
            self.active_orbitals = int(data['active_orbitals'])
            self.basis_set = str(data['basis_set'])
            self.energy_history = data['energy_history'].tolist()
            self.optimized_weights = data['optimized_weights'].tolist() if 'optimized_weights' in data and data['optimized_weights'].shape else None
            self.ground_state_energy = float(data['ground_state_energy'])
            print(f"Previous results loaded from {results_file}")
            return True
        else:
            return False

    def run_vqe(self, optimizer_choice='Adam', max_iterations=50, conv_tol=1e-6, stepsize=0.4, use_cached_results=False, results_dir=None):
        """
        Runs the VQE algorithm to find the minimum energy of the molecular Hamiltonian.
        """
        if use_cached_results and results_dir:
            if self.load_results(results_dir, optimizer_choice, max_iterations, conv_tol, stepsize):
                return  # Skip optimization if cached results are used

        # Determine whether to perform VQE based on the availability of virtual orbitals
        perform_vqe = self.H is not None and self.n_spin_orbitals > self.active_orbitals

        if perform_vqe:
            # Perform VQE
            # VQE execution begins here
            singles, doubles = qml.qchem.excitations(self.active_electrons, self.n_spin_orbitals)
            self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(singles, doubles, wires=range(self.n_spin_orbitals))

            # Set up the quantum device
            self.dev = qml.device('default.qubit', wires=self.n_spin_orbitals, shots=None)

            def circuit(weights):
                qml.templates.UCCSD(
                    weights,
                    wires=range(self.n_spin_orbitals),
                    s_wires=self.s_wires,
                    d_wires=self.d_wires,
                    init_state=self.hf_state,
                )

            @qml.qnode(self.dev)
            def cost_fn(weights):
                circuit(weights)
                return qml.expval(self.H)

            # Initialize optimizer and weights
            num_params = len(singles) + len(doubles)
            self.weights = np.zeros(num_params, requires_grad=True)

            optimizers = {
                'GradientDescent': qml.GradientDescentOptimizer,
                'Adam': qml.AdamOptimizer,
                'NesterovMomentum': qml.NesterovMomentumOptimizer,
                'RMSProp': qml.RMSPropOptimizer,
            }

            optimizer_class = optimizers.get(optimizer_choice, qml.GradientDescentOptimizer)
            optimizer = optimizer_class(stepsize=stepsize)

            # Run optimization
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
        
            pass  # (For brevity)
        else:
            # No VQE: use Hartree-Fock energy directly
            if self.ground_state_energy is None:
                # Calculate Hartree-Fock energy
                try:
                    mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=self.coordinates,
                        charge=0,
                        mult=self.multiplicity,
                        name=self.name
                    )
                    hf_energy_function = qml.qchem.hf_energy(mol)
                    self.ground_state_energy = hf_energy_function()
                    print(f"Hartree-Fock energy for {self.name}: {self.ground_state_energy:.8f} Ha")
                except Exception as ex:
                    raise QuantumSimulationError(f"Error calculating Hartree-Fock energy: {ex}")
            else:
                print(f"No virtual orbitals available for {self.name}. Using Hartree-Fock energy only.")
                print(f"Energy for {self.name} (Hartree-Fock): {self.ground_state_energy:.8f} Ha")
            self.energy_history = [self.ground_state_energy]
            self.optimized_weights = None

    def get_final_state(self):
        """
        Obtains the final quantum state after optimization.
        """
        if self.optimized_weights is None or self.dev is None or self.s_wires is None or self.d_wires is None:
            raise QuantumSimulationError("VQE has not been run or missing data to obtain the final state.")

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

    def animate_energy_convergence(self, save_path=None, show_plot=True):
        """
        Creates an animation of energy convergence over iterations.

        Args:
            save_path (str): File path where the animation will be saved.
            show_plot (bool): Indicates whether to display the animation.
        """
        if not self.energy_history:
            print("No energy history to animate.")
            return

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
            ax.set_xlim(1, max(10, len(self.energy_history)))
            ymin = min(y) - 0.1 * abs(min(y))
            ymax = max(y) + 0.1 * abs(max(y))
            ax.set_ylim(ymin, ymax)
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
        plt.close(fig)

    def plot_energy_convergence(self, save_path=None, show_plot=True):
        """
        Plots the energy convergence during optimization.
        If save_path is provided, saves the plot to the specified file.

        Args:
            save_path (str): File path where the plot will be saved.
            show_plot (bool): Indicates whether to display the plot.
        """
        if not self.energy_history:
            print("No energy history to plot.")
            return

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.energy_history) + 1), self.energy_history, 'o-', color='blue', label='Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Ha)')
        plt.title(f'Energy Convergence during Optimization of {self.name}')
        plt.grid(True)
        plt.legend()
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Convergence plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving convergence plot: {e}")
        if show_plot:
            plt.show()
        else:
            plt.close()