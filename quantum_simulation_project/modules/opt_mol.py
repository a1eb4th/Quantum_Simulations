import numpy as np
from .hamiltonian_builder import build_hamiltonian, compute_exact_energy
from .optimizer import optimize_molecule
from .molecule_manager import initialize_molecule 
from .visualizer import visualize_results

def mol_optimizer(selected_molecules):
    """
    Tests the molecular optimization for a list of selected molecules.

    Args:
        selected_molecules (list): List of selected molecule objects.

    Returns:
        None
    """

    for selected_molecule in selected_molecules:
        symbols = selected_molecule.symbols
        coordinates = selected_molecule.coordinates
        charge = selected_molecule.charge
        mult = selected_molecule.mult
        basis_name = selected_molecule.basis_name

        x_init = np.array(coordinates)
        hamiltonian = build_hamiltonian(x_init, symbols, charge, mult, basis_name)
        exact_energy = compute_exact_energy(hamiltonian)
        print(f"Exact Energy (FCI): {exact_energy:.8f} Ha")

        electrons, spin_orbitals = initialize_molecule(symbols, x_init, charge, mult, basis_name)

        results = optimize_molecule(symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name)
        
        visualize_results(results, symbols)
