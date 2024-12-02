import pennylane as qml
from pennylane import numpy as np

def build_hamiltonian(x, symbols, charge=0, mult=1, basis_name='sto-3g'):
    """
    Constructs the molecular Hamiltonian for the coordinates x.

    Source:
    https://pennylane.ai/qml/demos/qchem_hydrogen_molecule.html

    Args:
        x (np.ndarray): Array of current coordinates.
        symbols (list): List of atomic symbols.
        charge (int, optional): Charge of the molecule.
        mult (int, optional): Spin multiplicity.
        basis_name (str, optional): Name of the atomic basis.

    Returns:
        hamiltonian (qml.Hamiltonian): Molecular Hamiltonian.
    """
    x = np.array(x)
    coordinates = x.reshape(-1, 3)

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates, charge=charge, mult=mult, basis=basis_name
    )
    h_coeffs, h_ops = hamiltonian.terms()

    h_coeffs = np.array(h_coeffs)

    hamiltonian = qml.Hamiltonian(h_coeffs, h_ops)
    return hamiltonian


def generate_hf_state(electrons, spin_orbitals):
    """
    Generates the Hartree-Fock reference state.

    Source:
    https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html

    Args:
        electrons (int): Total number of electrons.
        spin_orbitals (int): Number of spin orbitals.

    Returns:
        hf_state (np.ndarray): Hartree-Fock state.
    """

    hf_state = qml.qchem.hf_state(electrons, spin_orbitals)
    print(hf_state)
    return hf_state

def get_operator_pool(electrons, spin_orbitals, excitation_level='both'):
    """
    Obtains single and double excitations to form the operator pool.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_circuits/

    Args:
        electrons (int): Total number of electrons.
        spin_orbitals (int): Number of spin orbitals.
        excitation_level (str, optional): Level of excitation ('single', 'double', 'both').

    Returns:
        operator_pool (list): List of excitations.
    """

    singles, doubles = qml.qchem.excitations(electrons, spin_orbitals)
    if excitation_level == 'single':
        operator_pool = singles
    elif excitation_level == 'double':
        operator_pool = doubles
    elif excitation_level == 'both':
        operator_pool = singles + doubles
    else:
        raise ValueError("The excitation level must be 'single', 'double', or 'both'.")
    print(f"Number of {excitation_level} excitations: {len(operator_pool)}")

    operator_pool = [tuple(exc) for exc in operator_pool]
    return operator_pool
def compute_exact_energy(hamiltonian):
    """
    Calculates the exact (FCI) energy of the molecular Hamiltonian.

    Args:
        hamiltonian (qml.Hamiltonian): Molecular Hamiltonian.

    Returns:
        exact_energy (float): Exact energy (FCI) in Hartrees.
    """

    H = hamiltonian.matrix()
    eigenvalues = np.linalg.eigvalsh(H)
    exact_energy = eigenvalues[0]
    return exact_energy