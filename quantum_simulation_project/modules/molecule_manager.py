
import pennylane as qml

def initialize_molecule(symbols, x_init, charge=0, mult=1, basis_name='sto-3g'):
    """
    Initializes the molecule with the provided symbols and coordinates.

    Source:
    https://pennylane.ai/qml/demos/qchem_hydrogen_molecule.html

    Args:
        symbols (list): List of atomic symbols.
        x_init (np.ndarray): Array of initial coordinates.
        charge (int, optional): Charge of the molecule.
        mult (int, optional): Spin multiplicity.
        basis_name (str, optional): Name of the atomic basis.

    Returns:
        molecule (qml.qchem.Molecule): PennyLane Molecule object.
        electrons (int): Total number of electrons.
        spin_orbitals (int): Number of spin orbitals.
    """

    coordinates = x_init.reshape(-1, 3)
    molecule = qml.qchem.Molecule(symbols, coordinates, charge=charge, mult=mult, basis_name=basis_name)
    electrons = molecule.n_electrons
    n_orbitals = len(molecule.basis_set)
    spin_orbitals = 2 * n_orbitals

    print(f"\n--- Molecule Information ---")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Number of electrons: {electrons}")
    print(f"Number of orbitals: {n_orbitals}")
    print(f"Number of spin orbitals: {spin_orbitals}\n")

    return electrons, spin_orbitals
