# config_functions.py
import argparse
import pennylane as qml
import sys
import json
import numpy as np
import os
import subprocess
import datetime

def from_user_input():
    """
    Creates a list of molecules from user input using PennyLane's Molecule class.

    Returns:
        (list, args, input_type): A list of PennyLane Molecule instances, the arguments, and the input type.
    """
    parser = argparse.ArgumentParser(description='Quantum simulation of molecules and chemical reactions using VQE.')
    
    parser.add_argument('--molecule', type=str, nargs='+', help='Molecule(s) to simulate (multiple molecules can be separated by spaces).')
    parser.add_argument('--reaction', action='store_true', help='Simulate a chemical reaction.')
    parser.add_argument('--opt', type=str, nargs='+', help='Molecule(s) to optimize (multiple molecules can be separated by spaces).')
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
    parser.add_argument('--scan_coordinate', type=str, help='Coordinate to scan (e.g., "bond_length").')
    parser.add_argument('--scan_values', type=float, nargs=3, metavar=('START', 'STOP', 'STEP'),
                        help='Values for scanning the coordinate: start, stop, step')

    args = parser.parse_args()

    # If the user wants to add a new molecule
    if args.add_molecule:
        add_new_molecule()  # Method that handles adding a new molecule
        sys.exit(0)  # Exit the program after adding the molecule

    # Determine if the input is a molecule or a reaction
    if args.reaction:
        input_type = 'reaction'
    elif args.molecule:
        input_type = 'molecule'
        molecule_names = args.molecule
    elif args.opt:
        input_type = 'optimization'
        molecule_names = args.opt
    else:
        print("Error: You must specify a type simulation.")
        sys.exit(1)
    
    # If the input is a reaction
    if args.reaction:
        return [], args, input_type

    # Load the defined molecules from a JSON file
    molecules_data = load_molecules()  # Method that loads molecules from the JSON file

    molecules = []
    for molecule_name in molecule_names:
        # Check if the specified molecule exists in the loaded database
        if molecule_name not in molecules_data:
            print(f"Molecule '{molecule_name}' is not defined. Use '--add_molecule' to add it.")
            sys.exit(1)  # Terminate the program if the molecule is not defined

        # Get the data of the specified molecule
        molecule_data = molecules_data[molecule_name]

        # Create the molecule using PennyLane
        molecule = qml.qchem.Molecule(
            symbols=molecule_data['symbols'],
            coordinates=np.array(molecule_data['coordinates']),
            basis_name=molecule_data.get('basis_set', args.basis_set),
            charge=molecule_data.get('charge', 0),
            mult=molecule_data.get('multiplicity', 1)
        )
        molecule.basis_name = molecule_data.get('basis_set', args.basis_set)
        molecule.active_electrons = molecule_data.get('active_electrons', molecule.n_electrons)
        molecule.active_orbitals = molecule_data.get('active_orbitals', molecule.n_orbitals)
        molecules.append(molecule)

    # Return the list of molecules, the user-provided arguments, and the input type
    return molecules, args, input_type


def add_new_molecule():
    """
    Adds a new molecule by interacting with the user.
    """
    print("Add a new molecule to the system.")
    name = input("Molecule name (unique identifier): ").strip()
    symbols_input = input("Atomic symbols separated by commas (e.g., H,H,O): ").strip().split(',')
    symbols = [s.strip() for s in symbols_input]
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
        multiplicity = int(input("Multiplicity (1 for singlet, 2 for doublet, etc.): ").strip())
        charge = int(input("Charge of the molecule: ").strip())
        if active_electrons <= 0 or active_orbitals <= 0 or multiplicity <= 0:
            print("Error: Number of active electrons, orbitals, and multiplicity must be positive integers.")
            sys.exit(1)
    except ValueError:
        print("Error: You must enter integer numbers for active electrons, orbitals, multiplicity, and charge.")
        sys.exit(1)

    # Create the molecule dictionary
    molecule = {
        'symbols': symbols,
        'coordinates': coordinates,
        'active_electrons': active_electrons,
        'active_orbitals': active_orbitals,
        'multiplicity': multiplicity,
        'charge': charge
    }

    # Load existing molecules
    molecules = load_molecules()

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
    try:
        with open('molecules.json', 'w') as f:
            json.dump(molecules, f, indent=4)
        print(f"Molecule '{name}' successfully added.")
    except Exception as e:
        print(f"Error saving the molecule: {e}")
        sys.exit(1)


def load_molecules():
    """
    Load molecules from the JSON file.
    """
    molecules_file = 'molecules.json'
    if not os.path.exists(molecules_file):
        print(f"The file '{molecules_file}' does not exist. Creating an empty file.")
        try:
            with open(molecules_file, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error creating the molecules file: {e}")
            sys.exit(1)
    try:
        with open(molecules_file, 'r') as f:
            molecules = json.load(f)
    except Exception as e:
        print(f"Error loading the molecules file: {e}")
        sys.exit(1)
    return molecules


def get_results_directory(input_type, args):
    """
    Retrieves the directory where results will be saved, based on Git version, process type, and timestamp.

    Args:
        input_type (str): The type of process ('molecule' or 'reaction').
        args (Namespace): Parsed arguments from the user.

    Returns:
        results_dir (str): Path to the results directory.
    """
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M')

    if input_type == 'molecule':
        process_name = args.molecule.replace(' ', '_')
    else:
        process_name = args.reaction if args.reaction else 'reaction'

    results_dir = os.path.join('results', input_type, process_name, timestamp)
    return results_dir