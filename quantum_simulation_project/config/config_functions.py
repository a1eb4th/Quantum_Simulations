# config_functions.py
import argparse
import pennylane as qml
from pennylane.optimize import (
    AdamOptimizer, AdagradOptimizer, NesterovMomentumOptimizer, 
    MomentumOptimizer, RMSPropOptimizer, GradientDescentOptimizer, QNGOptimizer
)
import sys
import json
from pennylane import numpy as np
import os
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
    parser.add_argument('--opt', action= 'store_true', help='Molecule(s) to optimize (multiple molecules can be separated by spaces).')
    parser.add_argument('--basis_set', type=str, default='sto-3g', help='Basis set to use.')
    parser.add_argument('--optimizer', type=str, nargs='+', help='Optimizer(s) to use (use abbreviations: Adam, Adagrad, NMomentum, Momentum, RMSProp, GD, QNG).')
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of optimization iterations.')
    parser.add_argument('--ansatz', type=str, nargs='+', help='Ansatz to use (use abbreviations: uccsd, vqe ).')
    parser.add_argument('--conv_tol', type=float, default=1e-6, help='Convergence tolerance.')
    parser.add_argument('--stepsize', type=float, nargs='+', default=[0.4], help='Step size(s) for the optimizer. Provide multiple values separated by spaces.')

    parser.add_argument('--save', action='store_true', help='Save results to the results folder.')
    parser.add_argument('--save_dir', type=str, help='Name of the directory where results will be saved.')
    parser.add_argument('--plot', action='store_true', help='Show the energy convergence plot.')
    parser.add_argument('--add_molecule', action='store_true', help='Add a new molecule to the system.')
    parser.add_argument('--use_cached_results', action='store_true', help='Use cached results if available.')
    parser.add_argument('--scan_coordinate', type=str, help='Coordinate to scan (e.g., "bond_length").')
    parser.add_argument('--scan_values', type=float, nargs=3, metavar=('START', 'STOP', 'STEP'),
                        help='Values for scanning the coordinate: start, stop, step')

    args = parser.parse_args()
    
    optimizer_map = {
        "Adam": AdamOptimizer,
        "Adagrad": AdagradOptimizer,
        "NMomentum": NesterovMomentumOptimizer,
        "Momentum": MomentumOptimizer,
        "RMSProp": RMSPropOptimizer,
        "GD": GradientDescentOptimizer,
        "QNG": QNGOptimizer
    }
    if args.ansatz:
        ansatz_list = args.ansatz
    else:
        ansatz_list = ["uccsd"]
    
    if args.optimizer:
        optimizers = {}
        new_ansatz_list = []
        for opt in args.optimizer:
            if opt not in optimizer_map:
                print(f"Error: Optimizer '{opt}' is not recognized. Use one of: {', '.join(optimizer_map.keys())}")
                sys.exit(1)
            for step in args.stepsize:
                for ans in ansatz_list:
                    opt_name = f"{opt}_{step}_{ans}"
                    optimizers[opt_name] = optimizer_map[opt](stepsize=step)
                    new_ansatz_list.append(ans)
        ansatz_list = new_ansatz_list
    else:
        optimizers={}
        new_ansatz_list = []
        for step in args.stepsize:
            for ans in ansatz_list: 
                opt_name = f"NMomentum_{step}_{ans}"
                optimizers[opt_name] = NesterovMomentumOptimizer(stepsize=step)
                new_ansatz_list.append(ans)
        ansatz_list = new_ansatz_list

    # If the user wants to add a new molecule
    if args.add_molecule:
        add_new_molecule()  
        sys.exit(0)  



    predefined_molecules = load_molecules() 
    molecules = []

    if args.molecule:
        molecule_names = args.molecule
        for molecule_name in molecule_names:
            
            if molecule_name not in predefined_molecules:
                print(f"Molecule '{molecule_name}' is not defined. Use '--add_molecule' to add it.")
                sys.exit(1)  

            
            molecule_data = predefined_molecules[molecule_name]

            
            molecule = qml.qchem.Molecule(
                symbols=molecule_data['symbols'],
                coordinates=np.array(molecule_data['coordinates']),
                basis_name=molecule_data.get('basis_name', args.basis_set),
                charge=molecule_data.get('charge', 0),
                mult=molecule_data.get('multiplicity', 1)
            )
            molecules.append(molecule)
    else:
        
        print("Select a molecule to simulate:")
        for idx, molecule_name in enumerate(predefined_molecules.keys(), start=1):
            print(f"{idx}. {molecule_name}")
        print(f"{len(predefined_molecules) + 1}. Exit")
        
        
        while True:
            try:
                choice = int(input(f"\nEnter the number of the molecule to simulate (1-{len(predefined_molecules) + 1}): "))
                if 1 <= choice <= len(predefined_molecules):
                    molecule_names = list(predefined_molecules.keys())[choice - 1]
                    selected_molecule = predefined_molecules[molecule_names]
                    print(f"\nYou have selected: {molecule_names}\n")
                    break
                elif choice == len(predefined_molecules) + 1:
                    print("Exiting the program.")
                    return
                else:
                    print(f"Please choose a number between 1 and {len(predefined_molecules) + 1}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        molecule = qml.qchem.Molecule(
            symbols=selected_molecule['symbols'],
            coordinates=np.array(selected_molecule['coordinates']),
            basis_name=selected_molecule.get('basis_name', args.basis_set),
            charge=selected_molecule.get('charge', 0),
            mult=selected_molecule.get('multiplicity', 1)
        )
        molecules.append(molecule)

    return molecules, optimizers, ansatz_list


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
        multiplicity = int(input("Multiplicity (1 for singlet, 2 for doublet, etc.): ").strip())
        charge = int(input("Charge of the molecule: ").strip())
        basis_set = input("Basis set (e.g., sto-3g): ").strip()
    except ValueError:
        print("Error: You must enter integer numbers for active electrons, orbitals, multiplicity, and charge.")
        sys.exit(1)

    # Create the molecule dictionary
    molecule = {
        'symbols': symbols,
        'coordinates': coordinates,
        'charge': charge,
        'mult': multiplicity,
        'basis_name': basis_set,
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
    molecules_file = 'config/molecules.json'
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