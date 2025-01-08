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

def load_molecules():
    filepath = 'config/molecules.json'
    if not os.path.exists(filepath):
        print(f"File '{filepath}' not found. Creating an empty file.")
        with open(filepath, 'w') as f:
            json.dump({}, f)
    with open(filepath, 'r') as f:
        return json.load(f)

def add_new_molecule():
    name = input("Molecule name (unique identifier): ").strip()
    syms = [s.strip() for s in input("Atomic symbols (e.g. H,H,O): ").split(',')]
    coords = []
    for i, s in enumerate(syms):
        xyz = input(f"Coordinates of {s} (x,y,z): ").split(',')
        coords.extend([float(v.strip()) for v in xyz])
    mult = int(input("Multiplicity (1=singlet,2=doublet,...): ").strip())
    chg = int(input("Charge of the molecule: ").strip())
    basis = input("Basis set (e.g. sto-3g): ").strip()

    data = load_molecules()
    if name in data:
        if input(f"'{name}' exists. Overwrite? (y/n): ").lower() != 'y':
            sys.exit(0)
    data[name] = {
        'symbols': syms,
        'coordinates': coords,
        'charge': chg,
        'mult': mult,
        'basis_name': basis,
    }
    with open('molecules.json', 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Molecule '{name}' added.")


def build_optimizers(args, ansatz_list, optimizer_map, predefined_steps):
    optimizers, new_ans = {}, []
    if args.all_optimizers:
        all_opts = list(optimizer_map.keys())
        user_steps = (args.stepsize != [0.4] or len(args.stepsize) > 1)
        for opt in all_opts:
            steps_for_opt = args.stepsize if user_steps else [predefined_steps[opt]]
            for step in steps_for_opt:
                for n in args.opt_step:
                    for ans_type, layer in ansatz_list:
                        name = f"{opt}_{step}_{ans_type}_{layer}layers_{n}steps"
                        optimizers[name] = optimizer_map[opt](stepsize=step)
                        # Guardamos la tupla con la info necesaria
                        new_ans.append((ans_type, layer, n))
    elif args.optimizer:
        for opt in args.optimizer:
            if opt not in optimizer_map:
                print(f"Error: Optimizer '{opt}' not recognized.")
                sys.exit(1)
            for step in args.stepsize:
                for n in args.opt_step:
                    for ans_type, layer in ansatz_list:
                        name = f"{opt}_{step}_{ans_type}_{layer}layers_{n}steps"
                        optimizers[name] = optimizer_map[opt](stepsize=step)
                        new_ans.append((ans_type, layer, n))
    else:
        # Si no se especifica --optimizer ni --all_optimizers, usar por defecto NesterovMomentum
        for step in args.stepsize:
            for n in args.opt_step:
                for ans_type, layer in ansatz_list:
                    name = f"NMomentum_{step}_{ans_type}_{layer}layers_{n}steps"
                    optimizers[name] = NesterovMomentumOptimizer(stepsize=step)
                    new_ans.append((ans_type, layer, n))
    return optimizers, new_ans


def from_user_input():
    parser = argparse.ArgumentParser(description='Quantum simulation of molecules using VQE.')
    parser.add_argument('--molecule', type=str, nargs='+', help='Molecule(s) to simulate.')
    parser.add_argument('--basis_set', type=str, default='sto-3g', help='Basis set.')
    parser.add_argument('--all_optimizers', action='store_true', help='Use all available optimizers.')
    parser.add_argument('--optimizer', type=str, nargs='+', help='Optimizers to use (Adam, Adagrad, etc.).')
    parser.add_argument('--ansatz', type=str, nargs='+', help='Ansatz to use (e.g. uccsd, vqe_classic).')
    parser.add_argument('--stepsize', type=float, nargs='+', default=[0.4], help='Optimizer step size(s).')
    parser.add_argument('--opt_step', type=int, nargs='+', default=[10], help='Number of steps for each operator selection block.')
    parser.add_argument('--add_molecule', action='store_true', help='Add a new molecule.')
    args = parser.parse_args()

    if args.add_molecule:
        add_new_molecule()
        sys.exit(0)

    optimizer_map = {
        "Adam": AdamOptimizer,
        "Adagrad": AdagradOptimizer,
        "NMomentum": NesterovMomentumOptimizer,
        "Momentum": MomentumOptimizer,
        "RMSProp": RMSPropOptimizer,
        "GD": GradientDescentOptimizer,
        "QNG": QNGOptimizer
    }

    predefined_steps = {
        'Adam': 0.1,
        'Adagrad': 0.1,
        'NMomentum': 0.2,
        'Momentum': 0.3,
        'RMSProp': 0.1,
        'GD': 0.3,
        'QNG': 0.1
    }

    # Construimos una lista de tuplas (ans_type, layer, nsteps)
    # Ajusta estos valores por defecto a tu criterio
    ans_list = []
    if args.ansatz:
        for ans_str in args.ansatz:
            ans_str_lower = ans_str.lower()
            if ans_str_lower == "uccsd":
                # Ejemplo: capa = 0 (no aplica)
                ans_list.append(("uccsd", 0))
            elif ans_str_lower == "vqe_classic":
                for l in [10, 15, 20, 25]:
                    ans_list.append(("vqe_classic", l))
            else:
                print(f"Warning: ansatz '{ans_str}' not recognized; using defaults.")
                ans_list.append(("uccsd", 0))
    else:
        # Si no se especifica nada, por defecto uccsd
        ans_list = [("uccsd", 0)]

    optimizers, ans_list = build_optimizers(args, ans_list, optimizer_map, predefined_steps)

    data = load_molecules()
    molecules = []
    if args.molecule:
        for name in args.molecule:
            if name not in data:
                print(f"Molecule '{name}' not defined. Use '--add_molecule' to add it.")
                sys.exit(1)
            mol = data[name]
            q_mol = qml.qchem.Molecule(
                symbols=mol['symbols'],
                coordinates=np.array(mol['coordinates']),
                basis_name=mol.get('basis_name', args.basis_set),
                charge=mol.get('charge', 0),
                mult=mol.get('mult', 1)
            )
            molecules.append(q_mol)
    else:
        print("Select a molecule to simulate:")
        keys = list(data.keys())
        for i, k in enumerate(keys, 1):
            print(f"{i}. {k}")
        print(f"{len(keys) + 1}. Exit")
        while True:
            try:
                c = int(input(f"Choose (1-{len(keys)+1}): "))
                if 1 <= c <= len(keys):
                    sel = keys[c - 1]
                    m = data[sel]
                    print(f"You selected: {sel}")
                    q_mol = qml.qchem.Molecule(
                        symbols=m['symbols'],
                        coordinates=np.array(m['coordinates']),
                        basis_name=m.get('basis_name', args.basis_set),
                        charge=m.get('charge', 0),
                        mult=m.get('mult', 1)
                    )
                    molecules.append(q_mol)
                    break
                elif c == len(keys) + 1:
                    print("Exiting.")
                    return
            except ValueError:
                print("Invalid input.")

    return molecules, optimizers, ans_list
