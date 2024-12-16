import os
import time
from pennylane import numpy as np
import pennylane as qml
from pennylane.optimize import (
    GradientDescentOptimizer, 
    QNGOptimizer, 
    AdamOptimizer, 
    AdagradOptimizer, 
    NesterovMomentumOptimizer, 
    MomentumOptimizer, 
    RMSPropOptimizer
)
from tabulate import tabulate
from .visualizer import write_simulation_times
from .hamiltonian_builder import build_hamiltonian, generate_hf_state, get_operator_pool
from .ansatz_preparer import prepare_ansatz, compute_operator_gradients, select_operator
import scipy.sparse.linalg


# Constants
MAX_ITER = 20     # Maximum number of main iterations
CONV = 1e-8       # Convergence criterion based on energy difference

def compute_exact_energy(symbols, coordinates, charge, mult, basis_name='sto-3g'):
    """
    Compute the FCI (exact) ground state energy by constructing the molecular Hamiltonian
    and then performing exact diagonalization using sparse linear algebra methods.

    Args:
        symbols (list): Atomic symbols of the molecule.
        coordinates (array): Flattened array of atomic coordinates [x1, y1, z1, x2, y2, z2, ...].
        charge (int): Molecular charge.
        mult (int): Spin multiplicity.
        basis_name (str): Basis set name.

    Returns:
        float: The exact ground state (FCI) energy in Hartrees.
    """
    coordinates = np.array(coordinates)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates.reshape(-1, 3), charge=charge, mult=mult, basis=basis_name
    )

    sparse_hamiltonian= hamiltonian.sparse_matrix()
    eigenvalues, _ = scipy.sparse.linalg.eigsh(sparse_hamiltonian, k=1, which='SA')
    exact_energy = float(eigenvalues[0])
    return exact_energy

def compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Compute energy gradients with respect to nuclear coordinates.
    """
    delta = 1e-3
    num_coords = len(x)
    grad_x = np.zeros_like(x)

    for i in range(num_coords):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        h_plus = build_hamiltonian(x_plus, symbols, charge, mult, basis_name)
        h_minus = build_hamiltonian(x_minus, symbols, charge, mult, basis_name)

        @qml.qnode(dev, interface=interface)
        def cost_fn_plus(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals)
            return qml.expval(h_plus)

        @qml.qnode(dev, interface=interface)
        def cost_fn_minus(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals)
            return qml.expval(h_minus)

        energy_plus = cost_fn_plus(params)
        energy_minus = cost_fn_minus(params)
        grad_x[i] = (energy_plus - energy_minus) / (2 * delta)

    return grad_x

def update_parameters_and_coordinates(opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, convergence, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Update parameters and nuclear coordinates for 10 steps.
    """
    energy_history = []
    x_history = []

    for opt_step in range(10):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = np.real(energy)
        energy_history.append(energy)
        x_history.append(np.array(x))

        grad_x = compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface, charge, mult, basis_name)
        x = x - learning_rate_x * grad_x

    return params, x, energy_history, x_history, opt_state

def run_optimization_uccsd(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, operator_pool_copy, selected_excitations, opt, opt_state, max_iterations, learning_rate_x, convergence):
    """
    Optimize with a UCCSD ansatz.
    Convergence: if abs(energy_diff) < CONV between consecutive main iterations.
    """
    execution_times = {
        'build_hamiltonian': 0.0,
        'compute_operator_gradients': 0.0,
        'update_parameters_and_coordinates': 0.0,
        'Total Time': 0.0
    }

    energy_history_total = []
    x_history_total = []
    params_history = []
    optimizer_start_time = time.time()

    def cost_fn_factory(hamiltonian):
        @qml.qnode(dev, interface=interface)
        def cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type="uccsd")
            return qml.expval(hamiltonian)
        return cost_fn

    prev_energy = None

    for iteration in range(max_iterations):
        iter_start_time = time.time()

        start_time = time.time()
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        end_time = time.time()
        execution_times['build_hamiltonian'] += end_time - start_time

        cost_fn = cost_fn_factory(hamiltonian)

        start_time = time.time()
        gradients = compute_operator_gradients(operator_pool_copy, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals, ansatz_type="uccsd")
        end_time = time.time()
        execution_times['compute_operator_gradients'] += end_time - start_time

        selected_gate, max_grad_value = select_operator(gradients, operator_pool_copy, convergence)
        if selected_gate is None:
            print("No operators selected. Stopping optimization for uccsd.")
            break

        selected_excitations.append(selected_gate)
        params = np.append(params, 0.0)
        params = np.array(params, requires_grad=True)
        opt = type(opt)(stepsize=learning_rate_x)

        start_time = time.time()
        params, x, energy_history, x_history, opt_state = update_parameters_and_coordinates(
            opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, 
            spin_orbitals, learning_rate_x, convergence, interface, charge, mult, basis_name
        )
        end_time = time.time()
        execution_times['update_parameters_and_coordinates'] += end_time - start_time

        energy_history_total.extend(energy_history)
        x_history_total.extend(x_history)
        params_history.append(params.copy())

        current_energy = energy_history[-1]
        print(f"Iteration {iteration + 1}, Energy = {current_energy:.8f} Ha, Max Gradient = {max_grad_value:.5e}")

        # Convergence check: difference in final energies
        if prev_energy is not None:
            if abs(prev_energy - current_energy) < 1e-8:
                print("Convergence reached: Energy difference < 1e-8")
                break
        prev_energy = current_energy

        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (uccsd): {total_time:.2f} seconds")

    return params, x, energy_history_total, x_history_total, params_history, execution_times

def run_optimization_vqe_classic(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, selected_excitations, opt, opt_state, max_iterations, learning_rate_x, convergence):
    """
    Optimize a VQE hardware-efficient ansatz.
    Convergence: difference in final energies < 1e-8.
    """
    execution_times = {
        'build_hamiltonian': 0.0,
        'compute_operator_gradients': 0.0,
        'update_parameters_and_coordinates': 0.0,
        'Total Time': 0.0
    }

    energy_history_total = []
    x_history_total = []
    params_history = []
    optimizer_start_time = time.time()

    def cost_fn_factory(hamiltonian):
        @qml.qnode(dev, interface=interface)
        def cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type="vqe_classic")
            return qml.expval(hamiltonian)
        return cost_fn

    prev_energy = None

    for iteration in range(max_iterations):
        iter_start_time = time.time()

        start_time = time.time()
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        end_time = time.time()
        execution_times['build_hamiltonian'] += end_time - start_time

        cost_fn = cost_fn_factory(hamiltonian)

        start_time = time.time()
        params, x, energy_history, x_history, opt_state = update_parameters_and_coordinates(
            opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals,
            learning_rate_x, convergence, interface, charge, mult, basis_name
        )
        end_time = time.time()
        execution_times['update_parameters_and_coordinates'] += end_time - start_time

        energy_history_total.extend(energy_history)
        x_history_total.extend(x_history)
        params_history.append(params.copy())

        current_energy = energy_history[-1]
        print(f"Iteration {iteration + 1}, Energy = {current_energy:.8f} Ha")

        # Convergence check
        if prev_energy is not None:
            if abs(prev_energy - current_energy) < 1e-8:
                print("Convergence reached: Energy difference < 1e-8")
                break
        prev_energy = current_energy

        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (vqe_classic): {total_time:.2f} seconds")

    return params, x, energy_history_total, x_history_total, params_history, execution_times

def optimize_molecule(symbols, x_init, electrons, spin_orbitals, optimizers, charge=0, mult=1, basis_name='sto-3g', ansatz_type=["uccsd"]):
    """
    Optimize the molecule using VQE. The exact FCI energy is computed for reference.
    Convergence based on energy difference between consecutive main iterations.
    """
    interface = 'autograd'
    print(f"\n===== Starting optimization with interface: {interface} =====\n")
    hf_state = generate_hf_state(electrons, spin_orbitals)
    dev = qml.device("default.qubit", wires=spin_orbitals)
    operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')

    exact_energy = compute_exact_energy(symbols, x_init, charge, mult, basis_name)
    print(f"Exact Energy (FCI): {exact_energy:.8f} Ha")

    interface_results = {}
    cont = 0
    convergence = CONV
    max_iterations = MAX_ITER

    for optimizer_name, opt in optimizers.items():
        print(f"\n--- Optimizing with {optimizer_name} ---")
        ansatz_type_opt = ansatz_type[cont]
        cont += 1
        learning_rate_x = opt.stepsize

        operator_pool_copy = operator_pool.copy()
        selected_excitations = []
        x = x_init.copy()

        if ansatz_type_opt == "vqe_classic":
            # For vqe_classic: random initialization
            num_layers = 30
            params_per_layer = 2 * spin_orbitals
            total_params = num_layers * params_per_layer
            params = 0.01 * np.random.randn(total_params)
            params = np.array(params, requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
             ) = run_optimization_vqe_classic(
                symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                dev, interface, selected_excitations,
                opt, None, max_iterations, learning_rate_x, convergence
             )
        else:
            # uccsd
            params = np.array([], requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
             ) = run_optimization_uccsd(
                symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                dev, interface, operator_pool_copy, selected_excitations,
                opt, None, max_iterations, learning_rate_x, convergence
             )

        final_energy = energy_history_total[-1] if energy_history_total else None

        interface_results[optimizer_name] = {
            "energy_history": energy_history_total,
            "x_history": x_history_total,
            "params_history": params_history,
            "final_energy": final_energy,
            "final_params": params,
            "final_x": x,
            "interface": interface,
            "total_time": execution_times['Total Time'],
            "execution_times": execution_times,
            "exact_energy_reference": exact_energy
        }

        if final_energy is not None:
            diff = final_energy - exact_energy
            print(f"\nFinal energy with {optimizer_name} ({interface}) = {final_energy:.8f} Ha")
            print(f"Difference from exact (FCI) energy: {diff:.8e} Ha")
        else:
            print(f"\nNo final energy obtained with {optimizer_name} ({interface})")

        final_x = x
        print(f"\nFinal geometry with {optimizer_name} ({interface}):")
        atom_coords = []
        final_x_np = np.array(final_x)
        for i, atom in enumerate(symbols):
            atom_coords.append([atom, final_x_np[3 * i], final_x_np[3 * i + 1], final_x_np[3 * i + 2]])
        print(tabulate(atom_coords, headers=["Symbol", "x (Å)", "y (Å)", "z (Å)"], floatfmt=".6f"))

        # Show final circuit
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        @qml.qnode(dev, interface=interface)
        def final_cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type=ansatz_type_opt)
            return qml.expval(hamiltonian)

        print(f"Quantum Circuit with {optimizer_name} ({interface}):\n")
        print(qml.draw(final_cost_fn)(params))

        write_simulation_times(symbols, interface, optimizer_name, execution_times)

    results = {interface: interface_results}

    print("\n=== Total Optimization Times ===")
    for interface, interface_results in results.items():
        print(f"\nInterface: {interface}")
        for optimizer_name, data in interface_results.items():
            total_time = data.get("total_time", 0)
            print(f"Optimizer: {optimizer_name}, Time: {total_time:.2f} seconds")

    return results
