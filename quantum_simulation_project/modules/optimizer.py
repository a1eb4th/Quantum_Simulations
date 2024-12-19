import os
import time
import concurrent.futures
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

MAX_ITER = 10
CONV = 1e-9
CONSECUTIVE_CONV = 3  # number of consecutive small improvements required to declare convergence

def compute_exact_energy(symbols, coordinates, charge, mult, basis_name='sto-3g'):
    """
    Compute FCI (exact) energy using sparse diagonalization.
    """
    coordinates = np.array(coordinates)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates.reshape(-1, 3), charge=charge, mult=mult, basis=basis_name
    )
    sparse_hamiltonian= hamiltonian.sparse_matrix()
    eigenvalues, _ = scipy.sparse.linalg.eigsh(sparse_hamiltonian, k=1, which='SA')
    return float(eigenvalues[0])

def compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Compute nuclear gradients via finite differences.
    """
    delta = 1e-3
    grad_x = np.zeros_like(x)
    for i in range(len(x)):
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

def check_convergence(energy, prev_energy, recent_diffs):
    """
    Check convergence based on recent energy differences.
    We store the absolute difference and check if we have had several consecutive small differences.
    """
    if prev_energy is not None:
        diff = abs(energy - prev_energy)
        recent_diffs.append(diff)
        if len(recent_diffs) > CONSECUTIVE_CONV:
            recent_diffs.pop(0)
        # If we have CONSECUTIVE_CONV consecutive differences below threshold, consider it converged
        if len(recent_diffs) == CONSECUTIVE_CONV and all(d < CONV for d in recent_diffs):
            return True
    return False

def update_parameters_and_coordinates(opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, interface, charge, mult, basis_name, prev_energy, recent_diffs):
    """
    Update parameters and geometry for 10 steps, checking convergence after each update.
    """
    energy_history = []
    x_history = []
    converged = False
    for opt_step in range(10):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = np.real(energy)
        energy_history.append(energy)
        x_history.append(np.array(x))

        grad_x = compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface, charge, mult, basis_name)
        x = x - learning_rate_x * grad_x

        # Check convergence after each step
        if check_convergence(energy, prev_energy, recent_diffs):
            print(f"Convergence reached: Energy difference < {CONV} for several consecutive steps")
            converged = True
            prev_energy = energy
            break

        prev_energy = energy

    return params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs

def run_optimization_uccsd(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, operator_pool_copy, selected_excitations, opt, opt_state, max_iterations, learning_rate_x):
    execution_times = {'build_hamiltonian':0.0,'compute_operator_gradients':0.0,'update_parameters_and_coordinates':0.0,'Total Time':0.0}
    energy_history_total = []
    x_history_total = []
    params_history = []
    recent_diffs = []
    prev_energy = None

    optimizer_start_time = time.time()

    def cost_fn_factory(hamiltonian):
        @qml.qnode(dev, interface=interface)
        def cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type="uccsd")
            return qml.expval(hamiltonian)
        return cost_fn

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

        selected_gate, max_grad_value = select_operator(gradients, operator_pool_copy, CONV)
        if selected_gate is None:
            print("No operators selected. Stopping optimization for uccsd.")
            break
        selected_excitations.append(selected_gate)
        params = np.append(params, 0.0)
        params = np.array(params, requires_grad=True)
        opt = type(opt)(stepsize=learning_rate_x)

        start_time = time.time()
        params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs = update_parameters_and_coordinates(
            opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals,
            learning_rate_x, interface='autograd', charge=charge, mult=mult, basis_name=basis_name,
            prev_energy=prev_energy, recent_diffs=recent_diffs
        )
        end_time = time.time()
        execution_times['update_parameters_and_coordinates'] += end_time - start_time

        energy_history_total.extend(energy_history)
        x_history_total.extend(x_history)
        params_history.append(params.copy())
        current_energy = energy_history[-1]

        print(f"Iteration {iteration + 1}, Energy = {current_energy:.8f} Ha, Max Gradient = {max_grad_value:.5e}")

        if converged:
            break

        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (uccsd): {total_time:.2f} seconds")
    return params, x, energy_history_total, x_history_total, params_history, execution_times

def run_optimization_vqe_classic(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, selected_excitations, opt, opt_state, max_iterations, learning_rate_x):
    execution_times = {'build_hamiltonian':0.0,'compute_operator_gradients':0.0,'update_parameters_and_coordinates':0.0,'Total Time':0.0}
    energy_history_total = []
    x_history_total = []
    params_history = []
    recent_diffs = []
    prev_energy = None

    optimizer_start_time = time.time()

    def cost_fn_factory(hamiltonian):
        @qml.qnode(dev, interface=interface)
        def cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type="vqe_classic")
            return qml.expval(hamiltonian)
        return cost_fn

    for iteration in range(max_iterations):
        iter_start_time = time.time()
        start_time = time.time()
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        end_time = time.time()
        execution_times['build_hamiltonian'] += end_time - start_time

        cost_fn = cost_fn_factory(hamiltonian)
        start_time = time.time()
        params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs = update_parameters_and_coordinates(
            opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals,
            learning_rate_x, interface='autograd', charge=charge, mult=mult, basis_name=basis_name,
            prev_energy=prev_energy, recent_diffs=recent_diffs
        )
        end_time = time.time()
        execution_times['update_parameters_and_coordinates'] += end_time - start_time

        energy_history_total.extend(energy_history)
        x_history_total.extend(x_history)
        params_history.append(params.copy())
        current_energy = energy_history[-1]
        print(f"Iteration {iteration + 1}, Energy = {current_energy:.8f} Ha")

        if converged:
            break

        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (vqe_classic): {total_time:.2f} seconds")
    return params, x, energy_history_total, x_history_total, params_history, execution_times

def run_single_optimizer(optimizer_name, opt, ansatz_type_opt, symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name, hf_state, dev, operator_pool, exact_energy, results_dir):
    """
    Run a single optimizer. Since we are dealing with a single molecule scenario,
    we do not show the molecule information here again. Just the optimization details.
    """
    output_file = os.path.join(results_dir, f"output_{optimizer_name}.txt")
    with open(output_file, "w", buffering=1) as f:
        orig_stdout = os.sys.stdout
        os.sys.stdout = f

        interface = 'autograd'
        learning_rate_x = opt.stepsize
        operator_pool_copy = operator_pool.copy()
        selected_excitations = []
        x = x_init.copy()

        print(f"\n--- Optimizing with {optimizer_name} ---\n")

        if ansatz_type_opt == "vqe_classic":
            num_layers = 30
            params_per_layer = 2 * spin_orbitals
            total_params = num_layers * params_per_layer
            params = 0.01 * np.random.randn(total_params)
            params = np.array(params, requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
                ) = run_optimization_vqe_classic(
                    symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                    dev, interface, selected_excitations, opt, None, MAX_ITER, learning_rate_x
                )
        else:
            params = np.array([], requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
                ) = run_optimization_uccsd(
                    symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                    dev, interface, operator_pool_copy, selected_excitations, opt, None, MAX_ITER, learning_rate_x
                )

        final_energy = energy_history_total[-1] if energy_history_total else None
        os.sys.stdout = orig_stdout

    result = {
        "energy_history": energy_history_total,
        "x_history": x_history_total,
        "params_history": params_history,
        "final_energy": final_energy,
        "final_params": params,
        "final_x": x,
        "interface": 'autograd',
        "exact_energy_reference": exact_energy,
        "execution_times": execution_times,
        "selected_excitations": selected_excitations
    }
    return optimizer_name, result

def optimize_molecule(symbols, x_init, electrons, spin_orbitals, optimizers, charge=0, mult=1, basis_name='sto-3g', ansatz_type=["uccsd"]):
    """
    Optimize a single molecule scenario with improved convergence criteria and layout.
    If user wants multiple molecules scenario, they should adapt the code similarly.
    """
    results_dir = "temp_results_autograd"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    exact_energy = compute_exact_energy(symbols, x_init, charge, mult, basis_name)
    hf_state = generate_hf_state(electrons, spin_orbitals)
    dev = qml.device("default.qubit", wires=spin_orbitals)
    operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')
    futures = []
    interface_results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        cont = 0
        for optimizer_name, opt in optimizers.items():
            ansatz_type_opt = ansatz_type[cont] if cont < len(ansatz_type) else "uccsd"
            cont += 1
            futures.append(
                executor.submit(
                    run_single_optimizer,
                    optimizer_name, opt, ansatz_type_opt, symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name,
                    hf_state, dev, operator_pool, exact_energy, results_dir
                )
            )
        for future in concurrent.futures.as_completed(futures):
            optimizer_name, data = future.result()
            interface_results[optimizer_name] = data

    # Concatenate outputs and remove temp files
    sorted_keys = list(optimizers.keys())
    for optimizer_name in sorted_keys:
        output_file = os.path.join(results_dir, f"output_{optimizer_name}.txt")
        with open(output_file, "r") as f:
            content = f.read()
        print(content, end="")
        os.remove(output_file)

        print()
        final_energy = interface_results[optimizer_name]["final_energy"]
        exact_energy_ref = interface_results[optimizer_name]["exact_energy_reference"]
        diff = final_energy - exact_energy_ref if final_energy is not None else None
        if final_energy is not None:
            print(f"Final energy with {optimizer_name} (autograd) = {final_energy:.8f} Ha")
            print(f"Difference from exact (FCI) energy: {diff:.8e} Ha\n")
        else:
            print(f"No final energy obtained with {optimizer_name} (autograd)\n")

        final_x = interface_results[optimizer_name]["final_x"]
        atoms_coords = []
        final_x_np = np.array(final_x)
        for i, atom in enumerate(symbols):
            atoms_coords.append([atom, final_x_np[3*i], final_x_np[3*i+1], final_x_np[3*i+2]])
        print("Final geometry:")
        print(tabulate(atoms_coords, headers=["Symbol", "x (Å)", "y (Å)", "z (Å)"], floatfmt=".6f"))
        print()

        hamiltonian = build_hamiltonian(final_x, symbols, charge, mult, basis_name)
        selected_excitations = interface_results[optimizer_name]["selected_excitations"]
        @qml.qnode(dev, interface='autograd')
        def final_cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type="uccsd")
            return qml.expval(hamiltonian)

        params = interface_results[optimizer_name]["final_params"]
        print("Quantum Circuit:\n")
        print(qml.draw(final_cost_fn)(params))
        print()

    print("=== Total Optimization Times ===\n")
    print("Interface: autograd")
    for optimizer_name in sorted_keys:
        total_time = interface_results[optimizer_name]["execution_times"].get('Total Time', 0)
        print(f"Optimizer: {optimizer_name}, Time: {total_time:.2f} seconds")

    profiler_output_path = os.path.join(results_dir, "profile_output_autograd.txt")
    filtered_report_path = os.path.join(results_dir, "filtered_report_autograd.txt")
    print(f"Report completely saved on: {profiler_output_path}")
    print(f"Filtered report saved on: {filtered_report_path}")

    return {"autograd": interface_results}

