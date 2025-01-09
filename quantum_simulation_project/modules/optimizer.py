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
import scipy.sparse.linalg

MAX_ITER = 10
CONV = 1e-8
CONSECUTIVE_CONV = 3

def compute_exact_energy(symbols, coordinates, charge, mult, basis_name='sto-3g'):
    """
    Computes the exact ground-state energy of a molecule using Full Configuration Interaction (FCI).

    This function constructs the molecular Hamiltonian for the given molecular geometry and computes
    its lowest eigenvalue using sparse linear algebra methods.

    Args:
        symbols (list of str): List of atomic symbols in the molecule (e.g., ['H', 'H']).
        coordinates (list or array-like): Atomic coordinates in Angstroms, flattened as [x1, y1, z1, x2, y2, z2, ...].
        charge (int): Total charge of the molecule.
        mult (int): Multiplicity of the molecule (e.g., 1 for singlet, 2 for doublet).
        basis_name (str, optional): Basis set to use for the calculation. Defaults to 'sto-3g'.

    Returns:
        float: The exact ground-state energy of the molecule in Hartree (Ha).
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
    Computes the gradients of the energy with respect to nuclear coordinates using finite differences.

    This function perturbs each nuclear coordinate slightly in the positive and negative directions,
    computes the corresponding energies, and estimates the gradient as the central difference.

    Args:
        params (array-like): Variational parameters for the ansatz.
        x (array-like): Current nuclear coordinates.
        symbols (list of str): List of atomic symbols in the molecule.
        selected_excitations (list of tuples): List of selected excitations for the ansatz.
        dev (qml.Device): Quantum device to execute the circuits.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        spin_orbitals (int): Number of spin orbitals in the system.
        interface (str, optional): Interface for PennyLane (e.g., 'autograd'). Defaults to 'autograd'.
        charge (int, optional): Total charge of the molecule. Defaults to 0.
        mult (int, optional): Multiplicity of the molecule. Defaults to 1.
        basis_name (str, optional): Basis set to use for the calculation. Defaults to 'sto-3g'.

    Returns:
        np.ndarray: Gradient of the energy with respect to each nuclear coordinate.
    """
    from .hamiltonian_builder import build_hamiltonian
    from .ansatz_preparer import prepare_ansatz
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
    This function monitors the recent differences between consecutive energy evaluations.
    If a specified number of consecutive differences are below the convergence threshold,
    the optimization is considered to have converged.

    Args:
        energy (float): Current energy value.
        prev_energy (float or None): Previous energy value.
        recent_diffs (list of float): List of recent energy differences.

    Returns:
        bool: True if convergence is achieved, False otherwise.
    """
    if prev_energy is not None:
        diff = abs(energy - prev_energy)
        recent_diffs.append(diff)
        if len(recent_diffs) > CONSECUTIVE_CONV:
            recent_diffs.pop(0)
        if len(recent_diffs) == CONSECUTIVE_CONV and all(d < CONV for d in recent_diffs):
            return True
    return False

def update_parameters_and_coordinates(iteration, execution_times, opt, nsteps, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, interface, charge, mult, basis_name, prev_energy, recent_diffs):
    """
    This function performs a series of optimization steps, updating the parameters and coordinates
    based on gradients. It also checks for convergence after each substep.

    Args:
        iteration (int): Current iteration number.
        execution_times (dict): Dictionary tracking execution times of various components.
        opt (Optimizer): Optimizer instance used for updating parameters.
        nsteps (int): Number of optimization steps to perform in this call.
        opt_state: Current state of the optimizer.
        cost_fn (callable): Cost function to minimize.
        params (np.ndarray): Current variational parameters.
        x (np.ndarray): Current nuclear coordinates.
        symbols (list of str): List of atomic symbols in the molecule.
        selected_excitations (list of tuples): List of selected excitations for the ansatz.
        dev (qml.Device): Quantum device to execute the circuits.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        spin_orbitals (int): Number of spin orbitals in the system.
        learning_rate_x (float): Learning rate for updating nuclear coordinates.
        interface (str): Interface for PennyLane (e.g., 'autograd').
        charge (int): Total charge of the molecule.
        mult (int): Multiplicity of the molecule.
        basis_name (str): Basis set to use for the calculation.
        prev_energy (float or None): Previous energy value.
        recent_diffs (list of float): List of recent energy differences for convergence checking.

    Returns:
        tuple:
            params (np.ndarray): Updated variational parameters.
            x (np.ndarray): Updated nuclear coordinates.
            energy_history (list of float): History of energy values.
            x_history (list of np.ndarray): History of nuclear coordinates.
            opt_state: Updated optimizer state.
            converged (bool): Whether convergence was achieved.
            prev_energy (float or None): Updated previous energy value.
            recent_diffs (list of float): Updated list of recent energy differences.
    """
    energy_history = []
    x_history = []
    converged = False
    start_substep_time = time.time()
    for substep_idx in range(nsteps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = np.real(energy)
        energy_history.append(energy)
        x_history.append(np.array(x))

        grad_x = compute_nuclear_gradients(
            params, x, symbols, selected_excitations, dev,
            hf_state, spin_orbitals, interface, charge, mult, basis_name
        )
        x = x - learning_rate_x * grad_x

        # Check for convergence
        if check_convergence(energy, prev_energy, recent_diffs):
            print(f"Convergence reached updating parameters and coordinates: Energy difference < {CONV}")
            converged = True
            prev_energy = energy
            end_substep_time = time.time()
            substep_duration = end_substep_time - start_substep_time
            execution_times[f"Iteration {iteration+1} - Substep {substep_idx+1}"] = substep_duration
            return params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs

        prev_energy = energy

        end_substep_time = time.time()
        substep_duration = end_substep_time - start_substep_time

        execution_times[f"Iteration {iteration+1} - Substep {substep_idx+1}"] = substep_duration

        start_substep_time = time.time()

    return params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs

def run_optimization_uccsd(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, operator_pool_copy, selected_excitations, opt, nsteps, opt_state, max_iterations, learning_rate_x):
    """
    Runs the optimization loop for the UCCSD ansatz to find the ground-state energy and optimized geometry. Iteratively builds the ansatz by selecting operators based on energy gradients,
    updates variational parameters and nuclear coordinates, and checks for convergence.

    Args:
        symbols (list of str): List of atomic symbols in the molecule.
        x (np.ndarray): Initial nuclear coordinates.
        params (np.ndarray): Initial variational parameters.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        spin_orbitals (int): Number of spin orbitals in the system.
        charge (int): Total charge of the molecule.
        mult (int): Multiplicity of the molecule.
        basis_name (str): Basis set to use for the calculation.
        dev (qml.Device): Quantum device to execute the circuits.
        interface (str): Interface for PennyLane (e.g., 'autograd').
        operator_pool_copy (list of tuples): Copy of the operator pool containing available excitations.
        selected_excitations (list of tuples): List of currently selected excitations.
        opt (Optimizer): Optimizer instance used for updating parameters.
        nsteps (int): Number of optimization steps per iteration.
        opt_state: Current state of the optimizer.
        max_iterations (int): Maximum number of optimization iterations.
        learning_rate_x (float): Learning rate for updating nuclear coordinates.

    Returns:
        tuple:
            params (np.ndarray): Optimized variational parameters.
            x (np.ndarray): Optimized nuclear coordinates.
            energy_history_total (list of float): History of energy values throughout optimization.
            x_history_total (list of np.ndarray): History of nuclear coordinates throughout optimization.
            params_history (list of np.ndarray): History of parameter sets throughout optimization.
            execution_times (dict): Dictionary tracking execution times of various components.
    """
    from .hamiltonian_builder import build_hamiltonian
    from .ansatz_preparer import prepare_ansatz, compute_operator_gradients, select_operator

    execution_times = {'build_hamiltonian':0.0,'compute_operator_gradients':0.0,'update_parameters_and_coordinates':0.0,'Total Time':0.0}
    energy_history_total = []
    x_history_total = []
    params_history = []
    recent_diffs = []
    prev_energy = None

    recent_diffs_top = []
    top_prev_energy = None

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
            iteration, execution_times, opt, nsteps, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals,
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

        if top_prev_energy is not None:
            diff_top = abs(current_energy - top_prev_energy)
            recent_diffs_top.append(diff_top)

            if len(recent_diffs_top) > 3:
                recent_diffs_top.pop(0)

            if len(recent_diffs_top) == 3 and all(d < CONV for d in recent_diffs_top):
                print(f"Top-level convergence reached: consecutive diffs < {CONV}")
                break
        top_prev_energy = current_energy
        
        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (uccsd): {total_time:.2f} seconds")
    return params, x, energy_history_total, x_history_total, params_history, execution_times

def run_optimization_vqe_classic(symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name, dev, interface, selected_excitations, opt, nsteps, opt_state, max_iterations, learning_rate_x, num_layers = 10):
    """
    Runs the optimization loop for the hardware-efficient VQE Classic ansatz to find the ground-state energy and optimized geometry.

    This function iteratively updates variational parameters and nuclear coordinates using a layered ansatz, performing optimization steps and checking for convergence.

    Args:
        symbols (list of str): List of atomic symbols in the molecule.
        x (np.ndarray): Initial nuclear coordinates.
        params (np.ndarray): Initial variational parameters.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        spin_orbitals (int): Number of spin orbitals in the system.
        charge (int): Total charge of the molecule.
        mult (int): Multiplicity of the molecule.
        basis_name (str): Basis set to use for the calculation.
        dev (qml.Device): Quantum device to execute the circuits.
        interface (str): Interface for PennyLane (e.g., 'autograd').
        selected_excitations (list of tuples): List of currently selected excitations.
        opt (Optimizer): Optimizer instance used for updating parameters.
        nsteps (int): Number of optimization steps per iteration.
        opt_state: Current state of the optimizer.
        max_iterations (int): Maximum number of optimization iterations.
        learning_rate_x (float): Learning rate for updating nuclear coordinates.
        num_layers (int, optional): Number of layers in the VQE Classic ansatz. Defaults to 10.

    Returns:
        tuple:
            params (np.ndarray): Optimized variational parameters.
            x (np.ndarray): Optimized nuclear coordinates.
            energy_history_total (list of float): History of energy values throughout optimization.
            x_history_total (list of np.ndarray): History of nuclear coordinates throughout optimization.
            params_history (list of np.ndarray): History of parameter sets throughout optimization.
            execution_times (dict): Dictionary tracking execution times of various components.
    """
    from .hamiltonian_builder import build_hamiltonian
    from .ansatz_preparer import prepare_ansatz

    execution_times = {
        'build_hamiltonian': 0.0,
        'compute_operator_gradients': 0.0,
        'update_parameters_and_coordinates': 0.0,
        'Total Time': 0.0
    }

    energy_history_total = []
    x_history_total = []
    params_history = []
    recent_diffs = []
    prev_energy = None

    # Top-level convergence tracking
    recent_diffs_top = []
    top_prev_energy = None

    optimizer_start_time = time.time()

    def cost_fn_factory(hamiltonian):
        @qml.qnode(dev, interface=interface)
        def cost_fn(p):
            prepare_ansatz(p,hf_state, selected_excitations, spin_orbitals, ansatz_type="vqe_classic", num_layers = num_layers)
            return qml.expval(hamiltonian)
        return cost_fn

    for iteration in range(max_iterations):
        iter_start_time = time.time()

        start_time = time.time()
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        end_time = time.time()
        execution_times['build_hamiltonian'] += (end_time - start_time)

        cost_fn = cost_fn_factory(hamiltonian)

        # Update parameters and coordinates
        params, x, energy_history, x_history, opt_state, converged, prev_energy, recent_diffs = update_parameters_and_coordinates( iteration, execution_times, opt, nsteps, opt_state, cost_fn, params,x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, interface='autograd', charge=charge, mult=mult, basis_name=basis_name, prev_energy=prev_energy,recent_diffs=recent_diffs
        )

        energy_history_total.extend(energy_history)
        x_history_total.extend(x_history)
        params_history.append(params.copy())
        current_energy = energy_history[-1]
        print(f"Iteration {iteration + 1}, Energy = {current_energy:.8f} Ha")

        if top_prev_energy is not None:
            diff_top = abs(current_energy - top_prev_energy)
            recent_diffs_top.append(diff_top)

            if len(recent_diffs_top) > 3:
                recent_diffs_top.pop(0)

            if len(recent_diffs_top) == 3 and all(d < CONV for d in recent_diffs_top):
                print(f"Top-level convergence reached: consecutive diffs < {CONV}")
                break

        top_prev_energy = current_energy

        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        execution_times[f"Iteration {iteration + 1}"] = iteration_time

    optimizer_end_time = time.time()
    total_time = optimizer_end_time - optimizer_start_time
    execution_times['Total Time'] = total_time
    print(f"Total optimization time (vqe_classic): {total_time:.2f} seconds")

    return params, x, energy_history_total, x_history_total, params_history, execution_times


def run_single_optimizer(optimizer_name, opt, ansatz_type_opt, layers, nsteps, symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name, hf_state, dev, operator_pool, exact_energy, results_dir):
    """
    Executes the optimization process for a single optimizer and saves the output to a file.

    This function sets up the optimization environment, runs the optimization loop using either
    the UCCSD or VQE Classic ansatz, and records the results and execution times.

    Args:
        optimizer_name (str): Name of the optimizer being used.
        opt (Optimizer): Optimizer instance used for updating parameters.
        ansatz_type_opt (str): Type of ansatz to use ("uccsd" or "vqe_classic").
        layers (int): Number of layers for the ansatz (applicable for layered ansatz types).
        nsteps (int): Number of optimization steps per iteration.
        symbols (list of str): List of atomic symbols in the molecule.
        x_init (np.ndarray): Initial nuclear coordinates.
        electrons (int): Number of electrons in the molecule.
        spin_orbitals (int): Number of spin orbitals in the system.
        charge (int): Total charge of the molecule.
        mult (int): Multiplicity of the molecule.
        basis_name (str): Basis set to use for the calculation.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        dev (qml.Device): Quantum device to execute the circuits.
        operator_pool (list of tuples): Pool of available excitations.
        exact_energy (float): Exact ground-state energy for reference.
        results_dir (str): Directory to save the optimizer's output files.

    Returns:
        tuple:
            optimizer_name (str): Name of the optimizer.
            result (dict): Dictionary containing optimization results.
    """
    output_file = os.path.join(results_dir, f"output_{optimizer_name}.txt")
    orig_stdout = os.sys.stdout
    with open(output_file, "w", buffering=1) as f:
        os.sys.stdout = f

        interface = 'autograd'
        learning_rate_x = opt.stepsize
        operator_pool_copy = operator_pool.copy()
        selected_excitations = []
        x = x_init.copy()

        print(f"\n--- Optimizing with {optimizer_name} ---\n")

        if ansatz_type_opt == "vqe_classic":
            num_layers = layers if layers > 0 else 1
            params_per_layer = 2 * spin_orbitals
            total_params = num_layers * params_per_layer
            params = 0.01 * np.random.randn(total_params)
            params = np.array(params, requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
                ) = run_optimization_vqe_classic(
                    symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                    dev, interface, selected_excitations, opt, nsteps, None, MAX_ITER, learning_rate_x, layers)
        else:
            params = np.array([], requires_grad=True)
            (params, x, energy_history_total, x_history_total, params_history, execution_times
                ) = run_optimization_uccsd(
                    symbols, x, params, hf_state, spin_orbitals, charge, mult, basis_name,
                    dev, interface, operator_pool_copy, selected_excitations, opt, nsteps, None, MAX_ITER, learning_rate_x
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
        "selected_excitations": selected_excitations,
        "ans_type": ansatz_type_opt,
        "layers": layers 
    }
    return optimizer_name, result
def optimize_molecule(symbols, x_init, electrons, spin_orbitals, optimizers, charge=0, mult=1, basis_name='sto-3g', ansatz_list=[("uccsd", 0, 10)], results_dir="temp_results_autograd"):
    """
    Orchestrates the optimization of a molecule's geometry and electronic structure using multiple optimizers. Computes the exact energy for reference, generates the Hartree-Fock state, and sets up the operator pool. It then runs optimizations in parallel for each optimizer, collects the results, and outputs the final energies, differences from exact energies, optimized geometries, and quantum circuits.

    Args:
        symbols (list of str): List of atomic symbols in the molecule.
        x_init (np.ndarray): Initial nuclear coordinates.
        electrons (int): Number of electrons in the molecule.
        spin_orbitals (int): Number of spin orbitals in the system.
        optimizers (dict): Dictionary mapping optimizer names to optimizer instances.
        charge (int, optional): Total charge of the molecule. Defaults to 0.
        mult (int, optional): Multiplicity of the molecule. Defaults to 1.
        basis_name (str, optional): Basis set to use for the calculation. Defaults to 'sto-3g'.
        ansatz_list (list of tuples, optional): List of tuples specifying ansatz types, number of layers, and number of steps (e.g., [("uccsd", 0, 10)]). Defaults to [("uccsd", 0, 10)].
        results_dir (str, optional): Directory to save optimization results. Defaults to "temp_results_autograd".

    Returns:
        dict: Dictionary containing results for the 'autograd' interface, including energy histories,
              coordinate histories, parameter histories, final energies, final parameters, final coordinates,
              exact energy references, execution times, selected excitations, ansatz types, and layers.
    """
    from .hamiltonian_builder import build_hamiltonian, generate_hf_state, get_operator_pool
    from .ansatz_preparer import prepare_ansatz

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    exact_energy = compute_exact_energy(symbols, x_init, charge, mult, basis_name)
    hf_state = generate_hf_state(electrons, spin_orbitals)
    dev = qml.device("default.qubit", wires=spin_orbitals)
    operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')
    interface_results = {}

    # Instead of using cont, we iterate over each optimizer
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        cont = 0
        for optimizer_name, opt in optimizers.items():
            if cont < len(ansatz_list):
                ans_type, layers, nsteps = ansatz_list[cont]
            else:
                ans_type, layers, nsteps = ("uccsd", 0, 10)
            cont += 1
            futures.append(
                executor.submit(
                    run_single_optimizer,
                    optimizer_name, opt, ans_type, layers, nsteps, symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name,
                    hf_state, dev, operator_pool, exact_energy, results_dir
                )
            )

        for future in concurrent.futures.as_completed(futures):
            optimizer_name, data = future.result()
            interface_results[optimizer_name] = data

    # Reading results
    for optimizer_name in optimizers.keys():
        output_file = os.path.join(results_dir, f"output_{optimizer_name}.txt")
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            print(content, end="")
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
        layers_used = interface_results[optimizer_name]["layers"]
        ans_type_used = interface_results[optimizer_name]["ans_type"]
        @qml.qnode(dev, interface='autograd')
        def final_cost_fn(p):
            prepare_ansatz(p, hf_state, selected_excitations, spin_orbitals, ansatz_type=ans_type_used, num_layers=layers_used)
            return qml.expval(hamiltonian)

        params = interface_results[optimizer_name]["final_params"]
        circuit_str = qml.draw(final_cost_fn)(params)
        print("Quantum Circuit:\n")
        print(circuit_str)
        print()

    print("=== Total Optimization Times ===\n")
    print("Interface: autograd")
    for optimizer_name in optimizers.keys():
        total_time = interface_results[optimizer_name]["execution_times"].get('Total Time', 0)
        print(f"Optimizer: {optimizer_name}, Time: {total_time:.2f} seconds")

    profiler_output_path = os.path.join(results_dir, "profile_output_autograd.txt")
    filtered_report_path = os.path.join(results_dir, "filtered_report_autograd.txt")
    print(f"Report completely saved on: {profiler_output_path}")
    print(f"Filtered report saved on: {filtered_report_path}")

    return {"autograd": interface_results}
