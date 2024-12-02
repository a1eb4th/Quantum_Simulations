from pennylane import numpy as np
from .hamiltonian_builder import build_hamiltonian, generate_hf_state, get_operator_pool
from .ansatz_preparer import prepare_ansatz, compute_operator_gradients, select_operator
import pennylane as qml
import time
from tabulate import tabulate

# Constants for convergence and maximum number of iterations
MAX_ITER = 5   # Adjust as needed
CONV = 1e-8    # Convergence criterion
STEP_SIZE = 0.01   # Step size for the optimizers
def compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Calculates the energy gradients with respect to the nuclear coordinates x.
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

        @qml.qnode(dev, interface="autograd")
        def cost_fn_plus(params):
            prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
            return qml.expval(h_plus)

        @qml.qnode(dev, interface="autograd")
        def cost_fn_minus(params):
            prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
            return qml.expval(h_minus)

        energy_plus = cost_fn_plus(params)
        energy_minus = cost_fn_minus(params)
        grad_x[i] = (energy_plus - energy_minus) / (2 * delta)

    return grad_x

def optimize_molecule(symbols, x_init, electrons, spin_orbitals, charge=0, mult=1, basis_name='sto-3g'):
    """
    Performs the optimization of the molecule using different interfaces and optimizers.
    """
    results = {}
    interface = 'autograd'

    print(f"\n===== Starting optimization with interface: {interface} =====\n")
    hf_state = generate_hf_state(electrons, spin_orbitals)

    dev = qml.device("default.qubit", wires=spin_orbitals)


    operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')

    from pennylane.optimize import GradientDescentOptimizer, QNGOptimizer
    optimizers = {
        "Gradient Descent": GradientDescentOptimizer(stepsize=STEP_SIZE),
        #"Quantum Natural Gradient": QNGOptimizer(stepsize=STEP_SIZE, approx="block-diag")
    }

    convergence = CONV
    max_iterations = MAX_ITER
    learning_rate_x = 0.01

    interface_results = {}

    for optimizer_name, opt in optimizers.items():
        print(f"\n--- Optimizing with {optimizer_name} ---")

        execution_times = {
            'build_hamiltonian': 0.0,
            'compute_operator_gradients': 0.0,
            'update_parameters_and_coordinates': 0.0,
            'Total Time': 0.0
        }

        optimizer_start_time = time.time()
        operator_pool_copy = operator_pool.copy()
        selected_excitations = []
        params = []

        energy_history_total = []
        x_history_total = []
        params_history = []

        x = x_init.copy()

        params = np.array(params, requires_grad=True)
        opt_state = None
        start_time = time.time()

        for iteration in range(max_iterations):
            iter_start_time = time.time()

            start_time = time.time()
            hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
            end_time = time.time()
            execution_times['build_hamiltonian'] += end_time - start_time

            start_time = time.time()
            gradients = compute_operator_gradients(operator_pool_copy, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals)
            end_time = time.time()
            execution_times['compute_operator_gradients'] += end_time - start_time

            selected_gate, max_grad_value = select_operator(gradients, operator_pool_copy, convergence)
            if selected_gate is None:
                break

            selected_excitations.append(selected_gate)
            params = np.append(params, 0.0)
            params = np.array(params, requires_grad=True)
            @qml.qnode(dev, interface=interface)
            def cost_fn(params):
                prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                return qml.expval(hamiltonian)

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

            print(f"Iteration {iteration + 1}, Energy = {energy_history[-1]:.8f} Ha, Max Gradient = {max_grad_value:.5e}")
            if max_grad_value < convergence:
                print("Convergence achieved in iteration", iteration + 1)
                break
            iter_end_time = time.time()
            iteration_time = iter_end_time - iter_start_time
            execution_times[f"Iteration {iteration + 1}"] = iteration_time

        # End timing
        optimizer_end_time = time.time()
        total_time = optimizer_end_time - optimizer_start_time
        execution_times['Total Time'] = total_time
        print(f"Total optimization time with {optimizer_name} ({interface}): {total_time:.2f} seconds")

        final_energy = energy_history_total[-1] if energy_history_total else None
        interface_results[optimizer_name] = {
            "energy_history": energy_history_total,
            "x_history": x_history_total,
            "params_history": params_history,
            "final_energy": final_energy,
            "final_params": params,
            "final_x": x,
            "interface": interface,
            "total_time": total_time
            }

            # Display final results
        if final_energy is not None:
            print(f"\nFinal energy with {optimizer_name} ({interface}) = {final_energy:.8f} Ha")
        else:
            print(f"\nNo final energy obtained with {optimizer_name} ({interface})")

        final_x = x
        print(f"\nFinal geometry with {optimizer_name} ({interface}):")
        atom_coords = []
        final_x_np = np.array(final_x)
        for i, atom in enumerate(symbols):
            atom_coords.append([atom, final_x_np[3 * i], final_x_np[3 * i + 1], final_x_np[3 * i + 2]])
        print(tabulate(atom_coords, headers=["Symbol", "x (Å)", "y (Å)", "z (Å)"], floatfmt=".6f"))

        print(f"Quantum Circuit with {optimizer_name} ({interface}):\n")
        print(qml.draw(cost_fn)(params))

        write_simulation_times(symbols, interface, optimizer_name, execution_times)

    results[interface] = interface_results

    # Compare total optimization times
    print("\n=== Total Optimization Times ===")
    for interface, interface_results in results.items():
        print(f"\nInterface: {interface}")
        for optimizer_name, data in interface_results.items():
            total_time = data.get("total_time", 0)
            print(f"Optimizer: {optimizer_name}, Time: {total_time:.2f} seconds")

    return results
def update_parameters_and_coordinates(opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, convergence, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Updates the circuit parameters and nuclear coordinates.

    Source:
    https://pennylane.ai/qml/demos/tutorial_vqe_qng.html

    Args:
        opt (qml.optimize.Optimizer): Quantum optimizer.
        opt_state: Current state of the optimizer (used for Optax).
        cost_fn (callable): Cost function to optimize.
        params (np.ndarray): Current parameters.
        x (np.ndarray): Current coordinates.
        symbols (list): List of atomic symbols.
        selected_excitations (list): List of selected excitations.
        dev (qml.Device): Quantum device.
        hf_state (np.ndarray): Hartree-Fock state.
        spin_orbitals (int): Number of spin orbitals.
        learning_rate_x (float): Learning rate for nuclear coordinates.
        convergence (float): Convergence criterion.
        interface (str, optional): 'jax' or 'autograd' for differentiation.
        charge (int, optional): Charge of the molecule.
        mult (int, optional): Spin multiplicity.
        basis_name (str, optional): Name of the atomic basis.

    Returns:
        params (np.ndarray): Updated parameters.
        x (np.ndarray): Updated nuclear coordinates.
        energy_history (list): Energy history during optimization.
        x_history (list): Coordinate history during optimization.
        opt_state: Updated optimizer state.
    """
    energy_history = []
    x_history = []

    for opt_step in range(10):
        
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = np.real(energy)

        energy_history.append(energy)
        x_history.append(np.array(x))


        energy_history.append(energy)
        x_history.append(x.copy())

        grad_x = compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface, charge, mult, basis_name)
        x = x - learning_rate_x * grad_x


    return params, x, energy_history, x_history, opt_state