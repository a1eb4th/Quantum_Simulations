import pennylane as qml
from pennylane import numpy as np  # Use PennyLane's NumPy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabulate import tabulate
import time
import optax
import functools  # For lru_cache

# Constants for convergence and maximum number of iterations
MAX_ITER = 50   # Adjust as needed
CONV = 1e-8    # Convergence criterion
STEP_SIZE = 0.01   # Step size for the optimizers
jax.config.update("jax_enable_x64", True)

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

    return molecule, electrons, spin_orbitals

def build_hamiltonian(x, symbols, charge=0, mult=1, basis_name='sto-3g', interface='autograd'):
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

    x_np = qml.math.toarray(x)
    coordinates = x_np.reshape(-1, 3)

    # Ensure coordinates are standard NumPy arrays
    coordinates_np = np.array(coordinates, dtype=float)

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates_np, charge=charge, mult=mult, basis=basis_name
    )
    h_coeffs, h_ops = hamiltonian.terms()
    if interface == 'jax':
        h_coeffs = jnp.array(h_coeffs)
    else:
        h_coeffs = np.array(h_coeffs, requires_grad=False)
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
    return operator_pool


def compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface='autograd', charge=0, mult=1, basis_name='sto-3g'):
    """
    Calculates the energy gradients with respect to the nuclear coordinates x.
    """
    delta = 1e-3  # Step size for finite differences
    num_coords = len(x)

    if interface == 'jax':
        grad_x = jnp.zeros_like(x)
        params_jax = jnp.array(params)
        x = jnp.array(x)

        for i in range(num_coords):
            # Shift the coordinate at position i
            x_plus = x.at[i].add(delta)
            x_minus = x.at[i].add(-delta)

            # Convert coordinates to standard NumPy arrays for compatibility
            x_plus_np = np.array(x_plus)
            x_minus_np = np.array(x_minus)

            # Build Hamiltonians
            h_plus = build_hamiltonian(x_plus_np, symbols, charge, mult, basis_name, interface)
            h_minus = build_hamiltonian(x_minus_np, symbols, charge, mult, basis_name, interface)

            # Define cost functions
            @qml.qnode(dev, interface="jax")
            def cost_fn_plus(params):
                prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                return qml.expval(h_plus)

            @qml.qnode(dev, interface="jax")
            def cost_fn_minus(params):
                prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                return qml.expval(h_minus)

            # Compute energies
            energy_plus = cost_fn_plus(params_jax)
            energy_minus = cost_fn_minus(params_jax)

            # Compute gradient
            grad = (energy_plus - energy_minus) / (2 * delta)
            grad_x = grad_x.at[i].set(grad)

        return grad_x

    else:
        # For autograd
        grad_x = np.zeros_like(x)
        for i in range(num_coords):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            h_plus = build_hamiltonian(x_plus, symbols, charge, mult, basis_name, interface)
            h_minus = build_hamiltonian(x_minus, symbols, charge, mult, basis_name, interface)

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

def prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals):
    """
    Prepares the quantum ansatz using the selected excitations.
    """

    qml.BasisState(hf_state, wires=range(spin_orbitals))
    for i, exc in enumerate(selected_excitations):
        if len(exc) == 2:
            qml.SingleExcitation(params[i], wires=exc)
        elif len(exc) == 4:
            qml.DoubleExcitation(params[i], wires=exc)

def compute_operator_gradients(operator_pool, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals, interface='autograd'):
    """
    Calculates the energy gradients with respect to each operator in the pool.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_circuits/

    Args:
        operator_pool (list): List of available excitations.
        selected_excitations (list): List of selected excitations.
        params (np.ndarray): Ansatz parameters.
        hamiltonian (qml.Hamiltonian): Molecular Hamiltonian.
        hf_state (np.ndarray): Hartree-Fock state.
        dev (qml.Device): Quantum device.
        spin_orbitals (int): Number of spin orbitals.

    Returns:
        gradients (list): List of absolute gradients for each operator.
    """

    gradients = []
    for gate_wires in operator_pool:
        param_init = 0.0

        if interface == 'jax':
            params_jax = jnp.array(params)
            param_init_jax = jnp.array(param_init)

            @qml.qnode(dev, interface="jax")
            def circuit_with_gate(param):
                prepare_ansatz(params_jax, hf_state, selected_excitations, spin_orbitals)
                if len(gate_wires) == 2:
                    qml.SingleExcitation(param, wires=gate_wires)
                elif len(gate_wires) == 4:
                    qml.DoubleExcitation(param, wires=gate_wires)
                return qml.expval(hamiltonian)

            grad_fn = jax.grad(circuit_with_gate, argnums=0)
            grad = grad_fn(param_init_jax)
            grad = jnp.abs(grad)
            gradients.append(grad)

        else:  # autograd
            param_init_autograd = np.array(param_init, requires_grad=True)

            @qml.qnode(dev, interface="autograd")
            def circuit_with_gate(param):
                prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                if len(gate_wires) == 2:
                    qml.SingleExcitation(param, wires=gate_wires)
                elif len(gate_wires) == 4:
                    qml.DoubleExcitation(param, wires=gate_wires)
                return qml.expval(hamiltonian)

            grad_fn = qml.grad(circuit_with_gate, argnum=0)
            grad = grad_fn(param_init_autograd)
            grad = np.abs(grad)
            gradients.append(grad)
    return gradients

def select_operator(gradients, operator_pool, convergence):
    """
    Selects the operator with the largest gradient from the pool.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_circuits/

    Args:
        gradients (list): List of absolute gradients.
        operator_pool (list): List of available excitations.
        convergence (float): Convergence criterion.

    Returns:
        selected_gate (tuple or None): Selected excitation or None if convergence is reached.
        max_grad_value (float or None): Maximum gradient value or None.
    """

    if not gradients or all(np.isnan(gradients)):
        print("No more operators to add.")
        return None, None

    max_grad_index = np.argmax(gradients)
    max_grad_value = gradients[max_grad_index]

    if max_grad_value < convergence:
        print("Convergence achieved in operator selection.")
        return None, None

    selected_gate = operator_pool[max_grad_index]
    return selected_gate, max_grad_value

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

    prev_energy = None
    energy_history = []
    x_history = []
    consecutive_increases = 0
    max_consecutive_increases = 3

    for opt_step in range(10):
        backup_params = params.copy()
        backup_x = x.copy()
        backup_prev_energy = prev_energy

        if interface == 'jax':
            # JIT-compile the cost function
            @jax.jit
            def cost_and_grad(params):
                energy, grad_params = jax.value_and_grad(cost_fn)(params)
                return energy, grad_params

            # Compute energy and gradient
            energy, grad_params = cost_and_grad(params)
            energy = jnp.real(energy)
            # Update parameters using Optax optimizer
            updates, opt_state = opt.update(grad_params, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            # For autograd
            params, energy = opt.step_and_cost(cost_fn, params)
            energy = np.real(energy)

        energy_history.append(energy)
        x_history.append(x.copy())

        # Compute nuclear gradients and update x
        grad_x = compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, interface, charge, mult, basis_name)
        x = x - learning_rate_x * grad_x

        # Check for convergence
        '''
        if prev_energy is not None:
            energy_diff = energy - prev_energy
            
            if energy_diff > 0:
                consecutive_increases += 1
                print(f"Warning: energy increased at step {opt_step}. Increase: {energy_diff}")

                params = backup_params
                x = backup_x
                prev_energy = backup_prev_energy
                energy_history.pop()
                x_history.pop()

                if consecutive_increases >= max_consecutive_increases:
                    print("Reached an optimized point after 3 consecutive energy increases.")
                    break
            else:
                consecutive_increases = 0
            
        else:
            consecutive_increases = 0
        '''
        prev_energy = energy

    return params, x, energy_history, x_history, opt_state

def optimize_molecule(molecule, symbols, x_init, electrons, spin_orbitals, interfaces=['autograd', 'jax'], charge=0, mult=1, basis_name='sto-3g'):
    """
    Performs the optimization of the molecule using different interfaces and optimizers.
    """
    results = {}

    for interface in interfaces:
        print(f"\n===== Starting optimization with interface: {interface} =====\n")
        hf_state = generate_hf_state(electrons, spin_orbitals)

        dev = qml.device("default.qubit", wires=spin_orbitals)

        operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')

        if interface == 'jax':
            optimizers = {
                "Adam": optax.adam(learning_rate=STEP_SIZE),
                "Gradient Descent": optax.sgd(learning_rate=STEP_SIZE),
            }
        else:
            from pennylane.optimize import GradientDescentOptimizer, QNGOptimizer
            optimizers = {
                "Gradient Descent": GradientDescentOptimizer(stepsize=STEP_SIZE),
                "Quantum Natural Gradient": QNGOptimizer(stepsize=STEP_SIZE, approx="block-diag")
            }

        convergence = CONV
        max_iterations = MAX_ITER
        learning_rate_x = 0.01

        interface_results = {}

        for optimizer_name, opt in optimizers.items():
            print(f"\n--- Optimizing with {optimizer_name} ---")

            operator_pool_copy = operator_pool.copy()
            selected_excitations = []
            params = []

            energy_history_total = []
            x_history_total = []
            params_history = []

            x = x_init.copy()

            # Initialize optimizer state
            if interface == 'jax':
                params = jnp.array(params)
                x = jnp.array(x)
                opt_state = opt.init(params)
            else:
                params = np.array(params, requires_grad=True)
                opt_state = None  # Not used for autograd

            # Start timing
            start_time = time.time()

            for iteration in range(max_iterations):
                hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name, interface)

                gradients = compute_operator_gradients(operator_pool_copy, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals, interface)

                selected_gate, max_grad_value = select_operator(gradients, operator_pool_copy, convergence)
                if selected_gate is None:
                    break

                selected_excitations.append(selected_gate)

                if interface == 'jax':
                    # Add a new parameter initialized to 0.0
                    new_param = jnp.array([0.0])
                    params = jnp.concatenate([params, new_param])

                    if optimizer_name == 'Adam':
                        # Manually update the optimizer state to include the new parameter
                        # Extract the existing state
                        old_state = opt_state[0]  # ScaleByAdamState
                        empty_state = opt_state[1]  # EmptyState

                        # Append zero to 'mu' and 'nu' for the new parameter
                        new_mu = jnp.concatenate([old_state.mu, jnp.array([0.0])])
                        new_nu = jnp.concatenate([old_state.nu, jnp.array([0.0])])

                        # Increment the count
                        new_count = old_state.count + 1

                        # Create a new ScaleByAdamState with updated 'mu', 'nu', and 'count'
                        new_scale_by_adam_state = optax.ScaleByAdamState(
                            count=new_count,
                            mu=new_mu,
                            nu=new_nu
                        )
                        # Update the optimizer state
                        opt_state = (new_scale_by_adam_state, empty_state)
                        #print(f"Opt state: {opt_state}")
                    else:
                        opt_state = opt.init(params)

                else:
                    # For autograd, append a new parameter
                    params = np.append(params, 0.0)
                    params = np.array(params, requires_grad=True)

                # JIT-compile the cost function for JAX
                if interface == 'jax':
                    @jax.jit
                    @qml.qnode(dev, interface=interface)
                    def cost_fn(params):
                        prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                        return qml.expval(hamiltonian)
                else:
                    @qml.qnode(dev, interface=interface)
                    def cost_fn(params):
                        prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                        return qml.expval(hamiltonian)

                # Update parameters and coordinates
                params, x, energy_history, x_history, opt_state = update_parameters_and_coordinates(
                    opt, opt_state, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals,
                    learning_rate_x, convergence, interface, charge, mult, basis_name
                )

                energy_history_total.extend(energy_history)
                x_history_total.extend(x_history)
                params_history.append(params.copy())

                print(f"Iteration {iteration + 1}, Energy = {energy_history[-1]:.8f} Ha, Max Gradient = {max_grad_value:.5e}")
                if max_grad_value < convergence:
                    print("Convergence achieved in iteration", iteration + 1)
                    break

            # End timing
            end_time = time.time()
            total_time = end_time - start_time
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

            # Optional: Print the final quantum circuit
            print(f"Quantum Circuit with {optimizer_name} ({interface}):\n")
            print(qml.draw(cost_fn)(params))

        results[interface] = interface_results

    # Compare total optimization times
    print("\n=== Total Optimization Times ===")
    for interface, interface_results in results.items():
        print(f"\nInterface: {interface}")
        for optimizer_name, data in interface_results.items():
            total_time = data.get("total_time", 0)
            print(f"Optimizer: {optimizer_name}, Time: {total_time:.2f} seconds")

    return results


def visualize_results(results, symbols):
    """
    Generates plots to compare optimizers and visualize the evolution of coordinates.

    Visualization code to compare results across interfaces and optimizers
    Plot energy over optimization steps comparing optimizers and interfaces

    Args:
        results (dict): Dictionary with results for each optimizer.
        symbols (list): List of atomic symbols.
    """
    plt.figure(figsize=(10, 6))
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            if data["energy_history"]:  # Ensure energy_history is not empty
                label = f"{optimizer_name} ({interface})"
                plt.plot(data["energy_history"], label=label)
    plt.xlabel('Optimization Step', fontsize=14)
    plt.ylabel('Energy (Ha)', fontsize=14)
    plt.title('Energy Evolution During Optimization', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualization of nuclear coordinates in a single figure
    num_atoms = len(symbols)
    axes = ['x', 'y', 'z']
    num_axes = len(axes)

    fig, axs = plt.subplots(num_atoms, num_axes, figsize=(5*num_axes, 4*num_atoms), sharex=True)

    for atom_index in range(num_atoms):
        for axis_index, axis_name in enumerate(axes):
            ax = axs[atom_index, axis_index] if num_atoms > 1 else axs[axis_index]
            for interface, interface_results in results.items():
                for optimizer_name, data in interface_results.items():
                    x_history = np.array(data["x_history"])
                    num_iterations = len(x_history)
                    coord_history = x_history[:, 3*atom_index + axis_index]
                    label = f"{optimizer_name} ({interface})"
                    ax.plot(range(num_iterations), coord_history, label=label)
            if atom_index == num_atoms - 1:
                ax.set_xlabel('Optimization Step', fontsize=12)
            ax.set_ylabel(f'{symbols[atom_index]} - {axis_name} (Å)', fontsize=12)
            ax.grid(True)
            if atom_index == 0 and axis_index == num_axes - 1:
                ax.legend()

    plt.tight_layout()
    plt.show()

    # 3D Visualization of the final geometries
    visualize_final_geometries(results, symbols)

def visualize_final_geometries(results, symbols):
    """
    Generates a 3D plot showing the final geometries for each optimizer and interface.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors and markers for each optimizer and interface
    colors = {'Gradient Descent': 'r', 'Adam': 'g', 'Quantum Natural Gradient': 'b'}
    markers = {'autograd': 'o', 'jax': '^'}

    # Get the range of coordinates to adjust point sizes
    all_coords = []
    for interface_results in results.values():
        for data in interface_results.values():
            final_coords = data['final_x'].reshape(-1, 3)
            all_coords.append(final_coords)
    all_coords = np.concatenate(all_coords)
    max_range = np.ptp(all_coords, axis=0).max()  # Maximum range in any axis

    # Scale for point sizes
    size_scale = 200 / max_range  # Adjust 200 as needed

    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            final_coords = data["final_x"].reshape(-1, 3)
            color = colors.get(optimizer_name, 'k')
            marker = markers.get(interface, 's')
            # Calculate point sizes based on distance from the molecule center
            center = final_coords.mean(axis=0)
            distances = np.linalg.norm(final_coords - center, axis=1)
            sizes = distances * size_scale + 50  # Add a minimum size for visibility

            for i, atom in enumerate(symbols):
                label = f"{atom} - {optimizer_name} ({interface})" if i == 0 else ""
                ax.scatter(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                           color=color, marker=marker, s=sizes[i], label=label)
                ax.text(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                        f"{atom}", size=10, color=color)

    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title('Final Geometries of Optimizers and Interfaces')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Avoid duplicate labels
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

def mol_optimizer(selected_molecules, interfaces=['jax', 'autograd']):
    """
    Tests the molecular optimization for a list of selected molecules.

    Args:
        selected_molecules (list): List of selected molecule objects.
        interfaces (list, optional): List of interfaces to use ('jax', 'autograd').

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

        molecule, electrons, spin_orbitals = initialize_molecule(symbols, x_init, charge, mult, basis_name)

        results = optimize_molecule(molecule, symbols, x_init, electrons, spin_orbitals, interfaces, charge, mult, basis_name)

        # Visualize results
        visualize_results(results, symbols)