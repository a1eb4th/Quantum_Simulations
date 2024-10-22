# coding=utf-8
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane.optimize import GradientDescentOptimizer, QNGOptimizer
from tabulate import tabulate

# Constants for convergence and maximum number of iterations
MAX_ITER = 5   # Increased the maximum number of iterations
CONV = 1e-8    # Adjusted the convergence criterion
STEP_SIZE = 0.01   # Step size for the optimizers

def get_predefined_molecules():
    """
    Defines a list of small predefined molecules with initial coordinates away from equilibrium.

    Source of inspiration:
    https://pennylane.ai/qml/demos/qchem_hydrogen_molecule.html

    Returns:
        predefined_molecules (dict): Dictionary with molecule names and their parameters.
    """
    predefined_molecules = {
        "H2": {
            "symbols": ["H", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,
                1.5, 0.0, 0.0  # Initial distance increased to 1.5 Å
            ],
            "charge": 0,
            "mult": 1,
            "basis_name": 'sto-3g'
        },
        "H2+": {
            "symbols": ["H", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,
                2.0, 0.0, 0.0  # Initial distance increased to 2.0 Å
            ],
            "charge": 1,
            "mult": 2,  # Spin multiplicity corrected to 2 for open-shell
            "basis_name": 'sto-3g'
        },
        "HeH+": {
            "symbols": ["He", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,
                1.6, 0.4, -0.1  # Initial distance increased to 1.6 Å
            ],
            "charge": 1,
            "mult": 1,
            "basis_name": 'sto-3g'
        },
        "H2O": {
            "symbols": ["O", "H", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,        # Oxygen at the origin
                0.958, 0.0, 0.0,      # First hydrogen at 0.958 Å (approximate O-H bond length)
                -0.239, 0.927, 0.0    # Second hydrogen adjusted for an approximate 104.5° angle
            ],
            "charge": 0,
            "mult": 1,  # Singlet state (ground state)
            "basis_name": 'sto-3g'
        },
        "LiH": {
            "symbols": ["Li", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,
                2.0, 0.0, 0.0  # Initial distance increased to 2.0 Å
            ],
            "charge": 0,
            "mult": 1,
            "basis_name": 'sto-3g'
        },
        "BeH2": {
            "symbols": ["Be", "H", "H"],
            "coordinates": [
                0.0, 0.0, 0.0,
                1.6, 0.0, 0.0,
                -1.6, 0.0, 0.0  # Initial distance increased to 1.6 Å
            ],
            "charge": 0,
            "mult": 1,
            "basis_name": 'sto-3g'
        },
        "H3+": {
            "symbols": ["H", "H", "H"],
            "coordinates": [
                0.028, 0.054, 0.0,         # First hydrogen atom at the origin
                0.986, 1.610, 0.0,         # Second atom at 1.5 Å on the X-axis
                1.855, 0.002, 0.0          # Third atom displaced in Y to form an isosceles triangle
            ],
            "charge": 1,
            "mult": 1,  # Spin multiplicity corrected to 1 for a singlet state
            "basis_name": 'sto-3g'  # Changed to a more extensive basis
        }
    }
    return predefined_molecules

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
    x_np = np.array(qml.math.toarray(x), requires_grad=False)
    coordinates = x_np.reshape(-1, 3)

    # Calculate mol.spin from multiplicity: mol.spin = mult - 1
    spin = mult - 1
    mol = qml.qchem.Molecule(symbols, coordinates, charge=charge, mult=mult, basis_name=basis_name)

    # Set mol.spin directly
    mol.spin = spin

    # Build the Hamiltonian using the 'pyscf' method for better compatibility
    if mult == 2:
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(mol, method='openfermion')
    else:    
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(mol)

    h_coeffs, h_ops = hamiltonian.terms()
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
    https://pennylane.ai/qml/demos/tutorial_adaptive_vqe.html

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

def compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, charge=0, mult=1, basis_name='sto-3g'):
    """
    Calculates the energy gradients with respect to the nuclear coordinates x.
    """
    grad_x = np.zeros_like(x)
    delta = 1e-3  # Step size for finite differences
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        h_plus = build_hamiltonian(x_plus, symbols, charge, mult, basis_name)
        h_minus = build_hamiltonian(x_minus, symbols, charge, mult, basis_name)

        # Define QNodes with the shifted Hamiltonians
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
        
        # Ensure energies are real
        energy_plus = qml.math.real(energy_plus)
        energy_minus = qml.math.real(energy_minus)

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
    # Get the dense matrix of the Hamiltonian
    H = hamiltonian.matrix()
    
    # Diagonalize the matrix to obtain eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    
    # The exact energy is the minimum eigenvalue
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

def compute_operator_gradients(operator_pool, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals):
    """
    Calculates the energy gradients with respect to each operator in the pool.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_vqe.html

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
        param_init = np.array(0.0, requires_grad=True)

        # Create a function to calculate energy with the current operator
        @qml.qnode(dev, interface="autograd")
        def circuit_with_gate(param):
            prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
            # Add the current gate with the given parameter
            if len(gate_wires) == 2:
                qml.SingleExcitation(param, wires=gate_wires)
            elif len(gate_wires) == 4:
                qml.DoubleExcitation(param, wires=gate_wires)
            return qml.expval(hamiltonian)

        # Calculate the gradient at param=0.0
        grad_fn = qml.grad(circuit_with_gate, argnum=0)
        grad = grad_fn(param_init)
        
        # Ensure the gradient is real
        grad = qml.math.real(grad)
        
        gradients.append(np.abs(grad))
    return gradients

def select_operator(gradients, operator_pool, convergence):
    """
    Selects the operator with the largest gradient from the pool.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_vqe.html

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

def update_parameters_and_coordinates(opt, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, learning_rate_x, convergence, charge=0, mult=1, basis_name='sto-3g'):
    """
    Updates the circuit parameters and nuclear coordinates.
    
    Source:
    https://pennylane.ai/qml/demos/tutorial_vqe_qng.html
    
    Args:
        opt (qml.optimize.Optimizer): Quantum optimizer.
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
        charge (int, optional): Charge of the molecule.
        mult (int, optional): Spin multiplicity.
        basis_name (str, optional): Name of the atomic basis.
    
    Returns:
        params (np.ndarray): Updated parameters.
        x (np.ndarray): Updated nuclear coordinates.
        energy_history (list): Energy history during optimization.
        x_history (list): Coordinate history during optimization.
    """
    prev_energy = None
    energy_history = []
    x_history = []
    consecutive_increases = 0  # Counter for consecutive energy increases
    max_consecutive_increases = 3  # Maximum allowed consecutive energy increases

    for opt_step in range(10):  # Number of optimization steps per iteration
        # Save current parameters and coordinates for possible reversion
        backup_params = params.copy()
        backup_x = x.copy()
        backup_prev_energy = prev_energy

        # Update the circuit parameters
        params, energy = opt.step_and_cost(cost_fn, params)

        # Store energy and coordinates for plotting
        energy_history.append(energy)
        x_history.append(x.copy())

        # Calculate the gradient with respect to x
        hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)
        grad_x = compute_nuclear_gradients(params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, charge, mult, basis_name)
        

        # Update x using gradient descent
        x = x - learning_rate_x * grad_x

        if prev_energy is not None:
            energy_diff = energy - prev_energy  # Energy difference (can be positive or negative)

            if abs(energy_diff) < convergence:
                print(f"Convergence achieved at step {opt_step}. Energy difference: {energy_diff}")
                break

            if energy_diff > 0:
                consecutive_increases += 1
                print(f"Warning: energy increased at step {opt_step}. Increase: {energy_diff}")
                
                # Revert to previous parameters and coordinates
                params = backup_params
                x = backup_x
                prev_energy = backup_prev_energy
                energy_history.pop()  # Remove the last recorded energy
                x_history.pop()       # Remove the last recorded coordinate

                if consecutive_increases >= max_consecutive_increases:
                    print("Reached an optimized point after 3 consecutive energy increases.")
                    break
            else:
                consecutive_increases = 0  # Reset counter if energy does not increase

        else:
            consecutive_increases = 0  # Initialize counter on the first step

        prev_energy = energy

    return params, x, energy_history, x_history

def optimize_molecule(molecule, symbols, x_init, electrons, spin_orbitals, charge=0, mult=1, basis_name='sto-3g'):
    """
    Performs the optimization of the molecule using different optimizers.

    Source:
    https://pennylane.ai/qml/demos/tutorial_adaptive_vqe.html
    https://pennylane.ai/qml/demos/tutorial_vqe_qng.html

    Args:
        molecule (qml.qchem.Molecule): PennyLane Molecule object.
        symbols (list): List of atomic symbols.
        x_init (np.ndarray): Initial coordinates.
        electrons (int): Total number of electrons.
        spin_orbitals (int): Number of spin orbitals.
        charge (int, optional): Charge of the molecule.
        mult (int, optional): Spin multiplicity.
        basis_name (str, optional): Name of the atomic basis.

    Returns:
        results (dict): Dictionary with results for each optimizer.
    """
    # Generate the Hartree-Fock state
    hf_state = generate_hf_state(electrons, spin_orbitals)

    # Quantum device
    dev = qml.device("default.qubit", wires=spin_orbitals)

    # Get the operator pool (single and double excitations)
    operator_pool = get_operator_pool(electrons, spin_orbitals, excitation_level='both')

    # Define the optimizers to compare
    optimizers = {
        "Gradient Descent": GradientDescentOptimizer(stepsize=STEP_SIZE),
        "Quantum Natural Gradient": QNGOptimizer(stepsize=STEP_SIZE, approx="block-diag")
    }

    # Structures to store results
    results = {}

    # Convergence criterion and other parameters
    convergence = CONV
    max_iterations = MAX_ITER
    learning_rate_x = 0.01  # Learning rate for nuclear coordinates

    # Adaptive optimization loop for each optimizer
    for optimizer_name, opt in optimizers.items():
        print(f"\n--- Optimizing with {optimizer_name} ---")
        
        # Reset the operator pool and parameters
        operator_pool_copy = operator_pool.copy()
        selected_excitations = []
        params = np.array([], requires_grad=True)

        # Energy and coordinate history for plotting
        energy_history_total = []
        x_history_total = []
        params_history = []

        # Copy of the initial coordinates for each optimizer
        x = x_init.copy()

        for iteration in range(max_iterations):
            # Build the Hamiltonian with current coordinates
            hamiltonian = build_hamiltonian(x, symbols, charge, mult, basis_name)

            # Calculate gradients with respect to operators in the pool
            gradients = compute_operator_gradients(operator_pool_copy, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals)

            # Select the operator with the largest gradient
            selected_gate, max_grad_value = select_operator(gradients, operator_pool_copy, convergence)
            if selected_gate is None:
                break

            # Add the selected operator and update parameters
            selected_excitations.append(selected_gate)
            params = np.append(params, 0.0)
            params.requires_grad = True

            # Define the cost function
            @qml.qnode(dev, interface="autograd")
            def cost_fn(params):
                prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
                return qml.expval(hamiltonian)

            # Update parameters and nuclear coordinates
            params, x, energy_history, x_history = update_parameters_and_coordinates(
                opt, cost_fn, params, x, symbols, selected_excitations, dev, hf_state, spin_orbitals, 
                learning_rate_x, convergence, charge, mult, basis_name
            )

            # Store the total history
            energy_history_total.extend(energy_history)
            x_history_total.extend(x_history)
            params_history.append(params.copy())

            print(f"Iteration {iteration + 1}, Energy = {energy_history[-1]:.8f} Ha, Maximum Gradient = {max_grad_value:.5e}")
            if max_grad_value < convergence:
                print("Convergence achieved in iteration", iteration + 1)
                break

        # Store the results for this optimizer
        final_energy = energy_history_total[-1] if energy_history_total else None
        results[optimizer_name] = {
            "energy_history": energy_history_total,
            "x_history": x_history_total,
            "params_history": params_history,
            "final_energy": final_energy,
            "final_params": params,
            "final_x": x
        }

        if final_energy is not None:
            print(f"\nFinal energy value with {optimizer_name} = {final_energy:.8f} Ha")
        else:
            print(f"\nNo final energy value obtained with {optimizer_name}")

        # Display the final geometry of the molecule for this optimizer
        final_x = x
        print(f"\nFinal geometry of the molecule with {optimizer_name}:")
        # Display coordinates in a clear table
        atom_coords = []
        for i, atom in enumerate(symbols):
            atom_coords.append([atom, final_x[3 * i], final_x[3 * i + 1], final_x[3 * i + 2]])
        print(tabulate(atom_coords, headers=["Symbol", "x (Å)", "y (Å)", "z (Å)"], floatfmt=".6f"))

        print(f"Quantum Circuit with {optimizer_name}:\n")
        print(qml.draw(cost_fn)(params))


    return results, cost_fn

def visualize_results(results, symbols):
    """
    Generates plots to compare the optimizers and visualize the evolution of coordinates.

    Source:
    https://matplotlib.org/stable/gallery/index.html

    Args:
        results (dict): Dictionary with results for each optimizer.
        symbols (list): List of atomic symbols.
    """
    # Plot energy over optimization steps comparing optimizers
    plt.figure(figsize=(8, 6))
    for optimizer_name, data in results.items():
        if data["energy_history"]:  # Ensure energy_history is not empty
            plt.plot(data["energy_history"], label=optimizer_name)
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
            for optimizer_name, data in results.items():
                x_history = np.array(data["x_history"])
                num_iterations = len(x_history)
                coord_history = x_history[:, 3*atom_index + axis_index]
                ax.plot(range(num_iterations), coord_history, label=optimizer_name)
            if atom_index == num_atoms - 1:
                ax.set_xlabel('Optimization Step', fontsize=12)
            ax.set_ylabel(f'{symbols[atom_index]} - {axis_name} (Å)', fontsize=12)
            ax.grid(True)
            if atom_index == 0 and axis_index == num_axes - 1:
                ax.legend()

    plt.tight_layout()
    plt.show()


    # 3D Visualization of the final geometries of both optimizers in the same plot
    visualize_final_geometries(results, symbols)

def visualize_final_geometries(results, symbols):
    """
    Generates a 3D plot showing the final geometries of both optimizers together.
    The size of the points is proportional to a determined scale.

    Source:
    https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html

    Args:
        results (dict): Dictionary with results for each optimizer.
        symbols (list): List of atomic symbols.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors and markers for each optimizer
    colors = {'Gradient Descent': 'r', 'Quantum Natural Gradient': 'b'}
    markers = {'Gradient Descent': 'o', 'Quantum Natural Gradient': '^'}

    # Get the range of coordinates to adjust point sizes
    all_coords = []
    for data in results.values():
        final_coords = data['final_x'].reshape(-1, 3)
        all_coords.append(final_coords)
    all_coords = np.concatenate(all_coords)
    max_range = np.ptp(all_coords, axis=0).max()  # Maximum range in any axis

    # Scale for point sizes
    size_scale = 200 / max_range  # Adjust 200 as needed

    for optimizer_name, data in results.items():
        final_coords = data["final_x"].reshape(-1, 3)
        color = colors.get(optimizer_name, 'k')
        marker = markers.get(optimizer_name, 'o')
        # Calculate point sizes based on distance from the molecule center
        center = final_coords.mean(axis=0)
        distances = np.linalg.norm(final_coords - center, axis=1)
        sizes = distances * size_scale + 50  # Add a minimum size for visibility

        for i, atom in enumerate(symbols):
            label = f"{atom} - {optimizer_name}" if i == 0 else ""
            ax.scatter(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                       color=color, marker=marker, s=sizes[i], label=label)
            ax.text(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                    f"{atom}", size=10, color=color)

    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title('Final Geometries of Both Optimizers')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Avoid duplicate labels
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

def compute_energy_landscape(cost_fn, param_ranges, num_points=50):
    """
    Computes the energy landscape on a grid of parameters.

    Args:
        cost_fn (callable): Cost function that depends on the parameters.
        param_ranges (list of tuples): Parameter ranges [(min0, max0), (min1, max1)].
        num_points (int): Number of points along each axis.

    Returns:
        theta0 (np.ndarray): Values of the first parameter.
        theta1 (np.ndarray): Values of the second parameter.
        energy_landscape (np.ndarray): Energy at each grid point.
    """
    theta0 = np.linspace(param_ranges[0][0], param_ranges[0][1], num_points)
    theta1 = np.linspace(param_ranges[1][0], param_ranges[1][1], num_points)
    energy_landscape = np.zeros((num_points, num_points))

    for i, t0 in enumerate(theta0):
        for j, t1 in enumerate(theta1):
            params = np.array([t0, t1])
            energy = cost_fn(params)
            energy_landscape[j, i] = energy  # Note: j, i for correct matrix orientation

    return theta0, theta1, energy_landscape

def visualize_energy_landscape(results, cost_fn):
    """
    Generates a contour plot of the energy landscape and overlays the optimization paths.

    Args:
        results (dict): Optimizer results.
        cost_fn (callable): Cost function.
    """
    # Define parameter ranges (adjust according to your values)
    param_ranges = [(0, 2*np.pi), (0, 2*np.pi)]

    # Compute the energy landscape
    theta0, theta1, energy_landscape = compute_energy_landscape(cost_fn, param_ranges)

    # Create the contour plot
    plt.figure(figsize=(8, 6))
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap("coolwarm")
    contour = plt.contourf(theta0, theta1, energy_landscape, levels=50, cmap=cmap)
    plt.colorbar(contour)
    plt.xlabel(r'$\theta_0$', fontsize=14)
    plt.ylabel(r'$\theta_1$', fontsize=14)
    plt.title('Energy Landscape and Optimization Paths', fontsize=16)

    # Overlay the optimization paths
    for optimizer_name, data in results.items():
        params_history = np.array(data["params_history"])
        if params_history.shape[1] < 2:
            continue  # Cannot plot if there are fewer than 2 parameters

        plt.plot(params_history[:, 0], params_history[:, 1], 'o-', label=optimizer_name)

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the molecular simulation.

    This procedure combines various elements and is customized, although it follows the typical
    structure of a Python script.
    """
    print("=== Molecular Optimization Simulation with PennyLane ===\n")
    
    # Get predefined molecules
    predefined_molecules = get_predefined_molecules()
    
    # Present options to the user
    print("Select a molecule to simulate:")
    for idx, molecule_name in enumerate(predefined_molecules.keys(), start=1):
        print(f"{idx}. {molecule_name}")
    print(f"{len(predefined_molecules) + 1}. Exit")
    
    # Prompt the user to choose a molecule
    while True:
        try:
            choice = int(input(f"\nEnter the number of the molecule to simulate (1-{len(predefined_molecules) + 1}): "))
            if 1 <= choice <= len(predefined_molecules):
                selected_molecule_name = list(predefined_molecules.keys())[choice - 1]
                selected_molecule = predefined_molecules[selected_molecule_name]
                print(f"\nYou have selected: {selected_molecule_name}\n")
                break
            elif choice == len(predefined_molecules) + 1:
                print("Exiting the program.")
                return
            else:
                print(f"Please choose a number between 1 and {len(predefined_molecules) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    # Extract parameters of the selected molecule
    symbols = selected_molecule["symbols"]
    coordinates = selected_molecule["coordinates"]
    charge = selected_molecule.get("charge", 0)
    mult = selected_molecule.get("mult", 1)
    basis_name = selected_molecule.get("basis_name", 'sto-3g')
    
    # Create the initial coordinates array with gradient enabled
    x_init = np.array(coordinates, requires_grad=True)
    exact_energy = compute_exact_energy(build_hamiltonian(x_init, symbols, charge, mult, basis_name))
    print(f"Exact Energy Value (FCI): {exact_energy:.8f} Ha")
    
    # Initialize the molecule
    molecule, electrons, spin_orbitals = initialize_molecule(symbols, x_init, charge, mult, basis_name)
    
    # Optimization
    results, cost_fn_for_landscape = optimize_molecule(molecule, symbols, x_init, electrons, spin_orbitals, charge, mult, basis_name)
    
    # Visualization of results
    visualize_results(results, symbols)
    # visualize_energy_landscape(results, cost_fn_for_landscape)

if __name__ == "__main__":
    main()
