import pennylane as qml
from pennylane import numpy as np

def prepare_ansatz_uccsd(params, hf_state, selected_excitations, spin_orbitals):
    """
    Prepares the Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz based on selected excitations.

    This ansatz initializes the Hartree-Fock state and applies single or double excitation gates depending on the selected excitations.

    Args:
        params (array-like): Variational parameters for the excitation gates.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        selected_excitations (list of tuples): List of selected excitations, where each excitation is represented by a tuple of wire indices.
        spin_orbitals (int): Number of spin orbitals in the system.

    Returns:
        None
    """
    qml.BasisState(hf_state, wires=range(spin_orbitals))
    for i, exc in enumerate(selected_excitations):
        if len(exc) == 2:
            qml.SingleExcitation(params[i], wires=exc)
        elif len(exc) == 4:
            qml.DoubleExcitation(params[i], wires=exc)

def prepare_ansatz_vqe_classic(num_layers, params, hf_state, selected_excitations, spin_orbitals):
    """
    Prepares a hardware-efficient VQE ansatz with a specified number of layers.

    The total number of parameters per layer is 2 times the number of spin orbitals (one for RX and one for RY per qubit).
    For `num_layers` layers, the total number of parameters is `num_layers * 2 * spin_orbitals`.

    Args:
        num_layers (int): Number of ansatz layers to apply.
        params (array-like): Variational parameters for the RX and RY gates.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        selected_excitations (list): Not used in this ansatz type but included for interface consistency.
        spin_orbitals (int): Number of spin orbitals in the system.

    Returns:
        None
    """
    num_layers = num_layers
    param_idx = 0
    for _ in range(num_layers):
        # Apply RX and RY rotations on each qubit
        for q in range(spin_orbitals):
            qml.RX(params[param_idx], wires=q)
            param_idx += 1
        for q in range(spin_orbitals):
            qml.RY(params[param_idx], wires=q)
            param_idx += 1

        # Apply a linear chain of CNOT gates
        for i in range(spin_orbitals - 1):
            qml.CNOT(wires=[i, i+1])



ANSATZ_MAP = {
    "uccsd": prepare_ansatz_uccsd,
    "vqe_classic": prepare_ansatz_vqe_classic,
}



def prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals, ansatz_type="uccsd", num_layers = 10):
    """
    Prepares the quantum ansatz based on the specified ansatz type.

    Depending on the `ansatz_type`, it calls the corresponding ansatz preparation function.
    For UCCSD, it utilizes selected excitations, whereas for hardware-efficient ansatzes like VQE Classic,
    it uses a layered approach with RX, RY, and CNOT gates.

    Args:
        params (array-like): Variational parameters for the ansatz gates.
        hf_state (array-like): Hartree-Fock state as a binary string indicating occupied orbitals.
        selected_excitations (list of tuples): List of selected excitations (used only for UCCSD ansatz).
        spin_orbitals (int): Number of spin orbitals in the system.
        ansatz_type (str, optional): Type of ansatz to prepare ("uccsd", "vqe_classic", etc.). Defaults to "uccsd".
        num_layers (int, optional): Number of layers for layered ansatz types like "vqe_classic". Defaults to 10.

    Raises:
        ValueError: If the specified `ansatz_type` is not recognized.

    Returns:
        None
    """
    if ansatz_type not in ANSATZ_MAP:
        raise ValueError(f"Ansatz type '{ansatz_type}' is not recognized. Available: {list(ANSATZ_MAP.keys())}")

    ansatz_fn = ANSATZ_MAP[ansatz_type]
    
    if ansatz_type == "uccsd":
        ansatz_fn(params, hf_state, selected_excitations, spin_orbitals)
    else:
        ansatz_fn(num_layers,params, hf_state, [], spin_orbitals)

def compute_operator_gradients(operator_pool, selected_excitations, params, hamiltonian, hf_state, dev, spin_orbitals, ansatz_type="uccsd"):
    """
    Computes the energy gradients with respect to each operator in the operator pool.

    For each operator in the `operator_pool`, this function calculates the absolute value of the gradient
    of the expectation value of the Hamiltonian with respect to the operator's parameter.

    Args:
        operator_pool (list of tuples): List of available excitations, where each excitation is represented by a tuple of wire indices.
        selected_excitations (list of tuples): List of currently selected excitations (used for ansatz preparation).
        params (np.ndarray): Variational parameters for the ansatz.
        hamiltonian (qml.Hamiltonian): Molecular Hamiltonian.
        hf_state (np.ndarray): Hartree-Fock state as a binary string indicating occupied orbitals.
        dev (qml.Device): Quantum device to execute the circuits.
        spin_orbitals (int): Number of spin orbitals in the system.
        ansatz_type (str, optional): Type of ansatz to use ("uccsd", "vqe_classic", etc.). Defaults to "uccsd".

    Returns:
        gradients (list of float): List of absolute gradient values for each operator in the pool.
    """
    gradients = []
    for gate_wires in operator_pool:
        param_init_autograd = np.array(0.0, requires_grad=True)
    
        @qml.qnode(dev, interface="autograd")
        def circuit_with_gate(param):
            prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals, ansatz_type=ansatz_type)
            if len(gate_wires) == 2:
                qml.SingleExcitation(param, wires=gate_wires)
            elif len(gate_wires) == 4:
                qml.DoubleExcitation(param, wires=gate_wires)
            return qml.expval(hamiltonian)
    
        grad_fn_autograd = qml.grad(circuit_with_gate, argnum=0)
        grad = grad_fn_autograd(param_init_autograd)
        gradients.append(np.abs(grad))
    
    return gradients

def select_operator(gradients, operator_pool, convergence):
    """
    Selects the operator with the largest gradient from the operator pool.

    This function identifies the operator corresponding to the highest absolute gradient.
    If the maximum gradient is below the specified convergence threshold, it indicates that convergence has been achieved.

    Args:
        gradients (list of float): List of absolute gradient values for each operator.
        operator_pool (list of tuples): List of available excitations, where each excitation is represented by a tuple of wire indices.
        convergence (float): Convergence criterion threshold. If the maximum gradient is below this value, selection stops.

    Returns:
        tuple:
            selected_gate (tuple or None): The selected excitation with the highest gradient. Returns `None` if convergence is achieved or no operators are left.
            max_grad_value (float or None): The value of the maximum gradient. Returns `None` if convergence is achieved or no operators are left.
    """
    if len(gradients) == 0 or np.all(np.isnan(gradients)):
        print("No more operators to add.")
        return None, None

    max_grad_index = np.argmax(gradients)
    max_grad_value = gradients[max_grad_index]

    if max_grad_value < convergence:
        print("Convergence achieved in operator selection.")
        return None, None

    selected_gate = operator_pool[max_grad_index]
    return selected_gate, max_grad_value