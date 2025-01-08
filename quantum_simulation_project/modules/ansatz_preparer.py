import pennylane as qml
from pennylane import numpy as np

def prepare_ansatz_uccsd(params, hf_state, selected_excitations, spin_orbitals):
    """
    Prepara el ansatz tipo UCCSD (basado en excitaciones simples y dobles).
    Este es el ansatz original que tenías.
    """
    qml.BasisState(hf_state, wires=range(spin_orbitals))
    for i, exc in enumerate(selected_excitations):
        if len(exc) == 2:
            qml.SingleExcitation(params[i], wires=exc)
        elif len(exc) == 4:
            qml.DoubleExcitation(params[i], wires=exc)

def prepare_ansatz_vqe_classic(num_layers, params, hf_state, selected_excitations, spin_orbitals):
    """
    Ansatzt tipo hardware-efficient con 10 capas.
    Cada capa:
    - Aplica Rx y Ry en cada qubit.
    - Aplica una cadena de CNOT lineal entre los qubits.

    Número de parámetros:
    Por capa:
      - Para cada qubit: 2 parámetros (uno para RX y otro para RY)
    Total por capa: 2 * spin_orbitals
    Total para 10 capas: 10 * 2 * spin_orbitals

    Se asume que 'params' es un array 1D con dimensión >= 20 * spin_orbitals.
    """
    num_layers = num_layers
    param_idx = 0
    for _ in range(num_layers):
        # RX y RY en cada qubit
        for q in range(spin_orbitals):
            qml.RX(params[param_idx], wires=q)
            param_idx += 1
        for q in range(spin_orbitals):
            qml.RY(params[param_idx], wires=q)
            param_idx += 1

        # CNOT en cadena
        for i in range(spin_orbitals - 1):
            qml.CNOT(wires=[i, i+1])



ANSATZ_MAP = {
    "uccsd": prepare_ansatz_uccsd,
    "vqe_classic": prepare_ansatz_vqe_classic,
}



def prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals, ansatz_type="uccsd", num_layers = 10):
    """
    Prepares the quantum ansatz using the selected ansatz type.
    
    Args:
        params (array): Variational parameters.
        hf_state (array): Hartree-Fock state.
        selected_excitations (list): List of selected excitations (only used for UCC-type ansatz).
        spin_orbitals (int): Number of spin orbitals.
        ansatz_type (str): Type of ansatz to use ("uccsd", "vqe_classic", etc.)
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
    Calcula los gradientes de energía con respecto a cada operador en el pool.
    
    Args:
        operator_pool (list): Lista de excitaciones disponibles.
        selected_excitations (list): Lista de excitaciones seleccionadas.
        params (np.ndarray o jnp.ndarray): Parámetros del ansatz.
        hamiltonian (qml.Hamiltonian): Hamiltoniano molecular.
        hf_state (np.ndarray): Estado de Hartree-Fock.
        dev (qml.Device): Dispositivo cuántico.
        spin_orbitals (int): Número de orbitales de spin.
        interface (str, optional): 'jax' o 'autograd' para diferenciación.
    
    Returns:
        gradients (list o jnp.ndarray): Lista de gradientes absolutos para cada operador.
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
    Selects the operator with the largest gradient from the pool.

    Args:
        gradients (list or jnp.ndarray): List of absolute gradients.
        operator_pool (list): List of available excitations.
        convergence (float): Convergence criterion.

    Returns:
        selected_gate (tuple or None): Selected excitation or None if convergence is reached.
        max_grad_value (float or None): Maximum gradient value or None.
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