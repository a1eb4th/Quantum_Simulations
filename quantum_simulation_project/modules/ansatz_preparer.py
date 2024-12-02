import pennylane as qml
from pennylane import numpy as np

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
    # Implementación existente para autograd
    gradients = []
    for gate_wires in operator_pool:
        param_init_autograd = np.array(0.0, requires_grad=True)
    
        @qml.qnode(dev, interface="autograd")
        def circuit_with_gate(param):
            prepare_ansatz(params, hf_state, selected_excitations, spin_orbitals)
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