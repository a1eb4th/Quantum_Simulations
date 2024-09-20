from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.circuit.library import UCCSD
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP

# 1. Definir la molécula H₂SO₄ (ácido sulfúrico)
# Coordenadas atómicas aproximadas del ácido sulfúrico
acid_sulfuric = "S 0.000 0.000 0.000; O 0.000 1.430 0.000; O 1.240 -0.715 0.000; O -1.240 -0.715 0.000; H 0.000 2.040 0.000"

# Usamos el driver PySCF para definir la molécula
driver = PySCFDriver(atom=acid_sulfuric, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")

# 2. Resolver el problema de estructura electrónica
es_problem = ElectronicStructureProblem(driver)

# 3. Mapeo Jordan-Wigner a operadores cuánticos
mapper = JordanWignerMapper()
qubit_converter = QubitConverter(mapper)

# 4. Obtener la Hamiltoniana de la molécula
second_q_op = es_problem.second_q_ops()

# 5. Algoritmo VQE con optimizador clásico SLSQP y ansatz UCCSD
quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
uccsd_ansatz = UCCSD(qubit_converter=qubit_converter, num_particles=es_problem.num_particles, num_spin_orbitals=es_problem.num_spin_orbitals)
vqe_solver = VQE(uccsd_ansatz, optimizer=SLSQP(), quantum_instance=quantum_instance)

# 6. Ejecutar el VQE para obtener la energía mínima
result = vqe_solver.compute_minimum_eigenvalue(second_q_op)
print("Energía del estado fundamental de H₂SO₄:", result.eigenvalue.real)

# 7. Visualizar el resultado
from qiskit.visualization import plot_histogram
plot_histogram(result.optimal_circuit.decompose())
