#!/usr/bin/env python3

import os
os.environ["JAX_ENABLE_X64"] = "1"
# Configurar variables de entorno antes de importar JAX
os.environ['OMP_NUM_THREADS'] = '128'  # Ajustar al número de hilos que deseas usar
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
import sys
import shutil
from exceptions import QuantumSimulationError
from molecule_simulation import QuantumSimulation
from chemical_reaction import ChemicalReaction
from functions import from_user_input

from mol_optimizer import mol_optimizer

import warnings
from numpy import ComplexWarning
from pennylane import numpy as np
warnings.filterwarnings("ignore")


TEMP_RESULTS_DIR = "temp_results_jax"

if os.path.exists(TEMP_RESULTS_DIR):
    shutil.rmtree(TEMP_RESULTS_DIR) 
os.makedirs(TEMP_RESULTS_DIR) 



def main():

    molecules, args, type_sim = from_user_input()

    if type_sim == 'optimization':
        mol_optimizer(molecules)

if __name__ == "__main__":
    main()