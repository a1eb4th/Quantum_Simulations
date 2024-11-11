#!/usr/bin/env python3

import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
import sys
import shutil
from exceptions import QuantumSimulationError
from molecule_simulation import QuantumSimulation
from chemical_reaction import ChemicalReaction
from functions import from_user_input  # Importa la función específica

# Importa las funciones necesarias
from mol_optimizer import mol_optimizer

import warnings
from numpy import ComplexWarning
from pennylane import numpy as np
warnings.filterwarnings("ignore")
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()


# Configurar el directorio de resultados temporales
TEMP_RESULTS_DIR = "temp_results"

if os.path.exists(TEMP_RESULTS_DIR):
    shutil.rmtree(TEMP_RESULTS_DIR) 
os.makedirs(TEMP_RESULTS_DIR) 

terminal_output_file = open(os.path.join(TEMP_RESULTS_DIR, "output.txt"), "w", buffering=1)
sys.stdout = terminal_output_file


def main():

    molecules, args, type_sim = from_user_input()

    if type_sim == 'optimization':
        mol_optimizer(molecules)

if __name__ == "__main__":
    main()

terminal_output_file.close()
sys.stdout = sys.__stdout__
pr.disable()
ps = pstats.Stats(pr).sort_stats('cumtime')
ps.dump_stats('profiling_results.prof')
