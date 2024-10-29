#!/usr/bin/env python3

import os
os.environ["JAX_ENABLE_X64"] = "1"

from exceptions import QuantumSimulationError
from molecule_simulation import QuantumSimulation
from chemical_reaction import ChemicalReaction
from functions import from_user_input  # Importa la función específica

# Importa las funciones necesarias
from mol_optimizer import mol_optimizer

import warnings
from numpy import ComplexWarning
from pennylane import numpy as np

warnings.filterwarnings("ignore", category=ComplexWarning)

def main():

    molecules, args, type_sim = from_user_input()

    if type_sim == 'optimization':
        mol_optimizer(molecules, interfaces=['jax'])

if __name__ == "__main__":
    main()
