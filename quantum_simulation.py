#!/usr/bin/env python3

import os
import sys

from exceptions import QuantumSimulationError
from molecule_simulation import QuantumSimulation
from chemical_reaction import ChemicalReaction
from mol_optimizer import *
from functions import *


import argparse

from pennylane import numpy as np

def main():
    molecules, args, type_sim = from_user_input()

    if type_sim == 'optimization':

        for mol in molecules:
            optimized_geometry, energy = optimize_geometry(mol)
            print(f"Molécula: {mol.symbols}")
            print(f"Energía mínima: {energy}")
            print(f"Geometría optimizada: {optimized_geometry}")

if __name__ == "__main__":
    main()
