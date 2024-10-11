#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import argparse

def run_command(command):
    """
    Executes a command in the command line and captures the output.

    Args:
        command (str): The command to execute.

    Returns:
        output (str): The output of the command.
    """
    print(f"Executing: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output = ''
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            output += line
    return output

def run_tests(test_category, show_plots):
    """
    Executes a series of tests based on the selected category.

    Args:
        test_category (str): Category of tests to run ('molecular_fast', 'molecular_slow',
                             'reaction_fast', 'reaction_slow', 'mixed_fast', 'mixed_full').
        show_plots (bool): Indicates whether to show plots during the tests.
    """
    # Define test commands for each category
    test_commands = {
        'molecular_fast': [
            # Simulación Molecular Rápida 1: Simular H2 con GradientDescent y mostrar plot
            "python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot",

            # Simulación Molecular Rápida 2: Simular LiH con Adam y guardar resultados
            "python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 50 --save --save_dir results/results_lih_fast",

            # Simulación Molecular Rápida 3: Simular He con GradientDescent sin plot
            "python quantum_simulation.py --molecule He --optimizer GradientDescent --max_iterations 50 --save --save_dir results/results_he_fast",
        ],
        'molecular_slow': [
            # Simulación Molecular Lenta 1: Simular CH4 con Adam y mayor número de iteraciones
            "python quantum_simulation.py --molecule CH4 --optimizer Adam --max_iterations 200 --save --save_dir results/results_ch4_slow --plot",

            # Simulación Molecular Lenta 2: Simular H2O2 con RMSProp y pasos más pequeños
            "python quantum_simulation.py --molecule H2O2 --optimizer RMSProp --stepsize 0.1 --max_iterations 150 --save --save_dir results/results_h2o2_slow --plot",

            # Simulación Molecular Lenta 3: Simular Cl2 con NesterovMomentum y alta tolerancia
            "python quantum_simulation.py --molecule Cl2 --optimizer NesterovMomentum --conv_tol 1e-8 --max_iterations 200 --save --save_dir results/results_cl2_slow --plot",
        ],
        'reaction_fast': [
            # Simulación de Reacción Rápida 1: H2 -> H + H
            'python quantum_simulation.py --reaction "H2 -> H + H" --optimizer Adam --max_iterations 100 --save --plot',

            # Simulación de Reacción Rápida 2: He + H -> HeH+
            'python quantum_simulation.py --reaction "He + H -> HeH+" --optimizer GradientDescent --max_iterations 100 --save --plot',

            # Simulación de Reacción Rápida 3: Li + H -> LiH
            'python quantum_simulation.py --reaction "Li + H -> LiH" --optimizer Adam --max_iterations 100 --save --plot',
        ],
        'reaction_slow': [
            # Simulación de Reacción Lenta 1: H2 + O2 -> H2O
            'python quantum_simulation.py --reaction "H2 + O2 -> H2O" --optimizer Adam --max_iterations 200 --save --save_dir results/results_h2o_slow --plot',

            # Simulación de Reacción Lenta 2: H2 -> H2+ + e-
            'python quantum_simulation.py --reaction "H2 -> H2+ + e-" --optimizer RMSProp --stepsize 0.05 --max_iterations 150 --save --save_dir results/results_h2_plus_slow --plot',

            # Simulación de Reacción Lenta 3: CH4 + Cl2 -> CH3Cl + HCl
            'python quantum_simulation.py --reaction "CH4 + Cl2 -> CH3Cl + HCl" --optimizer NesterovMomentum --max_iterations 250 --save --save_dir results/results_ch4_cl2_slow --plot',
        ],
        'mixed_fast': [
            # Simulación Mixta Rápida 1: Simular H2 y luego reaccionar H2 -> H + H
            'python quantum_simulation.py --molecule H2 --optimizer Adam --max_iterations 50 --save --save_dir results/results_h2_mixed_fast',
            'python quantum_simulation.py --reaction "H2 -> H + H" --optimizer Adam --max_iterations 100 --save --save_dir results/results_h2_h_h_mixed_fast --plot',

            # Simulación Mixta Rápida 2: Simular He y luego reaccionar He + H -> HeH+
            'python quantum_simulation.py --molecule He --optimizer GradientDescent --max_iterations 50 --save --save_dir results/results_he_mixed_fast',
            'python quantum_simulation.py --reaction "He + H -> HeH+" --optimizer GradientDescent --max_iterations 100 --save --save_dir results/results_heh_plus_mixed_fast --plot',
        ],
        'mixed_full': [
            # Simulación Mixta Completa 1: Simular H2, LiH y luego reaccionar H2 + O2 -> H2O
            'python quantum_simulation.py --molecule H2 --optimizer GradientDescent --max_iterations 100 --save --save_dir results/results_h2_mixed_full',
            'python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 150 --save --save_dir results/results_lih_mixed_full',
            'python quantum_simulation.py --reaction "H2 + O2 -> H2O" --optimizer Adam --max_iterations 200 --save --save_dir results/results_h2o_mixed_full --plot',

            # Simulación Mixta Completa 2: Simular CH4, Cl2 y luego reaccionar CH4 + Cl2 -> CH3Cl + HCl
            'python quantum_simulation.py --molecule CH4 --optimizer Adam --max_iterations 150 --save --save_dir results/results_ch4_mixed_full',
            'python quantum_simulation.py --molecule Cl2 --optimizer NesterovMomentum --max_iterations 200 --save --save_dir results/results_cl2_mixed_full',
            'python quantum_simulation.py --reaction "CH4 + Cl2 -> CH3Cl + HCl" --optimizer NesterovMomentum --max_iterations 250 --save --save_dir results/results_ch4_cl2_mixed_full --plot',
        ],
    }

    # Validar la categoría seleccionada
    if test_category not in test_commands:
        print(f"Categoría de prueba '{test_category}' no reconocida.")
        print("Categorías disponibles:")
        for key in test_commands.keys():
            print(f" - {key}")
        sys.exit(1)

    selected_tests = test_commands[test_category]

    # Ejecutar cada comando
    for idx, command in enumerate(selected_tests, 1):
        print(f"\n=== Ejecutando Prueba {idx}/{len(selected_tests)} ===")
        # Modificar el comando si no se desean mostrar plots
        if not show_plots and '--plot' in command:
            command = command.replace('--plot', '')
        output = run_command(command)

        # Verificar si se han guardado resultados
        if '--save_dir' in command:
            # Extraer el directorio de resultados
            parts = command.split('--save_dir')
            if len(parts) > 1:
                save_dir = parts[1].strip().split(' ')[0]
                print(f"Resultados guardados en: {save_dir}")

        print(f"=== Fin de Prueba {idx}/{len(selected_tests)} ===\n")
        # Esperar un momento entre pruebas para evitar sobrecarga
        time.sleep(2)

def main():
    # Crear el analizador de argumentos
    parser = argparse.ArgumentParser(description='Ejecuta simulaciones cuánticas de moléculas y reacciones.')
    parser.add_argument('--test_category', type=str, choices=[
        'molecular_fast', 'molecular_slow',
        'reaction_fast', 'reaction_slow',
        'mixed_fast', 'mixed_full'
    ], required=True, help='Categoría de prueba a ejecutar.')
    parser.add_argument('--show_plots', action='store_true', help='Mostrar plots durante las pruebas.')

    args = parser.parse_args()

    run_tests(test_category=args.test_category, show_plots=args.show_plots)

if __name__ == "__main__":
    main()
