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

def run_tests(test_type, show_plots):
    """
    Executes a series of tests with different configurations.

    Args:
        test_type (str): Type of test to run ('fast' or 'full').
        show_plots (bool): Indicates whether to show plots during the tests.
    """
    # Fast tests
    fast_tests = [
        # Test 1: Simulate H2 with GradientDescent and show the plot
        "python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot",

        # Test 2: Simulate LiH with Adam and save results
        "python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 50 --save --save_dir results/results_lih_fast",

        # Test 3: Test error handling: undefined molecule
        "python quantum_simulation.py --molecule XYZ --optimizer Adam",

        # Test 4: Test error handling: invalid optimizer
        "python quantum_simulation.py --molecule H2 --optimizer UnknownOptimizer",
    ]

    # Full tests
    full_tests = [
        # Test 1: Simulate H2 with GradientDescent and show the plot
        "python quantum_simulation.py --molecule H2 --optimizer GradientDescent --plot",

        # Test 2: Simulate LiH with Adam, increase iterations, and save results
        "python quantum_simulation.py --molecule LiH --optimizer Adam --max_iterations 100 --save --save_dir results/results_lih",

        # Test 3: Simulate H2O with RMSProp, change stepsize, and show the plot
        "python quantum_simulation.py --molecule H2O --optimizer RMSProp --stepsize 0.1 --plot",

        # Test 4: Simulate H2 with basis '6-31G' and save results
        "python quantum_simulation.py --molecule H2 --basis_set 6-31G --save --save_dir results/results_h2_631G",

        # Test 5: Simulate LiH with NesterovMomentum, change conv_tol, and show the plot
        "python quantum_simulation.py --molecule LiH --optimizer NesterovMomentum --conv_tol 1e-8 --plot",

        # Test 6: Simulate CH4 added to the main script
        "python quantum_simulation.py --molecule CH4 --optimizer Adam --max_iterations 100 --plot",

        # Test 7: Test error handling: undefined molecule
        "python quantum_simulation.py --molecule XYZ --optimizer Adam",

        # Test 8: Test error handling: invalid optimizer
        "python quantum_simulation.py --molecule H2 --optimizer UnknownOptimizer",

        # Test 9: Simulate H2 with small stepsize and high tolerance
        "python quantum_simulation.py --molecule H2 --optimizer GradientDescent --stepsize 0.01 --conv_tol 1e-4 --max_iterations 200 --plot",

        # Test 10: Simulate H2O with basis cc-pVDZ (if supported)
        "python quantum_simulation.py --molecule H2O --basis_set cc-pVDZ --optimizer Adam --max_iterations 100 --plot",

        # Test 11: Test saving and loading results
        "python quantum_simulation.py --molecule H2 --optimizer Adam --save --save_dir results/results_h2_adam",
    ]

    # Select the list of tests based on the type
    if test_type == 'fast':
        selected_tests = fast_tests
    elif test_type == 'full':
        selected_tests = full_tests
    else:
        print(f"Test type '{test_type}' not recognized. Using 'fast' tests by default.")
        selected_tests = fast_tests

    # Execute each command
    for idx, command in enumerate(selected_tests, 1):
        print(f"\n=== Running Test {idx}/{len(selected_tests)} ===")
        # Add or remove the '--plot' argument based on 'show_plots'
        if not show_plots and '--plot' in command:
            command = command.replace('--plot', '')
        output = run_command(command)
        
        # Check if results have been saved
        if '--save_dir' in command:
            # Extract the results directory
            save_dir = command.split('--save_dir')[-1].strip().split(' ')[0]
            print(f"Results saved in: {save_dir}")

        print(f"=== End of Test {idx}/{len(selected_tests)} ===\n")
        # Wait a moment between tests to avoid overloading
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs tests for quantum_simulation.py.')
    parser.add_argument('--test_type', type=str, choices=['fast', 'full'], default='fast',
                        help='Type of test to run: "fast" or "full". Default is "fast".')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during the tests.')
    args = parser.parse_args()

    run_tests(test_type=args.test_type, show_plots=args.show_plots)
