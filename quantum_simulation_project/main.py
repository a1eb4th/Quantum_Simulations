import os
import sys
import shutil
import cProfile
import pstats
from io import StringIO
import time

TEMP_RESULTS_DIR = "step_size/results_H2_QNG"

def main():
    from config.config_functions import from_user_input
    from modules.opt_mol import mol_optimizer

    molecules, optimizers, ansatz_list = from_user_input()
    mol_optimizer(molecules, optimizers, TEMP_RESULTS_DIR, ansatz_list)

if __name__ == "__main__":
    
    if os.path.exists(TEMP_RESULTS_DIR):
        shutil.rmtree(TEMP_RESULTS_DIR)
    os.makedirs(TEMP_RESULTS_DIR)

    output_file_path = os.path.join(TEMP_RESULTS_DIR, "output.txt")

    with open(output_file_path, "w", encoding="utf-8") as f:
        pass

    output_file = open(output_file_path, "a", buffering=1, encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = output_file

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        main()
    finally:
        profiler.disable()

        profiler_output_path = os.path.join(TEMP_RESULTS_DIR, "profile_output_autograd.txt")
        with open(profiler_output_path, "w", encoding="utf-8") as f:
            profiler_output = StringIO()
            stats = pstats.Stats(profiler, stream=profiler_output)
            stats.strip_dirs()
            stats.sort_stats('cumulative')

            filtered_output = StringIO()
            stats.stream = filtered_output
            stats.print_stats("mol_optimizer")  
            filtered_results = filtered_output.getvalue()

            filtered_report_path = os.path.join(TEMP_RESULTS_DIR, "filtered_report_autograd.txt")
            with open(filtered_report_path, "w", encoding="utf-8") as filtered_file:
                filtered_file.write(filtered_results)

            stats.stream = profiler_output
            stats.print_stats()
            f.write(profiler_output.getvalue())

        print(f"Report completely saved on: {profiler_output_path}")
        print(f"Filtered report saved on: {filtered_report_path}")

    # Restaurar stdout
    sys.stdout = original_stdout
    output_file.close()

    excluded_files = ["filtered_report_autograd.txt", "profile_output_autograd.txt"]

    for file_name in os.listdir(TEMP_RESULTS_DIR):
        if file_name.endswith(".txt") and file_name != "output.txt" and file_name not in excluded_files:
            os.remove(os.path.join(TEMP_RESULTS_DIR, file_name))
    