import os
import sys
import shutil 
import cProfile
import pstats
from io import StringIO


TEMP_RESULTS_DIR = "temp_results_autograd"


if os.path.exists(TEMP_RESULTS_DIR):
    for root, dirs, files in os.walk(TEMP_RESULTS_DIR):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))
else:
    os.makedirs(TEMP_RESULTS_DIR)


output_file_path = os.path.join(TEMP_RESULTS_DIR, "output.txt")
output_file = open(output_file_path, "w", buffering=1)
original_stdout = sys.stdout
sys.stdout = output_file
results_dir = TEMP_RESULTS_DIR

def main():
    from config.config_functions import from_user_input
    from modules.opt_mol import mol_optimizer

    molecules, optimizers, ansatz_list = from_user_input()
    mol_optimizer(molecules, optimizers, results_dir, ansatz_list)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    try:
        profiler.enable()

        main()

    finally:
        profiler.disable()

        profiler_output_path = os.path.join(results_dir, "profile_output_autograd.txt")
        with open(profiler_output_path, "w") as f:
            profiler_output = StringIO()
            stats = pstats.Stats(profiler, stream=profiler_output)

            stats.strip_dirs()
            stats.sort_stats('cumulative')

            filtered_output = StringIO()
            stats.stream = filtered_output
            stats.print_stats("mol_optimizer")
            filtered_results = filtered_output.getvalue()

            filtered_report_path = os.path.join(results_dir, "filtered_report_autograd.txt")
            with open(filtered_report_path, "w") as filtered_file:
                filtered_file.write(filtered_results)

            stats.stream = profiler_output
            stats.print_stats()
            f.write(profiler_output.getvalue())

        print(f"Report completely saved on: {profiler_output_path}")
        print(f"Filtered report saved on: {filtered_report_path}")

    sys.stdout = original_stdout
    output_file.close()

