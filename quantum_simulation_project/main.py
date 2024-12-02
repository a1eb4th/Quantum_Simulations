import os
import sys
import cProfile
import pstats
from io import StringIO

# Configuración de directorio y archivo de salida
TEMP_RESULTS_DIR = "temp_results_autograd"

# Crear el directorio si no existe
if not os.path.exists(TEMP_RESULTS_DIR):
    os.makedirs(TEMP_RESULTS_DIR)

# Abrir archivo para registrar la salida
output_file_path = os.path.join(TEMP_RESULTS_DIR, "output.txt")
output_file = open(output_file_path, "w", buffering=1)
original_stdout = sys.stdout
sys.stdout = output_file

def main():
    from config.config_functions import from_user_input
    from modules.opt_mol import mol_optimizer

    molecules, args, type_sim = from_user_input()

    if type_sim == 'optimization':
        mol_optimizer(molecules)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    try:
        profiler.enable()

        main()

    finally:
        profiler.disable()

        profiler_output_path = os.path.join(TEMP_RESULTS_DIR, "profile_output_autograd.txt")
        with open(profiler_output_path, "w") as f:
            profiler_output = StringIO()
            stats = pstats.Stats(profiler, stream=profiler_output)

            # Filtrar y analizar funciones específicas de 'mol_optimizer'
            stats.strip_dirs()
            stats.sort_stats('cumulative')

            # Filtrar las funciones definidas dentro de 'mol_optimizer'
            filtered_output = StringIO()
            stats.stream = filtered_output
            stats.print_stats("mol_optimizer")
            filtered_results = filtered_output.getvalue()

            # Escribir el resultado filtrado en un archivo separado
            filtered_report_path = os.path.join(TEMP_RESULTS_DIR, "filtered_report_autograd.txt")
            with open(filtered_report_path, "w") as filtered_file:
                filtered_file.write(filtered_results)

            # Escribir el reporte completo de perfil
            stats.stream = profiler_output
            stats.print_stats()
            f.write(profiler_output.getvalue())

        print(f"Reporte completo guardado en: {profiler_output_path}")
        print(f"Reporte filtrado guardado en: {filtered_report_path}")

    sys.stdout = original_stdout
    output_file.close()

