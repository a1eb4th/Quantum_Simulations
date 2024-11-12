import os
import sys
import shutil

# Configuración de directorio y archivo de salida
TEMP_RESULTS_DIR = "temp_results_jax"

# Crear el directorio si no existe
if not os.path.exists(TEMP_RESULTS_DIR):
    os.makedirs(TEMP_RESULTS_DIR)

# Abrir archivo para registrar la salida
output_file_path = os.path.join(TEMP_RESULTS_DIR, "output.txt")
output_file = open(output_file_path, "w", buffering=1)  # Line-buffered mode
original_stdout = sys.stdout  # Guardar salida estándar original
sys.stdout = output_file  # Redirigir salida estándar al archivo

try:
    def main():
        from functions import from_user_input  # Importa otras dependencias necesarias
        from mol_optimizer import mol_optimizer

        molecules, args, type_sim = from_user_input()
        
        if type_sim == 'optimization':
            mol_optimizer(molecules)

    if __name__ == "__main__":
        main()

finally:
    # Restaurar la salida estándar y cerrar el archivo
    sys.stdout = original_stdout
    output_file.close()
