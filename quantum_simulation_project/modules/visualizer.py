import os
from pennylane import numpy as np
import matplotlib.pyplot as plt
import portalocker
import pandas as pd

TEMP_RESULTS_DIR = "temp_results_autograd"
def write_simulation_times(symbols, interface, optimizer_name, execution_times):
    """
    Writes the execution time of a simulation to a CSV file.
    If the simulation ID already exists, its execution time is updated.
    Otherwise, a new entry is appended.

    :param csv_file: Path to the CSV file.
    :param simulation_id: Unique identifier for the simulation.
    :param execution_time: Execution time of the simulation.
    """
    filename = "execution_times.csv"
    simulation_id = f"{symbols}_{interface}_{optimizer_name}"

    df_new = pd.DataFrame(list(execution_times.items()), columns=['Function', simulation_id])
    df_new.set_index('Function', inplace=True)

    with portalocker.Lock(filename, 'a+', timeout=10) as fh:
        fh.seek(0)
        if os.path.getsize(filename) > 0:
            df_existing = pd.read_csv(fh, index_col='Function')

            if simulation_id in df_existing.columns:
                df_existing.drop(columns=[simulation_id], inplace=True)

            # Combining DataFrames
            df_combined = df_existing.join(df_new, how='outer')
        else:
            df_combined = df_new

        fh.seek(0)
        fh.truncate()
        df_combined.to_csv(fh)

def visualize_results(results, symbols):
    """
    Generates plots to compare optimizers and visualize the evolution of coordinates.

    Visualization code to compare results across interfaces and optimizers
    Plot energy over optimization steps comparing optimizers and interfaces

    Args:
        results (dict): Dictionary with results for each optimizer.
        symbols (list): List of atomic symbols.
    """
    plt.figure(figsize=(10, 6))
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            if data["energy_history"]:
                label = f"{optimizer_name} ({interface})"
                plt.plot(data["energy_history"], label=label)
    plt.xlabel('Optimization Step', fontsize=14)
    plt.ylabel('Energy (Ha)', fontsize=14)
    plt.title('Energy Evolution During Optimization', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TEMP_RESULTS_DIR, "energy_evolution.png"))
    plt.close()

    # Visualization of nuclear coordinates in a single figure
    num_atoms = len(symbols)
    axes = ['x', 'y', 'z']
    num_axes = len(axes)

    fig, axs = plt.subplots(num_atoms, num_axes, figsize=(5*num_axes, 4*num_atoms), sharex=True)

    for atom_index in range(num_atoms):
        for axis_index, axis_name in enumerate(axes):
            ax = axs[atom_index, axis_index] if num_atoms > 1 else axs[axis_index]
            for interface, interface_results in results.items():
                for optimizer_name, data in interface_results.items():
                    x_history = np.array(data["x_history"])
                    num_iterations = len(x_history)
                    coord_history = x_history[:, 3*atom_index + axis_index]
                    label = f"{optimizer_name} ({interface})"
                    ax.plot(range(num_iterations), coord_history, label=label)
            if atom_index == num_atoms - 1:
                ax.set_xlabel('Optimization Step', fontsize=12)
            ax.set_ylabel(f'{symbols[atom_index]} - {axis_name} (Å)', fontsize=12)
            ax.grid(True)
            if atom_index == 0 and axis_index == num_axes - 1:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(TEMP_RESULTS_DIR, "nuclear_coordinates.png"))
    plt.close()

    # 3D Visualization of the final geometries
    visualize_final_geometries(results, symbols)

def visualize_final_geometries(results, symbols):
    """
    Generates a 3D plot showing the final geometries for each optimizer and interface.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Gradient Descent': 'r', 'Adam': 'g', 'Quantum Natural Gradient': 'b'}
    markers = {'autograd': 'o', 'jax': '^'}

    # Get the range of coordinates to adjust point sizes
    all_coords = []
    for interface_results in results.values():
        for data in interface_results.values():
            final_coords = data['final_x'].reshape(-1, 3)
            all_coords.append(final_coords)
    all_coords = np.concatenate(all_coords)
    max_range = np.ptp(all_coords, axis=0).max()


    size_scale = 200 / max_range 

    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            final_coords = data["final_x"].reshape(-1, 3)
            color = colors.get(optimizer_name, 'k')
            marker = markers.get(interface, 's')
            
            center = final_coords.mean(axis=0)
            distances = np.linalg.norm(final_coords - center, axis=1)
            sizes = distances * size_scale + 50

            for i, atom in enumerate(symbols):
                label = f"{atom} - {optimizer_name} ({interface})" if i == 0 else ""
                ax.scatter(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                           color=color, marker=marker, s=sizes[i], label=label)
                ax.text(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                        f"{atom}", size=10, color=color)

    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title('Final Geometries of Optimizers and Interfaces')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(os.path.join(TEMP_RESULTS_DIR, "final_geometries_3D.png"))
    plt.close()