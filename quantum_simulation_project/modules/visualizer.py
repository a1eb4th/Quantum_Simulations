import os
import pandas as pd
import portalocker
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import random

def write_simulation_times(symbols, interface, optimizer_name, execution_times):
    """
    Records and appends the execution times of various functions during the simulation to a CSV file.

    Args:
        symbols (list of str): List of atomic symbols in the molecule (e.g., ['H', 'O']).
        interface (str): The interface used for the simulation (e.g., 'autograd').
        optimizer_name (str): Name of the optimizer used (e.g., 'Adam').
        execution_times (dict): Dictionary containing execution times with function names as keys and time durations in seconds as values.

    Returns:
        None
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
            df_combined = df_existing.join(df_new, how='outer')
        else:
            df_combined = df_new
        fh.seek(0)
        fh.truncate()
        df_combined.to_csv(fh)

def _get_all_energies(results):
    """
    This helper function traverses the nested results dictionary to collect all energy history lists
    from different interfaces and optimizers.

    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.

    Returns:
        list of float: A flattened list of all energy values across all optimizations.
    """
    all_energies = []
    for interface_results in results.values():
        for data in interface_results.values():
            energy_hist = data.get("energy_history", [])
            all_energies.extend(energy_hist)
    return all_energies

def _compute_offset_for_log(results):
    """
    Computes an appropriate offset to ensure all energy values are positive for logarithmic plotting.
    
    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.

    Returns:
        float: The computed offset to be added to energy values for log-scale plotting.
    """
    energies = _get_all_energies(results)
    if not energies:
        return 1e-9
    min_energy = min(energies)
    offset = 1e-9
    if min_energy <= 0:
        offset = abs(min_energy) + 1e-9
    return offset

def visualize_results(results, symbols, results_dir):
    """
    Generates and saves visualizations of the energy evolution, interatomic distances, and final geometries. Creates two plots for energy evolution (linear and logarithmic scales with offset), interatomic distance plots, and final geometry visualizations. All plots are saved in the specified results directory.

    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.
        symbols (list of str): List of atomic symbols in the molecule (e.g., ['H', 'O']).
        results_dir (str): Directory path where the generated plots will be saved.

    Returns:
        None
    """
    offset = _compute_offset_for_log(results)

    plt.figure(figsize=(10, 6))
    plt.title('Energy Evolution (Linear Scale)', fontsize=16)
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_hist = data.get("energy_history", [])
            if energy_hist:
                label = f"{optimizer_name} ({interface})"
                plt.plot(energy_hist, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                plt.plot(len(energy_hist)-1, energy_hist[-1], marker='*', markersize=10, color='red')
    plt.xlabel('Optimization Step', fontsize=14)
    plt.ylabel('Energy (Ha)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_evolution_linear.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.title('Energy Evolution (Log Scale with Offset)', fontsize=16)
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_hist = data.get("energy_history", [])
            if energy_hist:
                energy_offset = [e + offset for e in energy_hist]
                label = f"{optimizer_name} ({interface})"
                plt.plot(energy_offset, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                plt.plot(len(energy_offset)-1, energy_offset[-1], marker='*', markersize=10, color='red')
    plt.xlabel('Optimization Step', fontsize=14)
    ylabel = 'Energy (Ha)' + (f' + {offset:.2e}' if offset > 0 else '')
    plt.ylabel(ylabel, fontsize=14)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_evolution_log_offset.png"))
    plt.close()

    visualize_interatomic_distances(results, symbols, results_dir)
    visualize_final_geometries(results, symbols, results_dir)
    visualize_energy_vs_time(results, results_dir, offset)


def visualize_energy_vs_time(results, results_dir, offset=1e-9):
    """
    Generates and saves plots of energy versus time in both linear and logarithmic scales.

    This function creates two plots:
        1. Energy vs Time (Linear Scale)
        2. Energy vs Time (Logarithmic Scale with Offset)

    It extracts execution times from the `execution_times` dictionary within the results and correlates them with the corresponding energy values to plot the energy evolution over time.

    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.
        results_dir (str): Directory path where the generated plots will be saved.
        offset (float, optional): Offset added to energy values to ensure positivity for log-scale plotting. Defaults to 1e-9.

    Returns:
        None
    """
    def parse_substep_key(k):
                part_iter, part_sub = k.split(" - ")
                i = int(part_iter.split()[1]) 
                j = int(part_sub.split()[1])
                return (i, j)
    # ============= LINEAR PLOT =============
    plt.figure(figsize=(10, 6))
    plt.title('Energy vs Time (Linear)', fontsize=16)

    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_history = data.get("energy_history", [])
            execution_times = data.get("execution_times", {})

            substep_keys = [k for k in execution_times.keys() if 'Substep' in k]

            substep_keys.sort(key=lambda x: parse_substep_key(x))

            cumulative_times = []
            current_time = 0.0
            for key in substep_keys:
                duration = execution_times[key]
                current_time += duration
                cumulative_times.append(current_time)

            min_length = min(len(energy_history), len(cumulative_times))
            time_slice = cumulative_times[:min_length]
            energy_slice = energy_history[:min_length]

            if len(energy_slice) > 0:
                label = f"{optimizer_name} ({interface})"
                plt.plot(time_slice, energy_slice, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                plt.plot(time_slice[-1], energy_slice[-1], marker='*', markersize=10, color='red')

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Energy (Ha)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_vs_time_linear.png"))
    plt.close()

    # ============= LOGARITHMIC PLOT (with offset) =============
    plt.figure(figsize=(10, 6))
    plt.title('Energy vs Time (Log Scale with Offset)', fontsize=16)

    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_history = data.get("energy_history", [])
            execution_times = data.get("execution_times", {})

            substep_keys = [k for k in execution_times.keys() if 'Substep' in k]
            substep_keys.sort(key=lambda x: parse_substep_key(x))

            cumulative_times = []
            current_time = 0.0
            for key in substep_keys:
                duration = execution_times[key]
                current_time += duration
                cumulative_times.append(current_time)

            min_length = min(len(energy_history), len(cumulative_times))
            time_slice = cumulative_times[:min_length]
            energy_slice = energy_history[:min_length]

            energy_offset = [e + offset for e in energy_slice]

            min_val = min(energy_offset) if energy_offset else 1
            if min_val <= 0:
                shift = 1e-9 - min_val
                energy_offset = [val + shift for val in energy_offset]

            if len(energy_offset) > 0:
                label = f"{optimizer_name} ({interface})"
                plt.plot(time_slice, energy_offset, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                plt.plot(time_slice[-1], energy_offset[-1], marker='*', markersize=10, color='red')

    plt.xlabel('Time (s)', fontsize=14)
    ylabel = f'Energy (Ha) + {offset:.2e}' if offset != 0 else 'Energy (Ha)'
    plt.ylabel(ylabel, fontsize=14)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_vs_time_log_offset.png"))
    plt.close()

def visualize_interatomic_distances(results, symbols, results_dir):
    """
    For each unique pair of atoms in the molecule, this function plots the distance between them as it evolves over the optimization steps. Each optimizer and interface combination is represented with distinct labels in the plots.

    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.
        symbols (list of str): List of atomic symbols in the molecule (e.g., ['H', 'O']).
        results_dir (str): Directory path where the generated plots will be saved.

    Returns:
        None
    """
    num_atoms = len(symbols)
    atom_pairs = list(combinations(range(num_atoms), 2))
    if len(atom_pairs) == 0:
        return
    for pair in atom_pairs:
        i, j = pair
        plt.figure(figsize=(10, 6))
        plt.title(f'Distance {symbols[i]}-{symbols[j]} During Optimization', fontsize=16)
        for interface, interface_results in results.items():
            for optimizer_name, data in interface_results.items():
                x_history = np.array(data.get("x_history", []))
                if len(x_history) > 0:
                    coords = x_history.reshape((len(x_history), num_atoms, 3))
                    dist = np.linalg.norm(coords[:, j, :] - coords[:, i, :], axis=1)
                    label = f"{optimizer_name} ({interface})"
                    plt.plot(dist, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                    plt.plot(len(dist)-1, dist[-1], marker='*', markersize=10, color='red')
        plt.xlabel('Optimization Step', fontsize=14)
        plt.ylabel('Distance (Å)', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"distance_{symbols[i]}_{symbols[j]}.png"))
        plt.close()

def visualize_final_geometries(results, symbols, results_dir):
    """
    This function visualizes the final positions of atoms in 3D space, allowing for a comparison of optimized molecular geometries obtained from different optimizers and interfaces.

    Args:
        results (dict): Nested dictionary containing optimization results organized by interfaces and optimizers.
        symbols (list of str): List of atomic symbols in the molecule (e.g., ['H', 'O']).
        results_dir (str): Directory path where the generated plot will be saved.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    markers = ['o', '^', 's', 'D', 'P', 'X', '*']
    all_coords = []
    for interface_results in results.values():
        for data in interface_results.values():
            final_coords = data['final_x'].reshape(-1, 3)
            all_coords.append(final_coords)
    if all_coords:
        all_coords = np.concatenate(all_coords)
        max_range = np.ptp(all_coords, axis=0).max()
        size_scale = 200 / max_range if max_range != 0 else 200
    else:
        size_scale = 100
    used_labels = set()
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            final_coords = data["final_x"].reshape(-1, 3)
            random.seed(str((optimizer_name, interface)))
            color = random.choice(colors)
            marker = random.choice(markers)
            center = final_coords.mean(axis=0)
            distances = np.linalg.norm(final_coords - center, axis=1)
            sizes = distances * size_scale + 50
            label_shown = False
            for i, atom in enumerate(symbols):
                label = f"{atom} - {optimizer_name} ({interface})" if not label_shown else ""
                label_shown = True
                ax.scatter(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2], color=color, marker=marker, s=sizes[i], label=label if label and label not in used_labels else "")
                if label and label not in used_labels:
                    used_labels.add(label)
                ax.text(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                        f"{atom}", size=8, color=color)
    ax.set_xlabel('x (Å)', fontsize=12)
    ax.set_ylabel('y (Å)', fontsize=12)
    ax.set_zlabel('z (Å)', fontsize=12)
    ax.set_title('Final Geometries of Optimizers and Interfaces', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_geometries_3D.png"))
    plt.close()
