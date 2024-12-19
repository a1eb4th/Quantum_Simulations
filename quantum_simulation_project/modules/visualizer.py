import os
import pandas as pd
import portalocker
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import random

def write_simulation_times(symbols, interface, optimizer_name, execution_times):
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
    all_energies = []
    for interface_results in results.values():
        for data in interface_results.values():
            energy_hist = data.get("energy_history", [])
            all_energies.extend(energy_hist)
    return all_energies

def _compute_offset_for_log(results):
    energies = _get_all_energies(results)
    if not energies:
        return 1e-6
    min_energy = min(energies)
    offset = 1e-6
    if min_energy <= 0:
        offset = abs(min_energy) + 1e-6
    return offset

def visualize_results(results, symbols, results_dir):
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

def visualize_energy_vs_time(results, results_dir, offset=1e-6):
    plt.figure(figsize=(10, 6))
    plt.title('Energy vs Time (Linear)', fontsize=16)
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_history = data.get("energy_history", [])
            execution_times = data.get("execution_times", {})
            iteration_keys = [k for k in execution_times.keys() if k.startswith("Iteration")]
            iteration_keys.sort(key=lambda x: int(x.split()[1]))
            if energy_history and iteration_keys:
                cumulative_times = []
                current_time = 0.0
                for i_key in iteration_keys:
                    current_time += execution_times[i_key]
                    cumulative_times.append(current_time)
                min_length = min(len(energy_history), len(cumulative_times))
                time_slice = cumulative_times[:min_length]
                energy_slice = energy_history[:min_length]
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

    plt.figure(figsize=(10,6))
    plt.title('Energy vs Time (Log Scale with Offset)', fontsize=16)
    for interface, interface_results in results.items():
        for optimizer_name, data in interface_results.items():
            energy_history = data.get("energy_history", [])
            execution_times = data.get("execution_times", {})
            iteration_keys = [k for k in execution_times.keys() if k.startswith("Iteration")]
            iteration_keys.sort(key=lambda x: int(x.split()[1]))
            if energy_history and iteration_keys:
                cumulative_times = []
                current_time = 0.0
                for i_key in iteration_keys:
                    current_time += execution_times[i_key]
                    cumulative_times.append(current_time)
                min_length = min(len(energy_history), len(cumulative_times))
                time_slice = cumulative_times[:min_length]
                energy_offset = [e + offset for e in energy_history[:min_length]]
                # Asegurar que todos los valores sean positivos
                if min(energy_offset) <= 0:
                    add_val = 1e-6 - min(energy_offset) if min(energy_offset) <= 0 else 0
                    energy_offset = [val + add_val for val in energy_offset]
                label = f"{optimizer_name} ({interface})"
                plt.plot(time_slice, energy_offset, marker='o', linestyle='-', linewidth=1.0, markersize=4, label=label)
                plt.plot(time_slice[-1], energy_offset[-1], marker='*', markersize=10, color='red')
    plt.xlabel('Time (s)', fontsize=14)
    ylabel = 'Energy (Ha)' + (f' + {offset:.2e}' if offset > 0 else '')
    plt.ylabel(ylabel, fontsize=14)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_vs_time_log_offset.png"))
    plt.close()

def visualize_interatomic_distances(results, symbols, results_dir):
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
                ax.scatter(final_coords[i, 0], final_coords[i, 1], final_coords[i, 2],
                           color=color, marker=marker, s=sizes[i],
                           label=label if label and label not in used_labels else "")
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
