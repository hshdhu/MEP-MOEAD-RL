import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Import
from utils.config_loader import load_config
from utils.draw import (
    plot_environment_image,
    plot_pareto_fronts_by_generation,
    plot_pareto_size_history,
    plot_hypervolume_history
)
from algorithm.moead import MOEAD

def get_result_folder(num_sensors):
    if num_sensors <= 50:
        return "50"
    elif num_sensors <= 100:
        return "100"
    elif num_sensors <= 150:
        return "150"
    else:
        return "200"


def save_json(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"JSON save error: {e}")


def find_knee_solution(solutions: List[Dict]):
    if not solutions: return None
    exposures = np.array([s['exposure'] for s in solutions])
    lengths = np.array([s['length'] for s in solutions])

    norm_len = (lengths - lengths.min()) / (
            lengths.max() - lengths.min()) if lengths.max() != lengths.min() else np.zeros_like(lengths)
    norm_exp = (exposures.max() - exposures) / (
            exposures.max() - exposures.min()) if exposures.max() != exposures.min() else np.zeros_like(exposures)

    distances = np.sqrt(norm_len ** 2 + norm_exp ** 2)
    return solutions[np.argmin(distances)]


def plot_final_solutions(env, solutions, best_solution, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, zorder=5)

    for s in env.sensors:
        ax.add_patch(
            plt.Circle((s.position.x, s.position.y), s.radius, facecolor=(0.4, 0.7, 1.0, 0.2), edgecolor='none',
                       zorder=2))

    for sol in solutions:
        path = np.array(sol['path'])
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], color='green', linewidth=0.8, alpha=0.2, zorder=3)

    if best_solution:
        path = np.array(best_solution['path'])
        ax.plot(path[:, 0], path[:, 1], color='#D32F2F', linewidth=3.0, label='Best Balance', zorder=10)
        ax.set_title(f"MOEA/D Result\nExp={best_solution['exposure']:.2f}, Len={best_solution['length']:.2f}")

    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- SNAPSHOT PLOTTING ---
def plot_snapshot(env, solutions, gen, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, zorder=5)
    for s in env.sensors:
        ax.add_patch(plt.Circle((s.position.x, s.position.y), s.radius,
                                facecolor=(0.4, 0.7, 1.0, 0.2), edgecolor='none', zorder=2))

    # Draw paths
    for sol in solutions:
        path = np.array(sol['path'])
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], color='green', linewidth=0.5, alpha=0.3, zorder=3)

    if gen > 0:
        best_sol = find_knee_solution(solutions)
        if best_sol:
            path = np.array(best_sol['path'])
            ax.plot(path[:, 0], path[:, 1], color='#D32F2F', linewidth=2.0, label='Best Current', zorder=10)
            ax.set_title(f"Generation {gen}\nBest: Exp={best_sol['exposure']:.1f}, Len={best_sol['length']:.1f}")
        else:
            ax.set_title(f"Generation {gen}")
    else:
        ax.set_title(f"Generation {gen} (Initial Population)")

    if gen > 0: plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Snapshot saved: {save_path.name}")


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)

    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"MOEAD_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"MOEA/D Output: {result_dir}")
    try:
        plot_environment_image(env, save_path=str(result_dir / "environment.png"))
    except:
        pass

    print("\nStarting MOEA/D...")
    moead_params = config.get_moead_params()
    moead = MOEAD(env, **moead_params)

    n_gens = moead_params['n_generations']
    checkpoints = {
        0,
        int(n_gens * 0.25),
        int(n_gens * 0.5),
        int(n_gens * 0.75),
        n_gens
    }

    # --- SNAPSHOT CALLBACK ---
    # Callback used to generate snapshots at checkpoints
    def snapshot_callback(algo, current_gen):
        if current_gen in checkpoints:
            snapshot_data = []

            # GEN 0: Take the FULL population (including invalid individuals)
            if current_gen == 0:
                for ylist in algo.population:
                    path_points = [[x, y] for x, y in zip(algo.xs, ylist)]
                    snapshot_data.append({
                        'exposure': 0,
                        'length': 0,
                        'path': path_points
                    })
            # Other gens: only take the Pareto front
            else:
                current_pareto = algo.pareto_front()
                for i, (path, (f1, f2)) in enumerate(current_pareto):
                    snapshot_data.append({
                        'exposure': -f1,
                        'length': f2,
                        'path': [[p.x, p.y] for p in path.points]
                    })

            if snapshot_data:
                plot_snapshot(env, snapshot_data, current_gen, snapshot_dir / f"gen_{current_gen:03d}.png")

    start_time = time.perf_counter()

    # --- GEN 0 SNAPSHOT ---
    print("Taking snapshot of Generation 0 (Full Population)...")
    moead.initialize_population()
    snapshot_callback(moead, 0)

    # Run the main optimization
    moead.run(verbose=True, callback=snapshot_callback)

    run_time = time.perf_counter() - start_time
    pareto = moead.pareto_front()
    print(f"Finished in {run_time:.2f}s. Solutions found: {len(pareto)}")

    save_json(moead.hypervolume_history, result_dir / "hv_history.json")
    try:
        plot_hypervolume_history(moead.hypervolume_history, save_path=str(result_dir / "hv_history.png"))
    except Exception as e:
        print(f"Error plotting HV: {e}")

    if moead.pareto_size_history:
        try:
            plot_pareto_size_history(moead.pareto_size_history, save_path=str(result_dir / "pareto_size.png"))
            plot_pareto_fronts_by_generation(moead.pareto_front_history,
                                             save_path=str(result_dir / "pareto_fronts.png"))
        except:
            pass

    solutions_data = []
    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'MOEAD',
            'id': i,
            'exposure': -f1,
            'length': f2,
            'path': [[p.x, p.y] for p in path.points]
        })

    save_json(solutions_data, result_dir / "pareto_solutions.json")

    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")
        with open(result_dir / "best_solution.txt", "w") as f:
            f.write(
                f"Best Balance Solution:\nID: {best_sol['id']}\nExp: {best_sol['exposure']}\nLen: {best_sol['length']}")
        print(f"Best Balance Solution Saved (Exp: {best_sol['exposure']:.2f})")
    else:
        print("No valid solutions to plot.")


if __name__ == "__main__":
    main()