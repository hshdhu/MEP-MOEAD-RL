import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import Utilities
from utils.config_loader import load_config
from utils.draw import plot_environment_image

# Import Algorithms
from algorithm.moead_single import MOEAD_Single
from algorithm.standard_ga import StandardGA


def get_result_folder(num_sensors):
    """Categorizes the result directory based on sensor density."""
    if num_sensors <= 50:
        return "50"
    elif num_sensors <= 100:
        return "100"
    elif num_sensors <= 150:
        return "150"
    else:
        return "200"


def save_json(data, filepath):
    """Serializes execution data into a JSON format for post-processing."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"JSON serialization error: {e}")


def plot_benchmark_convergence(moead_hist, ga_hist, save_path):
    """
    Generates a convergence plot (Fitness vs. Generation).
    Maps negative exposure back to positive values for intuitive visualization
    where 'Higher is Better'.
    """
    plt.figure(figsize=(10, 6))

    # Normalize fitness values for plotting (absolute exposure)
    moead_data = [-v if v < 0 else v for v in moead_hist]
    ga_data = [-v if v < 0 else v for v in ga_hist]

    generations = range(1, len(moead_data) + 1)

    plt.plot(generations, moead_data, label='MOEA/D-Single', color='blue', linewidth=2)
    plt.plot(generations, ga_data, label='Standard GA', color='red', linewidth=2, linestyle='--')

    plt.xlabel('Generation')
    plt.ylabel('Exposure Value (Maximization Objective)')
    plt.title('Convergence Profile: MOEA/D vs. GA')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_benchmark_paths(env, moead_path, ga_path, moead_score, ga_score, save_path):
    """
    Overlay visualization of the optimal paths found by both algorithms.
    Includes sensor coverage areas and obstacle geometry.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    # Render Environment Obstacles
    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, zorder=5)

    # Render Sensor Sensing Fields
    for s in env.sensors:
        ax.add_patch(
            plt.Circle((s.position.x, s.position.y), s.radius,
                       facecolor=(0.4, 0.7, 1.0, 0.2), edgecolor='none', zorder=2))

    # Plot Best Path: MOEA/D (Solid Blue)
    if moead_path:
        path_arr = np.array([[p.x, p.y] for p in moead_path.points])
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='blue', linewidth=2.5,
                label=f'MOEA/D (Exp={moead_score:.2f})', zorder=10)

    # Plot Best Path: GA (Dashed Red)
    if ga_path:
        path_arr = np.array([[p.x, p.y] for p in ga_path.points])
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='red', linewidth=2.0, linestyle='--',
                label=f'GA (Exp={ga_score:.2f})', zorder=11)

    ax.set_title("Path Planning Benchmark: Comparative Trajectories")
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # 1. Configuration Loading
    config_loader = load_config()

    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]

    # 2. Environment Initialization
    print(f"Loading environment from: {env_file}")
    env = config_loader.get_environment(load_from_file=env_file)

    # 3. Parameter Aggregation
    # Combines path discretization, exposure step, and evolutionary parameters
    params = config_loader.get_moead_params()

    # 4. Output Directory Management
    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("result_benchmark") / f"{result_folder} sensors" / f"Benchmark_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- BENCHMARK STARTING ---")
    print(f"Output Directory: {base_dir}")
    print("\n[Configuration Loaded]")
    print(f" - Discretization (dx): {params['dx']}, Constraint (Length Max): {params['length_max']}")
    print(f" - Numerical Integration Step (Exposure): {params['step_exposure']}")
    print(f" - Hyperparameters: Pop={params['pop_size']}, Gen={params['n_generations']}, "
          f"Mut={params['mutation_prob']}")

    # Visualize initial scenario
    try:
        plot_environment_image(env, save_path=str(base_dir / "environment.png"))
    except Exception as e:
        print(f"Warning: Environmental visualization failed. {e}")

    # --- 5. Algorithm I: MOEA/D-Single Implementation ---
    print("\n>>> 1. Executing MOEA/D-Single Optimization")
    start_moead = time.perf_counter()

    solver_moead = MOEAD_Single(
        env=env,
        dx=params['dx'],
        pop_size=params['pop_size'],
        n_generations=params['n_generations'],
        neighborhood_size=params['neighborhood_size'],
        crossover_prob=params['crossover_prob'],
        mutation_prob=params['mutation_prob'],
        eta_c=params['eta_c'],
        eta_m=params['eta_m'],
        step_exposure=params['step_exposure'],
        length_max=params['length_max']
    )

    solver_moead.run(verbose=True)

    time_moead = time.perf_counter() - start_moead
    # Metrics: Objective maximization (positive exposure)
    score_moead = -solver_moead.best_solution[1] if solver_moead.best_solution else float('inf')
    print(f"[MOEA/D] Completed. Wall Time: {time_moead:.2f}s | Optimal Exposure: {score_moead:.2f}")

    # --- 6. Algorithm II: Standard GA (Baseline) ---
    print("\n>>> 2. Executing Standard GA Optimization")
    start_ga = time.perf_counter()

    # Benchmark under identical path discretization and constraints
    solver_ga = StandardGA(
        env=env,
        dx=params['dx'],
        pop_size=params['pop_size'],
        n_generations=params['n_generations'],
        crossover_prob=params['crossover_prob'],
        mutation_prob=params['mutation_prob'],
        eta_c=params['eta_c'],
        eta_m=params['eta_m'],
        step_exposure=params['step_exposure'],
        length_max=params['length_max'],
        tournament_size=3,
        n_elites=2
    )

    solver_ga.run(verbose=True)

    time_ga = time.perf_counter() - start_ga
    score_ga = -solver_ga.best_solution[1] if solver_ga.best_solution else float('inf')
    print(f"[GA] Completed. Wall Time: {time_ga:.2f}s | Optimal Exposure: {score_ga:.2f}")

    # --- 7. Statistical Data Logging & Visualization ---

    # Export benchmark metrics to JSON for statistical analysis
    benchmark_data = {
        "config_metadata": params,
        "metrics": {
            "moead": {
                "computation_time": time_moead,
                "best_fitness": score_moead,
                "convergence_history": solver_moead.best_history
            },
            "ga": {
                "computation_time": time_ga,
                "best_fitness": score_ga,
                "convergence_history": solver_ga.best_history
            }
        }
    }
    save_json(benchmark_data, base_dir / "benchmark_data.json")

    # Generate Comparative Graphics
    try:
        plot_benchmark_convergence(
            solver_moead.best_history,
            solver_ga.best_history,
            base_dir / "convergence_comparison.png"
        )
        plot_benchmark_paths(
            env,
            solver_moead.get_best_path(),
            solver_ga.get_best_path(),
            score_moead,
            score_ga,
            base_dir / "paths_comparison.png"
        )
    except Exception as e:
        print(f"Error during graphics generation: {e}")

    # Summary Report Generation for Documentation
    report_file = base_dir / "summary_report.txt"
    with open(report_file, "w") as f:
        f.write("ALGORITHM PERFORMANCE BENCHMARK REPORT\n")
        f.write("======================================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Environment: {env_file}\n")
        f.write(f"Sensor Density: {len(env.sensors)}\n")
        f.write("-" * 40 + "\n")
        f.write("EXPERIMENTAL PARAMETERS:\n")
        f.write(f"  Path Discretization (dx): {params['dx']}\n")
        f.write(f"  Maximum Length Constraint: {params['length_max']}\n")
        f.write(f"  Population Size: {params['pop_size']}\n")
        f.write(f"  Generations: {params['n_generations']}\n")
        f.write("-" * 40 + "\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"1. MOEA/D-Single:\n")
        f.write(f"   - Exposure (Max): {score_moead:.4f}\n")
        f.write(f"   - CPU Time: {time_moead:.4f}s\n\n")
        f.write(f"2. Standard GA:\n")
        f.write(f"   - Exposure (Max): {score_ga:.4f}\n")
        f.write(f"   - CPU Time: {time_ga:.4f}s\n\n")

        diff = score_moead - score_ga
        if diff > 0:
            f.write(f"FINDING: MOEA/D outperformed GA by {diff:.4f} exposure units.\n")
        elif diff < 0:
            f.write(f"FINDING: GA outperformed MOEA/D by {-diff:.4f} exposure units.\n")
        else:
            f.write("FINDING: Both algorithms converged to a statistically identical solution.\n")

    print(f"\n[Success] Benchmark results archived in: {base_dir}")


if __name__ == "__main__":
    main()