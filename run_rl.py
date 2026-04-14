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
from algorithm.sac import SAC


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
    if not solutions:
        return None
    exposures = np.array([s['exposure'] for s in solutions])
    lengths   = np.array([s['length']   for s in solutions])

    norm_len = (lengths - lengths.min()) / (
        lengths.max() - lengths.min()
    ) if lengths.max() != lengths.min() else np.zeros_like(lengths)

    norm_exp = (exposures.max() - exposures) / (
        exposures.max() - exposures.min()
    ) if exposures.max() != exposures.min() else np.zeros_like(exposures)

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
        ax.add_patch(plt.Circle(
            (s.position.x, s.position.y), s.radius,
            facecolor=(0.4, 0.7, 1.0, 0.2), edgecolor='none', zorder=2
        ))

    for sol in solutions:
        path = np.array(sol['path'])
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1],
                    color='green', linewidth=0.8, alpha=0.2, zorder=3)

    if best_solution:
        path = np.array(best_solution['path'])
        ax.plot(path[:, 0], path[:, 1],
                color='#D32F2F', linewidth=3.0, label='Best Balance', zorder=10)
        ax.set_title(
            f"SAC Result\n"
            f"Exp={best_solution['exposure']:.2f}, Len={best_solution['length']:.2f}"
        )

    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_snapshot(env, solutions, gen, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, zorder=5)

    for s in env.sensors:
        ax.add_patch(plt.Circle(
            (s.position.x, s.position.y), s.radius,
            facecolor=(0.4, 0.7, 1.0, 0.2), edgecolor='none', zorder=2
        ))

    for sol in solutions:
        path = np.array(sol['path'])
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1],
                    color='green', linewidth=0.5, alpha=0.3, zorder=3)

    if gen > 0:
        best_sol = find_knee_solution(solutions)
        if best_sol:
            path = np.array(best_sol['path'])
            ax.plot(path[:, 0], path[:, 1],
                    color='#D32F2F', linewidth=2.0, label='Best Current', zorder=10)
            ax.set_title(
                f"Generation {gen}\n"
                f"Best: Exp={best_sol['exposure']:.1f}, Len={best_sol['length']:.1f}"
            )
        else:
            ax.set_title(f"Generation {gen}")
    else:
        ax.set_title(f"Generation {gen} (Warm-up Population)")

    if gen > 0:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Snapshot saved: {save_path.name}")


def plot_training_curves(algo, save_dir: Path):
    """
    Extra diagnostic plots specific to SAC:
    alpha (entropy coeff) and replay buffer size over generations.
    Skipped gracefully if attributes are absent.
    """
    # Hypervolume — reuse shared utility
    try:
        plot_hypervolume_history(
            algo.hypervolume_history,
            save_path=str(save_dir / "hv_history.png")
        )
    except Exception as e:
        print(f"HV plot error: {e}")

    # Pareto size
    try:
        plot_pareto_size_history(
            algo.pareto_size_history,
            save_path=str(save_dir / "pareto_size.png")
        )
    except Exception as e:
        print(f"Pareto size plot error: {e}")

    # Pareto fronts by generation
    try:
        plot_pareto_fronts_by_generation(
            algo.pareto_front_history,
            save_path=str(save_dir / "pareto_fronts.png")
        )
    except Exception as e:
        print(f"Pareto fronts plot error: {e}")

    # SAC-specific: alpha history
    alpha_hist = getattr(algo, 'alpha_history', None)
    if alpha_hist:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(alpha_hist, color='purple')
            ax.set_xlabel("Generation")
            ax.set_ylabel("Entropy coefficient α")
            ax.set_title("SAC — Entropy Coefficient over Training")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "alpha_history.png", dpi=120)
            plt.close()
        except Exception as e:
            print(f"Alpha plot error: {e}")


def get_sac_params(config):
    """
    Pull SAC params from config if available, otherwise return sensible defaults.
    Falls back gracefully so the file works even without config changes.
    """
    # Try dedicated sac section first
    if hasattr(config, 'get_sac_params'):
        return config.get_sac_params()

    # Fall back: reuse moead params for shared keys, ignore EA-only ones
    base = {}
    try:
        moead_params = config.get_moead_params()
        shared_keys = {'dx', 'pop_size', 'n_generations', 'step_exposure', 'length_max'}
        base = {k: v for k, v in moead_params.items() if k in shared_keys}
    except Exception:
        pass

    # SAC-specific defaults (override via config if desired)
    sac_defaults = {
        'hidden_size':    128,
        'lr':             3e-4,
        'gamma':          0.99,
        'tau':            0.005,
        'alpha_init':     0.2,
        'buffer_size':    50_000,
        'batch_size':     64,
        'action_scale':   20.0,
        'window_size':    5,
        'updates_per_ep': 10,
        'w1':             0.5,
        'w2':             0.5,
    }
    return {**sac_defaults, **base}


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run_rl.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)

    result_folder = get_result_folder(len(env.sensors))
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir    = Path("result") / f"{result_folder} sensors" / f"SAC_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"SAC Output: {result_dir}")
    try:
        plot_environment_image(env, save_path=str(result_dir / "environment.png"))
    except Exception:
        pass

    print("\nStarting SAC...")
    sac_params = get_sac_params(config)
    sac = SAC(env, **sac_params)

    n_gens = sac_params.get('n_generations', 100)
    checkpoints = {
        0,
        int(n_gens * 0.25),
        int(n_gens * 0.50),
        int(n_gens * 0.75),
        n_gens
    }

    # ── Snapshot callback (same signature as MOEAD runner) ──────────────────
    def snapshot_callback(algo, current_gen):
        if current_gen not in checkpoints:
            return

        snapshot_data = []

        # Gen 0 → show warm-up population (same as MOEAD's gen-0 full population)
        if current_gen == 0:
            for ylist in algo.population:
                path_points = [[x, y] for x, y in zip(algo.xs, ylist)]
                snapshot_data.append({
                    'exposure': 0,
                    'length':   0,
                    'path':     path_points
                })
        else:
            current_pareto = algo.pareto_front()
            for path, (f1, f2) in current_pareto:
                snapshot_data.append({
                    'exposure': -f1,
                    'length':    f2,
                    'path':     [[p.x, p.y] for p in path.points]
                })

        if snapshot_data:
            plot_snapshot(
                env, snapshot_data, current_gen,
                snapshot_dir / f"gen_{current_gen:03d}.png"
            )

    start_time = time.perf_counter()

    # ── Gen-0 snapshot: after warm-up / initialize_population ───────────────
    print("Initialising SAC (warm-up + initial training)...")
    sac.initialize_population()
    snapshot_callback(sac, 0)

    # ── Main optimisation loop ───────────────────────────────────────────────
    sac.run(verbose=True, callback=snapshot_callback)

    run_time = time.perf_counter() - start_time
    pareto   = sac.pareto_front()
    print(f"Finished in {run_time:.2f}s. Pareto solutions found: {len(pareto)}")

    # ── Save histories ───────────────────────────────────────────────────────
    save_json(sac.hypervolume_history, result_dir / "hv_history.json")
    plot_training_curves(sac, result_dir)

    # ── Build & save Pareto solutions ────────────────────────────────────────
    solutions_data = []
    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'SAC',
            'id':        i,
            'exposure':  -f1,
            'length':     f2,
            'path':      [[p.x, p.y] for p in path.points]
        })

    save_json(solutions_data, result_dir / "pareto_solutions.json")

    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol,
                             result_dir / "solutions_map.png")
        with open(result_dir / "best_solution.txt", "w") as f:
            f.write(
                f"Best Balance Solution:\n"
                f"ID:  {best_sol['id']}\n"
                f"Exp: {best_sol['exposure']}\n"
                f"Len: {best_sol['length']}\n"
                f"Runtime: {run_time:.2f}s\n"
            )
        print(f"Best Balance Solution saved "
              f"(Exp: {best_sol['exposure']:.2f}, Len: {best_sol['length']:.2f})")
    else:
        print("No valid solutions found.")

    # ── Save runtime metadata ────────────────────────────────────────────────
    save_json({
        'algorithm':        'SAC',
        'runtime_seconds':  round(run_time, 2),
        'n_generations':    n_gens,
        'pareto_solutions': len(pareto),
        'sac_params':       {k: str(v) for k, v in sac_params.items()},
    }, result_dir / "run_metadata.json")


if __name__ == "__main__":
    main()