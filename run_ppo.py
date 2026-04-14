import sys
import time
from pathlib import Path
from datetime import datetime
from utils.config_loader import load_config
from utils.draw import plot_environment_image
from algorithm.ppo import PPO
import numpy as np

# Sử dụng lại các hàm helper từ run.py của bạn
from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run_ppo.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)

    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"PPO_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"PPO Output: {result_dir}")

    # Khởi tạo PPO với tham số từ config
    ppo_params = config.config['algorithm']['ppo']
    agent = PPO(env, **ppo_params)

    def snapshot_callback(algo, gen):
        # Chụp ảnh mỗi 25% quá trình
        checkpoints = [1, agent.max_episodes // 4, agent.max_episodes // 2, agent.max_episodes]
        if gen in checkpoints:
            pareto = algo.pareto_front()
            snapshot_data = []
            for path, (f1, f2) in pareto:
                snapshot_data.append({
                    'exposure': -f1,
                    'length': f2,
                    'path': [[p.x, p.y] for p in path]
                })
            if snapshot_data:
                plot_snapshot(env, snapshot_data, gen, snapshot_dir / f"gen_{gen:03d}.png")

    start_time = time.perf_counter()
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # Lưu kết quả cuối cùng
    pareto = agent.pareto_front()
    solutions_data = []
    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'PPO',
            'id': i,
            'exposure': -f1,
            'length': f2,
            'path': [[p.x, p.y] for p in path]
        })

    save_json(solutions_data, result_dir / "pareto_solutions.json")
    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")

    print(f"PPO finished in {run_time:.2f}s. Found {len(solutions_data)} solutions.")


if __name__ == "__main__":
    main()