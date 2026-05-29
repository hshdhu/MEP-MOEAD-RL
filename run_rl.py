import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from utils.config_loader import load_config

# Import các thuật toán
from algorithm.ppo import PPO
from algorithm.td3 import TD3
from algorithm.sac import SAC

# Hàm helper của bạn
from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def main():
    parser = argparse.ArgumentParser(description="Run Reinforcement Learning algorithms for Path Planning")
    parser.add_argument("env_json", help="Path to the environment JSON file")
    parser.add_argument("--algo", type=str, choices=["ppo", "td3", "sac"], default="ppo",
                        help="Choose the algorithm to run (ppo, td3, or sac). Default is ppo.")

    args = parser.parse_args()

    # Tải môi trường
    config = load_config()
    env = config.get_environment(load_from_file=args.env_json)

    # Chuẩn bị folder lưu kết quả
    algo_name = args.algo.upper()
    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"{algo_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"[{algo_name}] Output Directory: {result_dir}")

    # Khởi tạo Agent dựa trên lựa chọn
    algo_params = config.config['algorithm'].get(args.algo.lower(), {})

    if args.algo == "ppo":
        agent = PPO(env, **algo_params)
    elif args.algo == "td3":
        agent = TD3(env, **algo_params)
    elif args.algo == "sac":
        agent = SAC(env, **algo_params)

    # Callback chụp ảnh quá trình chạy
    def snapshot_callback(algo_instance, gen):
        # Vì PPO gọi n_generations, còn TD3/SAC gọi n_episodes/n_generations, ta lấy max_episodes từ instance
        max_ep = algo_instance.max_episodes
        checkpoints = [1, max_ep // 4, max_ep // 2, max_ep]

        if gen in checkpoints:
            pareto = algo_instance.pareto_front()
            snapshot_data = []
            for path, (f1, f2) in pareto:
                snapshot_data.append({
                    'exposure': -f1,
                    'length': f2,
                    'path': [[p.x, p.y] for p in path]
                })
            if snapshot_data:
                plot_snapshot(env, snapshot_data, gen, snapshot_dir / f"gen_{gen:03d}.png")

    # Bắt đầu chạy
    start_time = time.perf_counter()
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # Lưu kết quả cuối cùng
    pareto = agent.pareto_front()
    solutions_data = []
    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': algo_name,
            'id': i,
            'exposure': -f1,
            'length': f2,
            'path': [[p.x, p.y] for p in path]
        })

    save_json(solutions_data, result_dir / "pareto_solutions.json")

    # Tìm knee solution và vẽ
    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")

    print(f"[{algo_name}] Finished in {run_time:.2f}s. Found {len(solutions_data)} solutions.")


if __name__ == "__main__":
    main()