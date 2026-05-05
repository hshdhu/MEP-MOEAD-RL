import sys
import time
from pathlib import Path
from datetime import datetime
from utils.config_loader import load_config
from algorithm.td3 import TD3  # Import TD3 thay vì PPO
import numpy as np

# Sử dụng lại các hàm helper từ run.py
from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run_td3.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)

    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Đổi tên thư mục thành TD3
    result_dir = Path("result") / f"{result_folder} sensors" / f"TD3_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"TD3 Output: {result_dir}")

    # Khởi tạo TD3 với tham số từ config
    td3_params = config.config['algorithm']['td3']
    agent = TD3(env, **td3_params)

    def snapshot_callback(algo, gen):
        # Chụp ảnh mỗi 25% quá trình
        checkpoints = [1, algo.max_episodes // 4, algo.max_episodes // 2, algo.max_episodes]
        if gen in checkpoints:
            pareto = algo.pareto_front()
            snapshot_data = []
            for path, (f1, f2) in pareto:
                snapshot_data.append({
                    'exposure': -f1,  # Khôi phục giá trị dương của exposure
                    'length': f2,
                    'path': [[p.x, p.y] for p in path]
                })
            if snapshot_data:
                plot_snapshot(env, snapshot_data, gen, snapshot_dir / f"gen_{gen:03d}.png")

    start_time = time.perf_counter()
    # Chạy thuật toán TD3
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # Lưu kết quả cuối cùng
    pareto = agent.pareto_front()
    solutions_data = []
    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'TD3',  # Gắn nhãn là TD3
            'id': i,
            'exposure': -f1,  # Khôi phục giá trị dương
            'length': f2,
            'path': [[p.x, p.y] for p in path]
        })

    save_json(solutions_data, result_dir / "pareto_solutions.json")

    # Tìm và vẽ best solution (Knee point)
    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")

    print(f"TD3 finished in {run_time:.2f}s. Found {len(solutions_data)} solutions in Pareto front.")


if __name__ == "__main__":
    main()