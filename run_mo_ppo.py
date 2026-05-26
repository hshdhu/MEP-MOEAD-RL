import sys
import time
from pathlib import Path
from datetime import datetime
from utils.config_loader import load_config

# Import class PPO từ file mo_ppo.py (đặt alias là MO_PPO để tránh nhầm lẫn)
# Giả sử bạn lưu file mo_ppo ở thư mục algorithm/mo_ppo.py
from algorithm.mo_ppo import PPO as MO_PPO

import numpy as np

# Sử dụng lại các hàm helper từ run.py
from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run_mo_ppo.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)

    # 1. Tạo cấu trúc thư mục lưu kết quả cho MO-PPO
    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"MO_PPO_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"MO-PPO Output directory: {result_dir}")

    # 2. Khởi tạo MO-PPO Agent
    # Cố gắng lấy param của 'mo_ppo', nếu không có thì fallback về 'ppo' cũ
    algo_config = config.config.get('algorithm', {})
    mo_ppo_params = algo_config.get('mo_ppo', algo_config.get('ppo', {}))

    agent = MO_PPO(env, **mo_ppo_params)

    # 3. Callback để chụp lại quá trình hội tụ của tập Pareto
    def snapshot_callback(algo, gen):
        # Chụp ảnh ở 4 mốc: Bắt đầu, 25%, 50% và Kết thúc
        checkpoints = [1, agent.max_episodes // 4, agent.max_episodes // 2, agent.max_episodes]
        if gen in checkpoints:
            pareto = algo.pareto_front()
            snapshot_data = []
            for path, (f1, f2) in pareto:
                snapshot_data.append({
                    'exposure': -f1,  # f1 lưu là -exposure nên phải đảo dấu lại
                    'length': f2,
                    'path': [[p.x, p.y] for p in path]
                })
            if snapshot_data:
                plot_snapshot(env, snapshot_data, gen, snapshot_dir / f"gen_{gen:03d}.png")

    # 4. Thực thi huấn luyện
    print("=" * 60)
    print(f"🚀 STARTING MO-PPO TRAINING (Max Gens: {agent.max_episodes})")
    print("=" * 60)

    start_time = time.perf_counter()
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # 5. Xử lý và lưu kết quả tập Pareto cuối cùng
    pareto = agent.pareto_front()
    solutions_data = []

    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'MO-PPO',  # Đổi tên thuật toán ở đây
            'id': i,
            'exposure': -f1,
            'length': f2,
            'path': [[p.x, p.y] for p in path]
        })

    # Lưu file JSON
    save_json(solutions_data, result_dir / "pareto_solutions.json")

    # Tìm solution tốt nhất (Knee Point) và vẽ biểu đồ đường đi
    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")

    print("=" * 60)
    print(f"✅ MO-PPO finished in {run_time:.2f}s.")
    print(f"🏆 Found {len(solutions_data)} non-dominated solutions on the Pareto Front.")
    print(f"📂 Results saved to: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()