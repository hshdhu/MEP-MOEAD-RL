import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import pymoo để tính các metric Đa mục tiêu
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.config_loader import load_config

# Import 3 thuật toán MO-RL
from algorithm.mo_ppo import MO_PPO
from algorithm.mo_td3 import MO_TD3
from algorithm.mo_sac import MO_SAC

from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def plot_metrics(generations, hv_values, igd_values, save_dir):
    """Hàm phụ trợ để vẽ và lưu biểu đồ HV và IGD+"""
    # 1. Vẽ Hypervolume (HV)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, hv_values, label='Hypervolume', color='blue', linewidth=2)
    plt.xlabel('Generations / Episodes')
    plt.ylabel('Hypervolume (Normalized)')
    plt.title('Hypervolume Convergence over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "hypervolume_curve.png", dpi=300)
    plt.close()

    # 2. Vẽ IGD+
    plt.figure(figsize=(10, 6))
    plt.plot(generations, igd_values, label='IGD+', color='red', linewidth=2)
    plt.xlabel('Generations / Episodes')
    plt.ylabel('IGD+ (Normalized)')
    plt.title('IGD+ Convergence over Time (Lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "igd_plus_curve.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Objective RL algorithms for Path Planning")
    parser.add_argument("env_json", help="Path to the environment JSON file")
    parser.add_argument("--algo", type=str, choices=["mo_ppo", "mo_td3", "mo_sac"], default="mo_ppo",
                        help="Choose the MO-RL algorithm to run. Default is mo_ppo.")

    args = parser.parse_args()

    # 1. Tải môi trường và cấu hình
    config = load_config()
    env = config.get_environment(load_from_file=args.env_json)

    # 2. Tạo thư mục lưu kết quả dựa trên tên thuật toán
    algo_name = args.algo.upper()
    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"{algo_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = result_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"🚀 [{algo_name}] Output Directory: {result_dir}")

    # 3. Khởi tạo Agent dựa trên tham số dòng lệnh
    algo_config = config.config.get('algorithm', {})

    base_algo = args.algo.replace("mo_", "")

    algo_params = algo_config.get(args.algo, algo_config.get(base_algo, {}))

    if args.algo == "mo_ppo":
        agent = MO_PPO(env, **algo_params)
    elif args.algo == "mo_td3":
        agent = MO_TD3(env, **algo_params)
    elif args.algo == "mo_sac":
        agent = MO_SAC(env, **algo_params)
    else:
        raise ValueError(f"Thuật toán không được hỗ trợ: {args.algo}")

    # --- BIẾN TOÀN CỤC ĐỂ LƯU LỊCH SỬ PARETO FRONT ---
    history_fronts = []
    history_gens = []
    log_interval = 10

    # 4. Callback chụp ảnh bản đồ & thu thập Fronts
    def snapshot_callback(algo_instance, gen):
        max_ep = getattr(algo_instance, 'max_episodes', 1500)
        checkpoints = [1, max_ep // 4, max_ep // 2, max_ep]

        # A. Snapshot bản đồ
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

        # B. Thu thập Pareto Front để vẽ biểu đồ HV, IGD+
        if gen % log_interval == 0 or gen == max_ep:
            pareto = algo_instance.pareto_front()
            if len(pareto) > 0:
                front = np.array([[f1, f2] for _, (f1, f2) in pareto])
                history_fronts.append(front)
                history_gens.append(gen)

    # 5. Chạy thuật toán
    print("=" * 60)
    print(f"🔥 STARTING {algo_name} TRAINING")
    print("=" * 60)

    start_time = time.perf_counter()
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # =====================================================================
    # 6. TÍNH TOÁN VÀ VẼ BIỂU ĐỒ HYPERVOLUME VÀ IGD+
    # =====================================================================
    if len(history_fronts) > 0:
        print("📊 Calculating HV and IGD+ metrics...")

        all_points = np.vstack(history_fronts)

        # Normalize về [0, 1]
        global_min = np.min(all_points, axis=0)
        global_max = np.max(all_points, axis=0)
        range_val = global_max - global_min
        range_val[range_val == 0] = 1e-9

        # Trích xuất True Pareto Front
        nds = NonDominatedSorting()
        best_indices = nds.do(all_points, only_non_dominated_front=True)
        true_pareto_front = all_points[best_indices]
        normalized_true_pf = (true_pareto_front - global_min) / range_val

        ind_hv = HV(ref_point=np.array([1.1, 1.1]))
        ind_igd_plus = IGDPlus(normalized_true_pf)

        hv_values, igd_values = [], []
        for front in history_fronts:
            normalized_front = (front - global_min) / range_val
            hv_values.append(ind_hv(normalized_front))
            igd_values.append(ind_igd_plus(normalized_front))

        plot_metrics(history_gens, hv_values, igd_values, result_dir)
        print("✅ HV and IGD+ plots saved successfully.")
    else:
        print("⚠️ No valid paths found during training. Cannot calculate metrics.")

    # =====================================================================
    # 7. LƯU KẾT QUẢ JSON VÀ BẢN ĐỒ CUỐI CÙNG
    # =====================================================================
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

    best_sol = find_knee_solution(solutions_data)
    if best_sol:
        plot_final_solutions(env, solutions_data, best_sol, result_dir / "solutions_map.png")

    print("=" * 60)
    print(f"✅ {algo_name} finished in {run_time:.2f}s.")
    print(f"🏆 Found {len(solutions_data)} non-dominated solutions.")
    print(f"📂 Results saved to: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()