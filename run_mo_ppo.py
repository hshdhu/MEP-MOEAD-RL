import sys
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
from algorithm.mo_ppo import PPO as MO_PPO
from run import get_result_folder, save_json, find_knee_solution, plot_snapshot, plot_final_solutions


def plot_metrics(generations, hv_values, igd_values, save_dir):
    """Hàm phụ trợ để vẽ và lưu biểu đồ HV và IGD+"""
    # 1. Vẽ Hypervolume (HV)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, hv_values, label='Hypervolume', color='blue', linewidth=2)
    plt.xlabel('Generations')
    plt.ylabel('Hypervolume (Normalized)')
    plt.title('Hypervolume Convergence over Generations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "hypervolume_curve.png", dpi=300)
    plt.close()

    # 2. Vẽ IGD+
    plt.figure(figsize=(10, 6))
    plt.plot(generations, igd_values, label='IGD+', color='red', linewidth=2)
    plt.xlabel('Generations')
    plt.ylabel('IGD+ (Normalized)')
    plt.title('IGD+ Convergence over Generations (Lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "igd_plus_curve.png", dpi=300)
    plt.close()


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

    algo_config = config.config.get('algorithm', {})
    mo_ppo_params = algo_config.get('mo_ppo', algo_config.get('ppo', {}))

    agent = MO_PPO(env, **mo_ppo_params)

    # --- BIẾN TOÀN CỤC ĐỂ LƯU LỊCH SỬ PARETO FRONT ---
    history_fronts = []  # Lưu giá trị (f1, f2) của mỗi N thế hệ
    history_gens = []  # Lưu số thứ tự thế hệ tương ứng
    log_interval = 10  # Cứ 10 thế hệ sẽ lưu lại tập Pareto hiện tại 1 lần

    # 3. Callback để chụp lại quá trình hội tụ
    def snapshot_callback(algo, gen):
        # A. Phần chụp ảnh bản đồ ban đầu của bạn
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

        # B. PHẦN BỔ SUNG: Lưu trữ Fronts để tính HV và IGD+
        if gen % log_interval == 0 or gen == agent.max_episodes:
            pareto = algo.pareto_front()
            if len(pareto) > 0:
                # Trích xuất Objectives: [f1, f2] (cả 2 đều cần Minimize)
                # f1 = -exposure (minimize -exp <=> maximize exp), f2 = length
                front = np.array([[f1, f2] for _, (f1, f2) in pareto])
                history_fronts.append(front)
                history_gens.append(gen)

    # 4. Thực thi huấn luyện
    print("=" * 60)
    print(f"🚀 STARTING MO-PPO TRAINING (Max Gens: {agent.max_episodes})")
    print("=" * 60)

    start_time = time.perf_counter()
    agent.run(verbose=True, callback=snapshot_callback)
    run_time = time.perf_counter() - start_time

    # =====================================================================
    # 5. TÍNH TOÁN VÀ VẼ BIỂU ĐỒ HYPERVOLUME VÀ IGD+ SAU KHI TRAINING XONG
    # =====================================================================
    if len(history_fronts) > 0:
        print("📊 Calculating HV and IGD+ metrics...")

        # Gộp tất cả các điểm từng tìm được trong toàn bộ quá trình lại
        all_points = np.vstack(history_fronts)

        # Tìm Max / Min toàn cục để chuẩn hóa (Normalize) về dải [0, 1]
        # Điều này RẤT QUAN TRỌNG trong Multi-Objective vì Exposure và Length khác hẳn hệ quy chiếu
        global_min = np.min(all_points, axis=0)
        global_max = np.max(all_points, axis=0)
        range_val = global_max - global_min
        range_val[range_val == 0] = 1e-9  # Tránh chia cho 0

        # Lấy Pareto Front tốt nhất của toàn bộ quá trình làm Reference Front cho IGD+
        nds = NonDominatedSorting()
        best_indices = nds.do(all_points, only_non_dominated_front=True)
        true_pareto_front = all_points[best_indices]

        # Chuẩn hóa Reference Front
        normalized_true_pf = (true_pareto_front - global_min) / range_val

        # Khởi tạo Indicators từ pymoo
        # Điểm tham chiếu cho HV thường đặt ở [1.1, 1.1] trên không gian đã chuẩn hóa
        ind_hv = HV(ref_point=np.array([1.1, 1.1]))
        ind_igd_plus = IGDPlus(normalized_true_pf)

        hv_values = []
        igd_values = []

        # Tính HV và IGD+ cho từng thế hệ được lưu
        for front in history_fronts:
            normalized_front = (front - global_min) / range_val
            hv_values.append(ind_hv(normalized_front))
            igd_values.append(ind_igd_plus(normalized_front))

        # Vẽ và lưu biểu đồ
        plot_metrics(history_gens, hv_values, igd_values, result_dir)
        print("✅ HV and IGD+ plots saved successfully.")
    else:
        print("⚠️ No valid paths found during training. Cannot calculate metrics.")
    # =====================================================================

    # 6. Xử lý và lưu kết quả tập Pareto cuối cùng
    pareto = agent.pareto_front()
    solutions_data = []

    for i, (path, (f1, f2)) in enumerate(pareto):
        solutions_data.append({
            'algorithm': 'MO-PPO',
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