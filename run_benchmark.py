import argparse
import time
import random
import json
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import thư viện tính toán Đa mục tiêu
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Import môi trường và thuật toán
from utils.config_loader import load_config
from algorithm.moead import MOEAD
from algorithm.mo_ppo import MO_PPO
from algorithm.mo_td3 import MO_TD3
from algorithm.mo_sac import MO_SAC
from run import plot_final_solutions  # Để vẽ bản đồ đường đi


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def plot_initial_environment(env, save_path):
    # Thiết lập kích thước hệ trục đồng nhất (ví dụ 10x10 hoặc 8x8 tùy bạn chọn)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    # 1. Vẽ các vùng cảm biến (Lớp dưới cùng: zorder=1, 2)
    for s in env.sensors:
        ax.add_patch(
            plt.Circle(
                (s.position.x, s.position.y),
                s.radius,
                facecolor=(0.4, 0.7, 1.0, 0.25),  # Màu xanh lam nhạt đồng nhất
                edgecolor=(0.4, 0.7, 1.0, 0.4),
                linewidth=0.7,
                zorder=1
            )
        )
        # Vẽ tâm cảm biến dạng chấm tròn nhỏ thay vì chữ X đỏ
        ax.plot(s.position.x, s.position.y, 'o',
                color=(0.4, 0.7, 1.0, 0.7),
                markersize=3,
                zorder=2)

    # 2. Vẽ các vật cản (Lớp trên: zorder=10)
    for obs in env.obstacles:
        # Nếu obs.polygon tồn tại (Shapely object), lấy tọa độ exterior
        if hasattr(obs, 'polygon'):
            x, y = obs.polygon.exterior.xy
        # Ngược lại, nếu dùng hàm to_tuples() như hàm plot_environment gốc của bạn
        elif hasattr(obs, 'to_tuples'):
            x, y = zip(*obs.to_tuples())
        else:
            x, y = obs.exterior.xy

        ax.fill(x, y, color='gray', alpha=0.5, label='Obstacle', zorder=10)

    plt.title("Initial Environment Configuration", fontsize=14, fontweight='bold')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Khử trùng lặp nhãn Obstacle trong Legend nếu cần
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mean_std(ax, x, data, label, color):
    data_np = np.array(data, dtype=np.float64)
    mean = np.nanmean(data_np, axis=0)
    std = np.nanstd(data_np, axis=0)
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def clean_nan(val):
    if isinstance(val, (float, np.floating)) and math.isnan(val): return None
    return val


def main():
    parser = argparse.ArgumentParser(description="Benchmark 4 MO-RL Algorithms")
    parser.add_argument("env_json", help="Path to the environment JSON file")
    args = parser.parse_args()

    config = load_config()
    env_base = config.get_environment(load_from_file=args.env_json)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result_benchmark") / f"Bench_{len(env_base.sensors)}sensors_{timestamp}"

    # TẠO 3 THƯ MỤC CỐT LÕI
    data_dir = result_dir / "data"
    comp_plots_dir = result_dir / "comparative_plots"
    indiv_plots_dir = result_dir / "individual_algorithms"

    data_dir.mkdir(parents=True, exist_ok=True)
    comp_plots_dir.mkdir(parents=True, exist_ok=True)
    indiv_plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Bắt đầu Benchmark. Thư mục: {result_dir}")
    plot_initial_environment(env_base, comp_plots_dir / "00_initial_environment.png")

    env_info = {
        "width": env_base.width, "height": env_base.height,
        "n_sensors": len(env_base.sensors), "n_obstacles": len(env_base.obstacles)
    }
    save_json(env_info, data_dir / "environment_info.json")

    algorithms = {"MOEAD": MOEAD, "MO-PPO": MO_PPO, "MO-TD3": MO_TD3, "MO-SAC": MO_SAC}
    seeds = [42, 100, 2024]
    colors = {"MOEAD": "gray", "MO-PPO": "blue", "MO-TD3": "green", "MO-SAC": "red"}

    raw_results = {algo: {seed: {
        'fronts_objs': [], 'final_pareto_shape': [],
        'actor_loss': [], 'critic_loss': [], 'value': [],
        'reward_exp': [], 'reward_len': [], 'reward_feas': [],
        'time': 0.0, 'success': 0
    } for seed in seeds} for algo in algorithms}
    algo_configs = {}

    # =========================================================================
    # 1. QUÁ TRÌNH TRAINING
    # =========================================================================
    for algo_name, AlgoClass in algorithms.items():
        print("=" * 60)
        print(f"🔥 BẮT ĐẦU: {algo_name}")

        # Tạo thư mục riêng cho thuật toán này
        algo_indiv_dir = indiv_plots_dir / algo_name
        algo_indiv_dir.mkdir(exist_ok=True)

        for seed in seeds:
            print(f"👉 Chạy Seed {seed}...")
            set_global_seeds(seed)
            env = config.get_environment(load_from_file=args.env_json)

            algo_cfg_name = algo_name.lower().replace("-", "_")

            # 2. Đọc parameters từ file YAML
            algo_params = config.config.get('algorithm', {}).get(algo_cfg_name, {}).copy()

            # 3. Xử lý logic số vòng lặp (Ưu tiên YAML > Terminal > Mặc định)
            current_max_episodes = algo_params.get('n_generations', 3000)
            pop_size = algo_params.get('pop_size', 1)  # MOEAD là 50, RL là 1

            # Lưu lại để lát vẽ đồ thị
            algo_configs[algo_name] = {
                'steps': current_max_episodes,
                'evals_per_step': pop_size
            }

            # Gán lại vào algo_params để truyền vào __init__ của thuật toán
            algo_params['n_generations'] = current_max_episodes

            # Đối với RL (MO-PPO, MO-TD3, MO-SAC), ta cần thêm key 'n_episodes'
            if algo_name != "MOEAD":
                algo_params['n_episodes'] = current_max_episodes

            # 4. Khởi tạo thuật toán với các tham số đã đọc được
            agent = AlgoClass(env, **algo_params)

            def create_callback(current_seed):
                def cb(algo_instance, gen):
                    pf = algo_instance.pareto_front()

                    obj_front = []
                    for item in pf:
                        objs = item[1]
                        if objs[0] != float('inf'): obj_front.append([objs[0], objs[1]])

                    raw_results[algo_name][current_seed]['fronts_objs'].append(obj_front)

                    raw_results[algo_name][current_seed]['actor_loss'].append(
                        getattr(algo_instance, 'current_actor_loss', float('nan')))
                    raw_results[algo_name][current_seed]['critic_loss'].append(
                        getattr(algo_instance, 'current_critic_loss', float('nan')))
                    raw_results[algo_name][current_seed]['reward_exp'].append(
                        getattr(algo_instance, 'current_reward_exp', float('nan')))
                    raw_results[algo_name][current_seed]['reward_len'].append(
                        getattr(algo_instance, 'current_reward_len', float('nan')))
                    raw_results[algo_name][current_seed]['reward_feas'].append(
                        getattr(algo_instance, 'current_reward_feas', float('nan')))
                    raw_results[algo_name][current_seed]['value'].append(
                        getattr(algo_instance, 'current_value', float('nan')))

                return cb

            start_time = time.perf_counter()
            agent.run(verbose=True, callback=create_callback(seed))
            run_time = time.perf_counter() - start_time

            raw_results[algo_name][seed]['time'] = run_time

            final_pf = agent.pareto_front()

            valid_final_objs = []
            final_pareto_shape = []
            solutions_data = []  # Data để vẽ map

            for idx, item in enumerate(final_pf):
                path, objs = item[0], item[1]
                if objs[0] != float('inf'):
                    valid_final_objs.append([objs[0], objs[1]])
                    final_pareto_shape.append({"exposure": -objs[0], "length": objs[1]})
                    solutions_data.append({
                        'algorithm': algo_name, 'id': idx,
                        'exposure': -objs[0], 'length': objs[1],
                        'path': [[p.x, p.y] for p in path]
                    })

            raw_results[algo_name][seed]['success'] = 1 if len(valid_final_objs) > 0 else 0
            raw_results[algo_name][seed]['final_pareto_shape'] = final_pareto_shape

            # --- VẼ BẢN ĐỒ ĐƯỜNG ĐI CHO SEED NÀY ---
            if len(solutions_data) > 0:
                map_save_path = algo_indiv_dir / f"{algo_name}_Seed_{seed}_Map.png"
                best_sol = solutions_data[0]  # Đại diện 1 nghiệm để highlight
                plot_final_solutions(env, solutions_data, best_sol, map_save_path)

            print(f"   ✅ Xong Seed {seed} trong {run_time:.2f}s | Nghiệm: {len(valid_final_objs)}")

    # =========================================================================
    # 2. TÍNH TOÁN METRICS
    # =========================================================================
    print("\n📊 Đang tính toán Metrics (HV, IGD+)...")
    all_final_points = []
    for algo in algorithms:
        for seed in seeds:
            front = raw_results[algo][seed]['fronts_objs'][-1] if raw_results[algo][seed]['fronts_objs'] else []
            all_final_points.extend(front)

    if not all_final_points:
        print("⚠️ TẤT CẢ thuật toán thất bại. Dừng script.")
        return

    all_final_points = np.array(all_final_points)
    global_min = np.min(all_final_points, axis=0)
    global_max = np.max(all_final_points, axis=0)
    range_val = global_max - global_min
    range_val[range_val < 1e-6] = 1.0

    nds = NonDominatedSorting()
    true_pareto_idx = nds.do(all_final_points, only_non_dominated_front=True)
    normalized_true_pf = (all_final_points[true_pareto_idx] - global_min) / range_val

    ind_hv = HV(ref_point=np.array([1.1, 1.1]))
    ind_igd = IGDPlus(normalized_true_pf)

    metrics = {algo: {'hv': [], 'igd': [], 'size': []} for algo in algorithms}

    for algo in algorithms:
        steps = algo_configs[algo]['steps']
        metrics[algo]['hv'] = np.zeros((len(seeds), steps))
        metrics[algo]['igd'] = np.zeros((len(seeds), steps))
        metrics[algo]['size'] = np.zeros((len(seeds), steps))

        for s_idx, seed in enumerate(seeds):
            seed_fronts = raw_results[algo][seed]['fronts_objs']
            for g_idx in range(steps):
                gen_front = seed_fronts[g_idx] if g_idx < len(seed_fronts) else (seed_fronts[-1] if seed_fronts else [])
                metrics[algo]['size'][s_idx, g_idx] = len(gen_front)

                if len(gen_front) > 0:
                    norm_front = (np.array(gen_front) - global_min) / range_val
                    metrics[algo]['hv'][s_idx, g_idx] = ind_hv(norm_front)
                    metrics[algo]['igd'][s_idx, g_idx] = ind_igd(norm_front)
                else:
                    metrics[algo]['hv'][s_idx, g_idx] = 0.0
                    metrics[algo]['igd'][s_idx, g_idx] = 1.5

    # =========================================================================
    # 3. VẼ BIỂU ĐỒ GỘP SO SÁNH (COMPARATIVE PLOTS)
    # =========================================================================
    print("📈 Đang vẽ Comparative Plots...")
    plots = [
        ("01_Hypervolume_Compare.png", "Hypervolume Comparison", "HV", 'hv'),
        ("02_IGD_Plus_Compare.png", "IGD+ Comparison", "IGD+", 'igd'),
        ("03_Pareto_Size_Compare.png", "Pareto Solutions Count", "Count", 'size')
    ]
    for filename, title, ylabel, metric_key in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algorithms:
            # TRỤC X MỚI: Số thứ tự bước * Số lượng cá thể (NFE)
            steps = algo_configs[algo]['steps']
            evals = algo_configs[algo]['evals_per_step']
            x_axis = np.arange(1, steps + 1) * evals

            plot_mean_std(ax, x_axis, metrics[algo][metric_key], algo, colors[algo])

        ax.set_title(title)
        ax.set_xlabel("Number of Path Evaluations (NFE)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        plt.savefig(comp_plots_dir / filename, dpi=300)
        plt.close()

    # Time & Success Bar Chart
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    algo_names = list(algorithms.keys())
    avg_t = [np.mean([raw_results[a][s]['time'] for s in seeds]) for a in algo_names]
    std_t = [np.std([raw_results[a][s]['time'] for s in seeds]) for a in algo_names]
    suc_r = [np.mean([raw_results[a][s]['success'] for s in seeds]) * 100 for a in algo_names]
    axs[0].bar(algo_names, avg_t, yerr=std_t, capsize=5, color=[colors[a] for a in algo_names], alpha=0.7)
    axs[0].set_title("Execution Time (s)")
    axs[1].bar(algo_names, suc_r, color=[colors[a] for a in algo_names], alpha=0.7)
    axs[1].set_title("Success Rate (%)")
    axs[1].set_ylim(0, 110)
    plt.savefig(comp_plots_dir / "06_Time_SuccessRate.png", dpi=300)
    plt.close()

    # =========================================================================
    # 4. VẼ BIỂU ĐỒ CHI TIẾT TỪNG THUẬT TOÁN (INDIVIDUAL PLOTS)
    # =========================================================================
    print("📈 Đang vẽ Individual Plots...")
    for algo in algorithms:
        algo_dir = indiv_plots_dir / algo

        # TẠO TRỤC X RIÊNG CHO TỪNG THUẬT TOÁN
        steps = algo_configs[algo]['steps']
        evals = algo_configs[algo]['evals_per_step']
        x_axis = np.arange(1, steps + 1) * evals

        # 4.1 HV và IGD+ của từng thuật toán
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        for s_idx, s in enumerate(seeds):
            axs[0].plot(x_axis, metrics[algo]['hv'][s_idx], alpha=0.3, label=f'Seed {s}')
            axs[1].plot(x_axis, metrics[algo]['igd'][s_idx], alpha=0.3, label=f'Seed {s}')

        axs[0].plot(x_axis, np.nanmean(metrics[algo]['hv'], axis=0), 'k-', linewidth=2.5, label='Mean')
        axs[1].plot(x_axis, np.nanmean(metrics[algo]['igd'], axis=0), 'k-', linewidth=2.5, label='Mean')

        axs[0].set_title(f"{algo} - Hypervolume")
        axs[0].set_xlabel("Number of Path Evaluations (NFE)")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].set_title(f"{algo} - IGD+")
        axs[1].set_xlabel("Number of Path Evaluations (NFE)")
        axs[1].grid(True)
        axs[1].legend()
        plt.savefig(algo_dir / f"{algo}_Metrics_HV_IGD.png", dpi=300)
        plt.close()

        # 4.2 Nếu là RL, vẽ thêm Loss và Reward
        if algo != "MOEAD":
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            metrics_rl = [
                ('actor_loss', 0, 0, 'Actor Loss'),
                ('critic_loss', 0, 1, 'Critic Loss'),
                ('value', 0, 2, 'Value (Q/V)'),
                ('reward_exp', 1, 0, 'Reward: Exposure'),
                ('reward_len', 1, 1, 'Reward: Length'),
                ('reward_feas', 1, 2, 'Reward: Feasibility')
            ]

            for key, row, col, title in metrics_rl:
                data_matrix = np.array([raw_results[algo][s][key] for s in seeds])
                if not np.all(np.isnan(data_matrix)):
                    for s_idx, s in enumerate(seeds):
                        axs[row, col].plot(x_axis, data_matrix[s_idx], alpha=0.3, label=f'Seed {s}')
                    axs[row, col].plot(x_axis, np.nanmean(data_matrix, axis=0), 'k-', linewidth=2.5, label='Mean')
                axs[row, col].set_title(f"{algo} - {title}")
                axs[row, col].set_xlabel("Number of Path Evaluations (NFE)")
                axs[row, col].grid(True)
                axs[row, col].legend()

            plt.tight_layout()
            plt.savefig(algo_dir / f"{algo}_Loss_Reward.png", dpi=300)
            plt.close()
    # =========================================================================
    # 5. LƯU DỮ LIỆU JSON
    # =========================================================================
    summary_data, detailed_data = {}, {}
    for algo in algorithms:
        summary_data[algo] = {
            "avg_time": float(np.mean([raw_results[algo][s]['time'] for s in seeds])),
            "success_rate": float(np.mean([raw_results[algo][s]['success'] for s in seeds]) * 100),
            "final_hv_mean": float(np.mean(metrics[algo]['hv'][:, -1])),
            "final_igd_mean": float(np.mean(metrics[algo]['igd'][:, -1]))
        }

        detailed_data[algo] = {}
        for idx, seed in enumerate(seeds):
            detailed_data[algo][f"Seed_{seed}"] = {
                "time": float(raw_results[algo][seed]['time']),
                "success": bool(raw_results[algo][seed]['success']),
                "final_hv": float(metrics[algo]['hv'][idx, -1]),
                "final_igd_plus": float(metrics[algo]['igd'][idx, -1]),
                "final_pareto_size": int(metrics[algo]['size'][idx, -1]),
                "final_pareto_front_shape": raw_results[algo][seed]['final_pareto_shape'],
                "history_actor_loss": [clean_nan(x) for x in raw_results[algo][seed]['actor_loss']],
                "history_critic_loss": [clean_nan(x) for x in raw_results[algo][seed]['critic_loss']],
                "history_reward_exp": [clean_nan(x) for x in raw_results[algo][seed]['reward_exp']],
                "history_reward_len": [clean_nan(x) for x in raw_results[algo][seed]['reward_len']],
                "history_reward_feas": [clean_nan(x) for x in raw_results[algo][seed]['reward_feas']],
                "history_value": [clean_nan(x) for x in raw_results[algo][seed]['value']]
            }

    save_json(summary_data, data_dir / "benchmark_summary.json")
    save_json(detailed_data, data_dir / "detailed_per_seed.json")

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT BENCHMARK TOÀN DIỆN!")
    print(f"📂 Kết quả lưu tại: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()