import argparse
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình Font chữ to, rõ ràng cho Đồ án Tốt nghiệp
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'figure.autolayout': True
})

ALGO_STYLES = {
    "MOEAD": {"color": "gray", "marker": "*", "label": "MOEAD"},
    "MO-PPO": {"color": "blue", "marker": "o", "label": "MO-PPO"},
    "MO-TD3": {"color": "green", "marker": "^", "label": "MO-TD3"},
    "MO-SAC": {"color": "red", "marker": "s", "label": "MO-SAC"}
}


def load_json(filepath):
    if not filepath.exists():
        print(f"⚠️ Không tìm thấy file: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def smooth_data(data_list, window=50):
    """Làm mượt bằng Moving Average dùng cho HV, IGD, Compare"""
    s = pd.Series(data_list, dtype=float)
    s = s.interpolate(method='linear', limit_direction='both').fillna(0)
    smoothed = s.rolling(window=window, min_periods=1).mean()
    return smoothed.to_numpy()


def ema(data, alpha=0.05):
    """Làm mượt bằng EMA (Exponential Moving Average) dùng cho Loss, Reward giống hệt run_benchmark"""
    data_np = np.array(data, dtype=np.float64)
    if len(data_np) == 0: return data_np
    out = np.zeros_like(data_np)
    out[0] = data_np[0] if not math.isnan(data_np[0]) else 0.0
    for i in range(1, len(data_np)):
        if math.isnan(data_np[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * data_np[i] + (1 - alpha) * out[i - 1]
    return out


def plot_comparative(ax, data_dict, metric_name, window=20):
    """Vẽ biểu đồ gộp có dải mờ (SEM) và chia đều Marker cho đúng NFE"""

    # 1. Tìm độ dài mảng lớn nhất (Max NFE) để đồng bộ trục X cho MOEAD và RL
    max_len = 0
    for algo in ALGO_STYLES:
        if algo in data_dict:
            for sk in data_dict[algo].keys():
                if sk.startswith("Seed_") and data_dict[algo][sk]:
                    max_len = max(max_len, len(data_dict[algo][sk]))

    for algo, style in ALGO_STYLES.items():
        if algo not in data_dict: continue
        seed_keys = [k for k in data_dict[algo].keys() if k.startswith("Seed_")]
        if not seed_keys: continue

        smoothed_seeds = []
        for sk in seed_keys:
            raw_data = data_dict[algo][sk]
            if not raw_data: continue
            sm = smooth_data(raw_data, window=window)
            smoothed_seeds.append(sm)

        if not smoothed_seeds: continue

        matrix = np.array(smoothed_seeds)
        mean_val = np.nanmean(matrix, axis=0)
        sem_val = np.nanstd(matrix, axis=0) / np.sqrt(matrix.shape[0])

        # 2. Stretch trục X dựa trên độ dài của từng mảng để chúng kết thúc cùng 1 mốc NFE
        # Cách này giúp MOEAD (30 điểm) trải dài bằng RL (3000 điểm)
        x = np.linspace(1, max_len, len(mean_val))

        # 3. Chia đều khoảng cách hiển thị Marker (khoảng 12 marker trên 1 line)
        num_markers = 12
        mark_step = max(1, len(mean_val) // num_markers)

        # Vẽ dải mờ
        ax.fill_between(x, mean_val - sem_val, mean_val + sem_val, color=style["color"], alpha=0.15, zorder=1)
        # Vẽ line và marker
        ax.plot(x, mean_val, color=style["color"], marker=style["marker"],
                markevery=mark_step, markersize=8, label=style["label"], zorder=5)


def main():
    parser = argparse.ArgumentParser(description="Re-draw plots from saved JSON data")
    parser.add_argument("result_dir", help="Path to the output directory (e.g. result_benchmark/Bench_...)")
    args = parser.parse_args()

    base_dir = Path(args.result_dir)
    data_dir = base_dir / "data"

    if not data_dir.exists():
        print(f"❌ Không tìm thấy thư mục data tại: {data_dir}")
        return

    out_comp_dir = base_dir / "replot_comparative"
    out_indiv_dir = base_dir / "replot_individual"
    out_comp_dir.mkdir(exist_ok=True)
    out_indiv_dir.mkdir(exist_ok=True)

    print(f"🚀 Bắt đầu vẽ lại biểu đồ từ: {data_dir}...")

    # =========================================================================
    # 1. VẼ LẠI BIỂU ĐỒ COMPARATIVE (GỘP)
    # =========================================================================
    comparative_configs = [
        ("data_hv.json", "01_Hypervolume_Compare.png", "Hypervolume Comparison", "HV", 30),
        ("data_igd_plus.json", "02_IGD_Plus_Compare.png", "IGD+ Comparison", "IGD+", 30),
        ("data_pareto_size.json", "03_Pareto_Size_Compare.png", "Pareto Solutions Count", "Count", 50),
        ("data_success_rate.json", "04_SuccessRate_Compare.png", "Training Success Rate", "Success Rate (%)", 50)
    ]

    for json_file, out_name, title, ylabel, window in comparative_configs:
        data = load_json(data_dir / json_file)
        if data:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            plot_comparative(ax, data, title, window=window)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel("Number of Path Evaluations (NFE)")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc="best", framealpha=0.9)
            plt.savefig(out_comp_dir / out_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Đã vẽ Compare: {out_name}")

    # =========================================================================
    # 2. VẼ LẠI BIỂU ĐỒ INDIVIDUAL (TÁCH RIÊNG TỪNG ALGORITHM)
    # =========================================================================
    d_hv = load_json(data_dir / "data_hv.json")
    d_igd = load_json(data_dir / "data_igd_plus.json")
    d_al = load_json(data_dir / "data_actor_loss.json")
    d_cl = load_json(data_dir / "data_critic_loss.json")
    d_v = load_json(data_dir / "data_value.json")
    d_re = load_json(data_dir / "data_reward_exp.json")
    d_rl = load_json(data_dir / "data_reward_len.json")
    d_rf = load_json(data_dir / "data_reward_feas.json")

    algorithms = ALGO_STYLES.keys()

    for algo in algorithms:
        algo_out_dir = out_indiv_dir / algo
        algo_out_dir.mkdir(exist_ok=True)

        # --- 2A. Vẽ HV & IGD (Làm mượt Window = 20, THÊM LẠI ĐƯỜNG MEAN MÀU ĐEN) ---
        if d_hv and d_igd and algo in d_hv:
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            seed_keys = [k for k in d_hv[algo].keys() if k.startswith("Seed_")]

            all_sm_hv, all_sm_igd = [], []
            for sk in seed_keys:
                if d_hv[algo][sk]:
                    sm_hv = smooth_data(d_hv[algo][sk], window=20)
                    axs[0].plot(sm_hv, alpha=0.4, label=sk)
                    all_sm_hv.append(sm_hv)
                if d_igd[algo][sk]:
                    sm_igd = smooth_data(d_igd[algo][sk], window=20)
                    axs[1].plot(sm_igd, alpha=0.4, label=sk)
                    all_sm_igd.append(sm_igd)

            # VẼ ĐƯỜNG MEAN MÀU ĐEN CHO HV/IGD
            if all_sm_hv: axs[0].plot(np.nanmean(all_sm_hv, axis=0), color='black', linewidth=2.5, label='Mean')
            if all_sm_igd: axs[1].plot(np.nanmean(all_sm_igd, axis=0), color='black', linewidth=2.5, label='Mean')

            axs[0].set_title(f"{algo} - Hypervolume", fontweight='bold')
            axs[1].set_title(f"{algo} - IGD+", fontweight='bold')
            for ax in axs:
                ax.set_xlabel("NFE")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

            plt.savefig(algo_out_dir / f"{algo}_Metrics_HV_IGD.png", dpi=300, bbox_inches='tight')
            plt.close()

        # --- 2B. Vẽ Loss và Reward (SỬ DỤNG EMA NHƯ RUN_BENCHMARK, KHÔNG LÀM DẠI ẢNH) ---
        if algo != "MOEAD":
            # TÁCH 1: FIGURE LOSSES (1 hàng x 3 cột)
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))
            metrics_loss = [(d_al, 'Actor Loss', 0), (d_cl, 'Critic Loss', 1), (d_v, 'Value (Q/V)', 2)]

            for d_dict, title, col in metrics_loss:
                if not d_dict or algo not in d_dict: continue
                all_ema = []
                for sk in seed_keys:
                    raw = d_dict[algo][sk]
                    if raw and not np.all(np.isnan(raw)):
                        sm = ema(raw, alpha=0.05)  # Sử dụng EMA tự nhiên
                        axs[col].plot(sm, alpha=0.3, label=sk)
                        all_ema.append(sm)

                if all_ema:
                    mean_ema = np.nanmean(all_ema, axis=0)
                    axs[col].plot(mean_ema, color='black', linewidth=2.5, label="Mean EMA")

                axs[col].set_title(f"{algo} - {title}", fontweight='bold')
                axs[col].set_xlabel("NFE")
                axs[col].grid(True, linestyle='--', alpha=0.7)
                axs[col].legend()

            plt.savefig(algo_out_dir / f"{algo}_Losses.png", dpi=300, bbox_inches='tight')
            plt.close()

            # TÁCH 2: FIGURE REWARDS (1 hàng x 3 cột)
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))
            metrics_reward = [(d_re, 'Reward: Exposure', 0), (d_rl, 'Reward: Length', 1),
                              (d_rf, 'Reward: Feasibility', 2)]

            for d_dict, title, col in metrics_reward:
                if not d_dict or algo not in d_dict: continue
                all_ema = []
                for sk in seed_keys:
                    raw = d_dict[algo][sk]
                    if raw and not np.all(np.isnan(raw)):
                        sm = ema(raw, alpha=0.05)  # Sử dụng EMA tự nhiên
                        axs[col].plot(sm, alpha=0.3, label=sk)
                        all_ema.append(sm)

                if all_ema:
                    mean_ema = np.nanmean(all_ema, axis=0)
                    axs[col].plot(mean_ema, color='black', linewidth=2.5, label="Mean EMA")

                axs[col].set_title(f"{algo} - {title}", fontweight='bold')
                axs[col].set_xlabel("NFE")
                axs[col].grid(True, linestyle='--', alpha=0.7)
                axs[col].legend()

            plt.savefig(algo_out_dir / f"{algo}_Rewards.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Đã vẽ các biểu đồ chi tiết cho: {algo}")

    print("\n🎉 HOÀN TẤT VẼ LẠI BIỂU ĐỒ!")
    print(f"📁 Hình ảnh mới đã được lưu tại:")
    print(f"   - {out_comp_dir}")
    print(f"   - {out_indiv_dir}")


if __name__ == "__main__":
    main()