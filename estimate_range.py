"""
Estimate Range - Robust version (MOEA/D-style sampling)
"""

import numpy as np
import random
import sys
import json

from general.point import Point
from general.path import Path as PathClass
from utils.config_loader import load_config


def ylist_to_path(xs, ys):
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    return PathClass(pts)


# ✅ Sampling "có não" (giống MOEA/D + PPO)
def sample_smooth_path(xs, env):
    base_y = random.uniform(0, env.height)

    # noise nhẹ → path mượt
    noise = np.random.normal(0, env.height * 0.05, size=len(xs))

    ys = np.clip(base_y + noise, 0, env.height)
    return ys


def estimate_ranges(env, dx=5, target_valid=200):
    xs = list(np.arange(0, env.width + 1, dx))

    exp_list = []
    len_list = []

    valid_count = 0
    trials = 0
    max_trials = target_valid * 20  # retry mạnh

    print(f"\nTarget valid samples: {target_valid}")
    print(f"Max trials: {max_trials}")

    while valid_count < target_valid and trials < max_trials:
        trials += 1

        ys = sample_smooth_path(xs, env)
        path = ylist_to_path(xs, ys)

        # reject nếu invalid
        if not env.is_valid_path(path):
            continue

        exp = path.exposure(
            env.sensors,
            step=1.0,
            obstacles=env.obstacles
        )
        length = path.length()

        exp_list.append(exp)
        len_list.append(length)
        valid_count += 1

        if valid_count % 50 == 0:
            print(f"Collected {valid_count} valid samples...")

    if valid_count < 20:
        raise ValueError(
            f"Too few valid samples ({valid_count}). "
            f"Map might be too hard or dx too small."
        )

    exp_min, exp_max = min(exp_list), max(exp_list)
    len_min, len_max = min(len_list), max(len_list)

    return exp_min, exp_max, len_min, len_max, valid_count, trials


def main():
    print("=" * 60)
    print("Robust Range Estimator")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Usage:")
        print("   python estimate_range.py <env_path>")
        return

    env_path = sys.argv[1]

    config = load_config()
    env = config.get_environment(load_from_file=env_path)

    print(f"\nLoaded environment: {env_path}")
    print(f"   Size: {env.width}x{env.height}")
    print(f"   Sensors: {len(env.sensors)}")
    print(f"   Obstacles: {len(env.obstacles)}")

    # 🚀 Estimate
    exp_min, exp_max, len_min, len_max, valid_count, trials = estimate_ranges(env)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Valid samples: {valid_count} / Trials: {trials}")
    print(f"Exposure range: {exp_min:.4f} → {exp_max:.4f}")
    print(f"Length range:   {len_min:.4f} → {len_max:.4f}")

    # 🎯 Suggest normalize
    print("\nSuggested normalization:")
    print(f"exp_norm = (exp - {exp_min:.4f}) / ({exp_max - exp_min:.4f} + 1e-6)")
    print(f"len_norm = (length - {len_min:.4f}) / ({len_max - len_min:.4f} + 1e-6)")

    # 💾 Save file
    range_data = {
        "exp_min": exp_min,
        "exp_max": exp_max,
        "len_min": len_min,
        "len_max": len_max
    }

    range_file = env_path.replace(".json", "_range.json")

    with open(range_file, "w") as f:
        json.dump(range_data, f, indent=4)

    print(f"\nSaved range to: {range_file}")


if __name__ == "__main__":
    main()