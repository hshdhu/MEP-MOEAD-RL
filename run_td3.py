import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import từ project của bạn
from utils.config_loader import load_config
from utils.draw import plot_environment_image
from algorithm.td3 import TD3, ReplayBuffer
from general.point import Point
from general.path import Path as RobotPath


def get_result_folder(num_sensors):
    if num_sensors <= 50:
        return "50"
    elif num_sensors <= 100:
        return "100"
    elif num_sensors <= 150:
        return "150"
    else:
        return "200"


def find_knee_solution(solutions):
    if not solutions: return None
    exposures = np.array([s['exposure'] for s in solutions])
    lengths = np.array([s['length'] for s in solutions])

    # Normalize để tìm điểm gần gốc tọa độ nhất (Best Balance)
    norm_len = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-6)
    norm_exp = (exposures - exposures.min()) / (exposures.max() - exposures.min() + 1e-6)
    dist = np.sqrt(norm_len ** 2 + norm_exp ** 2)
    return solutions[np.argmin(dist)]


def save_snapshot(env, path_points, ep, result_dir, info=""):
    """Hàm vẽ snapshot mô phỏng giống MOEA/D"""
    fig, ax = plt.subplots(figsize=(10, 10))
    # Sử dụng hàm vẽ môi trường gốc của bạn nếu nó hỗ trợ truyền ax,
    # nếu không ta vẽ thủ công đơn giản ở đây để đảm bảo snapshot chạy được
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    for obs in env.obstacles:
        x_obs, y_obs = zip(*obs.to_tuples())
        ax.fill(x_obs, y_obs, color='gray', alpha=0.5)

    for s in env.sensors:
        circle = plt.Circle((s.position.x, s.position.y), s.radius, color='blue', alpha=0.1)
        ax.add_patch(circle)

    pts = np.array([[p.x, p.y] for p in path_points])
    ax.plot(pts[:, 0], pts[:, 1], color='red', linewidth=2)
    ax.set_title(f"Episode {ep} {info}")

    save_path = result_dir / "snapshots" / f"gen_{ep:03d}.png"
    plt.savefig(save_path)
    plt.close()


def main():
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python run_td3.py <path_to_env_json>")
        sys.exit(1)

    env_file = sys.argv[1]
    env = config.get_environment(load_from_file=env_file)
    td3_params = config.config['algorithm']['td3']

    dx = config.config['path']['dx']
    xs = np.arange(0, env.width + 1, dx)
    state_dim = 2  # (x, y) - Huy có thể nâng cấp lên Radar sau
    action_dim = 1

    agent = TD3(state_dim, action_dim, env.height, td3_params)
    replay_buffer = ReplayBuffer(int(td3_params['max_replay_buffer']))

    result_folder = get_result_folder(len(env.sensors))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result") / f"{result_folder} sensors" / f"TD3_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "snapshots").mkdir(exist_ok=True)

    # Vẽ môi trường gốc để kiểm tra
    try:
        plot_environment_image(env, save_path=str(result_dir / "environment.png"))
    except:
        pass

    print(f"Starting TD3 Training...")
    reward_history = []
    checkpoints = [1, int(td3_params['n_episodes'] * 0.25), int(td3_params['n_episodes'] * 0.5),
                   int(td3_params['n_episodes'] * 0.75), td3_params['n_episodes']]

    for ep in range(1, td3_params['n_episodes'] + 1):
        current_y = env.height / 2
        state = np.array([xs[0], current_y], dtype=np.float32)
        ep_reward = 0
        current_path = [Point(xs[0], current_y)]

        for i in range(1, len(xs)):
            # Exploration
            action = agent.select_action(state)
            noise = np.random.normal(0, td3_params['exploration_noise'])
            action = np.clip(action + noise, 0, env.height)

            p2 = Point(xs[i], action[0])
            seg = RobotPath([current_path[-1], p2])

            terminated, truncated = False, False
            if not env.is_valid_path(seg):
                reward, terminated = -200.0, True  # Giảm penalty chút để agent dám thử
            else:
                exp = seg.exposure(env.sensors, obstacles=env.obstacles)
                reward = -(td3_params['w_exp'] * exp + td3_params['w_len'] * seg.length() * 0.1)
                if i == len(xs) - 1:
                    reward, truncated = reward + 100.0, True

            next_state = np.array([xs[i], action[0]], dtype=np.float32)
            replay_buffer.push(state, action, reward, next_state, terminated, truncated)

            if len(replay_buffer) > td3_params['batch_size']:
                agent.train(replay_buffer, td3_params['batch_size'])

            state, ep_reward = next_state, ep_reward + reward
            current_path.append(p2)
            if terminated or truncated: break

        reward_history.append(ep_reward)

        # Lưu Snapshot
        if ep in checkpoints:
            print(f"Saving snapshot at episode {ep}...")
            save_snapshot(env, current_path, ep, result_dir, f"Reward: {ep_reward:.1f}")

        if ep % 20 == 0:
            print(f"Episode {ep}/{td3_params['n_episodes']} | Reward: {ep_reward:.2f}")

    # --- Kết thúc & Vẽ hình cuối cùng ---
    print("Saving final results...")
    # Chạy một lần không noise (Greedy)
    eval_pts = [Point(xs[0], env.height / 2)]
    s = np.array([xs[0], env.height / 2], dtype=np.float32)
    for i in range(1, len(xs)):
        a = np.clip(agent.select_action(s), 0, env.height)
        p = Point(xs[i], a[0]);
        eval_pts.append(p)
        s = np.array([xs[i], a[0]], dtype=np.float32)

    final_path_obj = RobotPath(eval_pts)
    f_exp = final_path_obj.exposure(env.sensors, obstacles=env.obstacles)
    f_len = final_path_obj.length()

    # Lưu Reward Plot
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel("Episode");
    plt.ylabel("Reward")
    plt.savefig(result_dir / "reward_history.png");
    plt.close()

    # Vẽ đường đi cuối cùng lên Map
    # Gọi plot_environment_image của Huy (không truyền ax)
    plot_environment_image(env, save_path=str(result_dir / "final_map.png"))

    # Dùng matplotlib mở lại file đó và vẽ đè lên
    img = plt.imread(str(result_dir / "final_map.png"))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=[0, env.width, 0, env.height])
    pts_final = np.array([[p.x, p.y] for p in eval_pts])
    ax.plot(pts_final[:, 0], pts_final[:, 1], color='red', linewidth=3, label='TD3 Best')
    plt.title(f"Final TD3 Result\nExp={f_exp:.2f}, Len={f_len:.2f}")
    plt.savefig(result_dir / "final_path_overlay.png")
    plt.close()

    print(f"All done! Results in {result_dir}")


if __name__ == "__main__":
    main()