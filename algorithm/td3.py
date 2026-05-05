import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import math

from general.point import Point
from general.path import Path
from shapely.geometry import Point as ShapelyPoint, LineString

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )


# ==========================================
# 2. ACTOR & CRITIC NETWORKS
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        # Đầu ra của Tanh thuộc [-1, 1], nhân với max_action để scale về dải hợp lệ
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # Q1 Architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Q2 Architecture (Twin Critic để chống overestimation)
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa)


# ==========================================
# 3. TD3 AGENT
# ==========================================
class TD3:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 5
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        # Kích thước state/action
        self.state_dim = 10
        self.action_dim = 1

        # Load hyperparameters từ file config
        self.max_episodes = kwargs.get('n_episodes', 500)
        self.batch_size = kwargs.get('batch_size', 128)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.lr = kwargs.get('lr', 0.0003)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        # Các tham số Noise đặc trưng của TD3
        self.max_action = kwargs.get('action_scale', 15.0)  # Tương ứng action_scale trong config
        self.expl_noise = kwargs.get('exploration_noise', 5.0)
        self.policy_noise = kwargs.get('policy_noise', 2.0)
        self.noise_clip = kwargs.get('noise_clip', 5.0)
        self.policy_freq = kwargs.get('policy_freq', 2)
        self.max_replay_buffer = kwargs.get('max_replay_buffer', 100000)

        # Trọng số cho Terminal Reward
        self.w_exp = kwargs.get('w_exp', 0.8)
        self.w_len = kwargs.get('w_len', 0.2)

        # Khởi tạo Actor & Target Actor
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        # Khởi tạo Critic & Target Critic
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=self.max_replay_buffer)

        self.total_it = 0
        self.EP = []  # Lưu Pareto Front

    def get_safe_start_y_in_range(self, low, high, x=0.0):
        max_retries = 50
        for _ in range(max_retries):
            y = np.random.uniform(low, high)
            pt = ShapelyPoint(x, y)
            is_safe = True
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt) or obs.polygon.distance(pt) < 10.0:
                    is_safe = False
                    break
            if is_safe:
                return y
        return self.env.height / 2

    def get_state(self, x, y, prev_action):
        # 1. Base features (chuẩn hóa về khoảng [0, 1] hoặc [-1, 1])
        dist_top = (self.env.height - y) / self.env.height
        dist_bottom = y / self.env.height
        norm_prev_action = prev_action / self.max_action  # Chuẩn hóa prev_action

        base_features = [x / self.env.width, y / self.env.height, norm_prev_action, dist_top, dist_bottom]

        # 2. Radar features
        look_ahead = self.dx * 2.0
        look_side = self.max_action * 2.0

        points_to_check = [
            ShapelyPoint(x + look_ahead, y),
            ShapelyPoint(x + look_ahead, min(y + look_side, self.env.height)),
            ShapelyPoint(x + look_ahead, max(y - look_side, 0)),
            ShapelyPoint(x, min(y + look_side, self.env.height)),
            ShapelyPoint(x, max(y - look_side, 0))
        ]

        obs_features = [1.0] * 5
        for i, pt in enumerate(points_to_check):
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt):
                    obs_features[i] = 0.0
                    break
                else:
                    d = obs.polygon.distance(pt)
                    norm_d = min(d, 20.0) / 20.0
                    if norm_d < obs_features[i]:
                        obs_features[i] = norm_d

        return np.array(base_features + obs_features, dtype=np.float32)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()[0]

        if add_noise:
            # Thêm Exploration Noise cho TD3
            noise = np.random.normal(0, self.expl_noise)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def train(self):
        # Chỉ train nếu buffer đủ lớn
        if self.replay_buffer.size < self.batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Thêm nhiễu vào hành động mục tiêu (Target Policy Smoothing)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Lấy min của 2 hàm Q để tính Target Q (Cốt lõi của TD3 chống overestimation)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Lấy giá trị Q hiện tại
        current_Q1, current_Q2 = self.critic(state, action)

        # Cập nhật Critic
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Cập nhật Actor trễ (Delayed Policy Updates)
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Cập nhật Target Networks (Soft Update)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path):
            return float('inf'), float('inf')

        exp = path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles)
        length = path.length()
        # Trả về -exp để thuật toán update_ep tự động hiểu là muốn Maximize Exposure
        # (Tìm giá trị âm nhất của -exp)
        return -exp, length

    def update_ep(self, points, objs):
        if objs[0] == float('inf'):
            return

        new_ep = []
        is_dominated = False

        for sol_p, sol_o in self.EP:
            if sol_o[0] <= objs[0] and sol_o[1] <= objs[1]:
                is_dominated = True
                new_ep.append((sol_p, sol_o))
            elif not (objs[0] <= sol_o[0] and objs[1] <= sol_o[1]):
                new_ep.append((sol_p, sol_o))

        if not is_dominated:
            new_ep.append((points, objs))

        self.EP = new_ep

    def run(self, verbose=True, callback=None):
        for episode in range(1, self.max_episodes + 1):
            episode_transitions = []

            # --- Curriculum Learning ---
            progress = episode / self.max_episodes
            if progress < 0.4:
                center = self.env.height / 2
                range_scale = progress * 1.5
            else:
                center = np.random.uniform(0.1 * self.env.height, 0.9 * self.env.height)
                range_scale = 1.0

            half_range = (self.env.height / 2) * min(1.0, range_scale)
            low = max(0, center - half_range)
            high = min(self.env.height, center + half_range)

            state_y = self.get_safe_start_y_in_range(low, high, self.xs[0])
            prev_x = self.xs[0]
            prev_action = 0.0

            current_points = [Point(prev_x, state_y)]
            crashed = False
            total_reward = 0

            for i, x in enumerate(self.xs[1:]):
                state = self.get_state(prev_x, state_y, prev_action)

                # TD3 Action selection (có noise để khám phá)
                action = self.select_action(state, add_noise=True)

                raw_next_y = state_y + action
                next_y = np.clip(raw_next_y, 0, self.env.height)

                prev_point = Point(prev_x, state_y)
                next_point = Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                # --- STEP REWARD ---
                step_reward = 1.0
                step_reward -= abs(action - prev_action) * 0.1
                step_reward -= abs(action) * 0.05

                done = False

                if raw_next_y <= 0 or raw_next_y >= self.env.height:
                    crashed = True
                    done = True

                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = True
                        done = True
                        break
                    else:
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0:
                            step_reward -= (4.0 - dist) * 0.5

                next_state = self.get_state(x, next_y, action)

                if crashed:
                    progress_ratio = x / self.env.width
                    crash_penalty = -50.0 - (50.0 * progress_ratio)
                    step_reward += crash_penalty

                    episode_transitions.append((state, action, step_reward, next_state, done))
                    current_points.append(next_point)
                    total_reward += step_reward
                    break

                # Nếu là bước cuối cùng (tới đích)
                if i == len(self.xs[1:]) - 1:
                    done = True

                episode_transitions.append((state, action, step_reward, next_state, done))
                total_reward += step_reward

                state_y = next_y
                prev_x = x
                prev_action = action
                current_points.append(next_point)

            # --- TÍNH PHẦN THƯỞNG KẾT THÚC VÀ CẬP NHẬT TẬP PARETO ---
            if not crashed:
                objs = self.evaluate_path(current_points)
                self.update_ep(current_points, objs)

                if objs[0] != float('inf'):
                    actual_exposure = -objs[0]  # Khôi phục giá trị dương (Max Exposure)
                    actual_length = objs[1]

                    # Thưởng kết thúc kết hợp trọng số từ file config
                    # w_exp = 0.8, w_len = 0.2
                    terminal_reward = 50.0 + (actual_exposure * self.w_exp) - (actual_length * self.w_len)
                    total_reward += terminal_reward

                    # Cộng terminal_reward vào bước cuối cùng trong buffer tạm thời
                    s, a, r, ns, d = episode_transitions[-1]
                    episode_transitions[-1] = (s, a, r + terminal_reward, ns, d)
            else:
                objs = [float('inf'), float('inf')]

            # --- LƯU VÀO REPLAY BUFFER VÀ TRAIN TD3 ---
            for (s, a, r, ns, d) in episode_transitions:
                self.replay_buffer.add(s, np.array([a]), r, ns, d)
                # TD3 là off-policy, ta train mô hình liên tục trên mỗi step (hoặc mỗi episode)
                self.train()

            # --- LOGGING ---
            if verbose and episode % 50 == 0:
                best_exp = -objs[0] if objs[0] != float('inf') else "Crash"
                progress_pct = (prev_x / self.env.width) * 100
                print(
                    f"Gen {episode:3d}/{self.max_episodes} | "
                    f"Total Rwd: {total_reward:6.1f} | "
                    f"Progress: {progress_pct:5.1f}% | "
                    f"Exp: {best_exp}"
                )

            if callback:
                callback(self, episode)

    def pareto_front(self):
        return self.EP