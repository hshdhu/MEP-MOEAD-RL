import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from general.point import Point
from general.path import Path
from shapely.geometry import Point as ShapelyPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MULTI-OBJECTIVE REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════
class MOReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))

        # Lưu trữ 3 loại reward độc lập
        self.reward_exp = np.zeros((max_size, 1))
        self.reward_len = np.zeros((max_size, 1))
        self.reward_feas = np.zeros((max_size, 1))

        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, r_exp, r_len, r_feas, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward_exp[self.ptr] = r_exp
        self.reward_len[self.ptr] = r_len
        self.reward_feas[self.ptr] = r_feas
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward_exp[ind]).to(device),
            torch.FloatTensor(self.reward_len[ind]).to(device),
            torch.FloatTensor(self.reward_feas[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. ACTOR & TRIPLE-HEAD CRITIC
# ══════════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class MultiObjectiveCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(MultiObjectiveCritic, self).__init__()

        # --- Q1 Architecture ---
        self.q1_backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.q1_exp = nn.Linear(hidden_size, 1)
        self.q1_len = nn.Linear(hidden_size, 1)
        self.q1_feas = nn.Linear(hidden_size, 1)

        # --- Q2 Architecture ---
        self.q2_backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.q2_exp = nn.Linear(hidden_size, 1)
        self.q2_len = nn.Linear(hidden_size, 1)
        self.q2_feas = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1 outputs
        feat1 = self.q1_backbone(sa)
        v1_exp = self.q1_exp(feat1)
        v1_len = self.q1_len(feat1)
        v1_feas = self.q1_feas(feat1)

        # Q2 outputs
        feat2 = self.q2_backbone(sa)
        v2_exp = self.q2_exp(feat2)
        v2_len = self.q2_len(feat2)
        v2_feas = self.q2_feas(feat2)

        return (v1_exp, v1_len, v1_feas), (v2_exp, v2_len, v2_feas)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        feat1 = self.q1_backbone(sa)
        return self.q1_exp(feat1), self.q1_len(feat1), self.q1_feas(feat1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MO-TD3 ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
class MO_TD3:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 5
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        # Giữ nguyên 16 để tránh State Aliasing cho TD3
        self.base_state_dim = 13
        self.state_dim = 16
        self.action_dim = 1
        self.action_scale = kwargs.get('action_scale', 8.0)

        # [CẢI TIẾN 1]: Thêm biến này phục vụ cho Terminal Reward
        self.max_expected_length = np.hypot(env.width, env.height)

        self.max_episodes = kwargs.get('n_episodes', kwargs.get('n_generations', 1500))
        self.batch_size = kwargs.get('batch_size', 256)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        self.expl_noise = kwargs.get('exploration_noise', 0.2)
        self.policy_noise = kwargs.get('policy_noise', 0.2)
        self.noise_clip = kwargs.get('noise_clip', 0.5)
        self.policy_freq = kwargs.get('policy_freq', 2)

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = MultiObjectiveCritic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = MOReplayBuffer(self.state_dim, self.action_dim, kwargs.get('max_replay_buffer', 100000))
        self.total_it = 0
        self.EP = []

    # [CẢI TIẾN 2]: Sample weights giống hệt MO-PPO
    def _sample_weights(self) -> tuple[float, float]:
        if np.random.rand() < 0.15:
            return (1.0, 0.0) if np.random.rand() < 0.5 else (0.0, 1.0)
        w = np.random.uniform(0.0, 1.0)
        return float(w), float(1.0 - w)

    def get_safe_start_y_in_range(self, low, high, x=0.0):
        for _ in range(50):
            y = np.random.uniform(low, high)
            pt = ShapelyPoint(x, y)
            if all(not obs.polygon.contains(pt) and obs.polygon.distance(pt) >= 10.0 for obs in self.env.obstacles):
                return y
        return self.env.height / 2.0

    def get_base_state(self, x, y, prev_action):
        dist_top = (self.env.height - y) / self.env.height
        dist_bottom = y / self.env.height
        base_features = [x / self.env.width, y / self.env.height, prev_action, dist_top, dist_bottom]

        look_ahead = self.dx * 2.0
        look_side = self.action_scale * 2.0

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
                    if norm_d < obs_features[i]: obs_features[i] = norm_d

        sensor_points = [
            Point(x + look_ahead, y),
            Point(x + look_ahead, min(y + look_side, self.env.height)),
            Point(x + look_ahead, max(y - look_side, 0))
        ]
        sensor_features = [0.0, 0.0, 0.0]
        for i, pt in enumerate(sensor_points):
            exp_val = sum(sensor.exposure_at(pt, obstacles=self.env.obstacles) for sensor in self.env.sensors)
            sensor_features[i] = min(exp_val / 2.0, 1.0)

        return np.array(base_features + obs_features + sensor_features, dtype=np.float32)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()[0]
        if add_noise:
            noise = np.random.normal(0, self.expl_noise)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def train(self):
        if self.replay_buffer.size < self.batch_size: return
        self.total_it += 1

        state, action, r_exp, r_len, r_feas, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Bóc tách weights (tại 3 vị trí cuối của vector state kích thước 16)
        w_exp = state[:, -3].unsqueeze(1)
        w_len = state[:, -2].unsqueeze(1)
        w_feas = state[:, -1].unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            q1_tups, q2_tups = self.critic_target(next_state, next_action)

            # Lấy min cho từng objective để chống Overestimation
            target_q_exp = r_exp + (1 - done) * self.gamma * torch.min(q1_tups[0], q2_tups[0])
            target_q_len = r_len + (1 - done) * self.gamma * torch.min(q1_tups[1], q2_tups[1])
            target_q_feas = r_feas + (1 - done) * self.gamma * torch.min(q1_tups[2], q2_tups[2])

        q1_curr, q2_curr = self.critic(state, action)

        # Tổng hợp Loss của Critic cho 3 nhánh mục tiêu
        critic_loss_exp = F.mse_loss(q1_curr[0], target_q_exp) + F.mse_loss(q2_curr[0], target_q_exp)
        critic_loss_len = F.mse_loss(q1_curr[1], target_q_len) + F.mse_loss(q2_curr[1], target_q_len)
        critic_loss_feas = F.mse_loss(q1_curr[2], target_q_feas) + F.mse_loss(q2_curr[2], target_q_feas)

        critic_loss = critic_loss_exp + critic_loss_len + critic_loss_feas

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor trễ (Delayed Update)
        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)
            q1_exp_a, q1_len_a, q1_feas_a = self.critic.Q1(state, actor_action)

            # Scalarization ngay tại hàm Loss của Actor (dựa vào weights của môi trường)
            q_scalar = w_exp * q1_exp_a + w_len * q1_len_a + w_feas * q1_feas_a

            actor_loss = -q_scalar.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update mạng target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def update_ep(self, points, objs):
        if objs[0] == float('inf'): return
        if any(self._dominates(o, objs) for _, o in self.EP): return
        self.EP = [(p, o) for p, o in self.EP if not self._dominates(objs, o)]
        self.EP.append((points, objs))

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path): return float('inf'), float('inf')
        return -path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles), path.length()

    def run(self, verbose=True, callback=None):
        for episode in range(1, self.max_episodes + 1):

            # Sử dụng logic sample weights đã cải tiến
            w_exp, w_len = self._sample_weights()
            w_feas = max(0.2, 1.0 - 0.8 * (episode / self.max_episodes))

            sector = episode % 5
            target_y = [0.9, 0.7, 0.5, 0.3, 0.1][sector] * self.env.height
            low = max(0.0, target_y - 5.0)
            high = min(self.env.height, target_y + 5.0)
            state_y = self.get_safe_start_y_in_range(low, high, self.xs[0])

            prev_x, prev_action = self.xs[0], 0.0
            current_pts = [Point(prev_x, state_y)]
            crashed = False
            episode_transitions = []

            for i, x in enumerate(self.xs[1:]):
                base_state = self.get_base_state(prev_x, state_y, prev_action)

                # Nối đủ 3 trọng số vào state
                state = np.append(base_state, [w_exp, w_len, w_feas]).astype(np.float32)

                action = self.select_action(state, add_noise=True)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0.0, self.env.height)

                prev_point, next_point = Point(prev_x, state_y), Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                # Thu thập 3 mảng Reward
                r_exp = sum(s.exposure_on_segment(prev_point, next_point, 1.0, self.env.obstacles) for s in
                            self.env.sensors) * 2.0
                r_len = -0.05 - abs(action - prev_action) * 0.05
                r_feas = 0.1

                done = False
                if raw_next_y <= 0.0 or raw_next_y >= self.env.height: crashed = done = True

                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = done = True
                        break
                    else:
                        dist_obs = obs.polygon.distance(next_shapely)
                        if dist_obs < 4.0:
                            r_feas -= (4.0 - dist_obs) * 0.5

                next_base_state = self.get_base_state(x, next_y, action)
                next_state = np.append(next_base_state, [w_exp, w_len, w_feas]).astype(np.float32)

                if crashed:
                    progress_ratio = x / self.env.width
                    crash_penalty = -50.0 - (50.0 * progress_ratio)

                    # Reward r_feas bị gánh penalty
                    episode_transitions.append((state, action, r_exp, r_len, r_feas + crash_penalty, next_state, done))
                    current_pts.append(next_point)
                    break

                if i == len(self.xs[1:]) - 1: done = True

                episode_transitions.append((state, action, r_exp, r_len, r_feas, next_state, done))

                state_y, prev_x, prev_action = next_y, x, action
                current_pts.append(next_point)

            # [CẢI TIẾN 3]: Đánh giá cuối đường đi & Add Terminal Bonus giống hệt MO-PPO
            if not crashed:
                objs = self.evaluate_path(current_pts)
                self.update_ep(current_pts, objs)

                if objs[0] != float('inf') and len(episode_transitions) > 0:
                    s, a, r_e, r_l, r_f, ns, d = episode_transitions[-1]

                    actual_exp = -objs[0]
                    actual_len = objs[1]

                    new_r_e = r_e + (actual_exp * 0.5)
                    new_r_l = r_l + (max(0.0, self.max_expected_length - actual_len) * 0.5)
                    new_r_f = r_f + 50.0

                    # Gán lại giá trị reward điểm cuối
                    episode_transitions[-1] = (s, a, new_r_e, new_r_l, new_r_f, ns, d)
            else:
                objs = (float('inf'), float('inf'))

            # Đưa Data vào Replay Buffer & Train mạng
            for (s, a, r_e, r_l, r_f, ns, d) in episode_transitions:
                self.replay_buffer.add(s, np.array([a]), r_e, r_l, r_f, ns, d)
                self.train()  # TD3 training step

            if verbose and episode % 100 == 0:
                exp_str = f"{-objs[0]:.3f}" if objs[0] != float('inf') else "Crash"
                len_str = f"{objs[1]:.1f}" if objs[0] != float('inf') else "Crash"
                print(
                    f"Gen {episode:4d}/{self.max_episodes} | w=({w_exp:.2f},{w_len:.2f}) fw={w_feas:.2f} | Exp: {exp_str:>8} | Len: {len_str:>8} | Pareto: {len(self.EP):3d}")

            if callback: callback(self, episode)

    def pareto_front(self):
        return self.EP