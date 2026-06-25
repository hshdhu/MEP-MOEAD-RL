import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from general.point import Point
from general.path import Path
from shapely.geometry import Point as ShapelyPoint
from collections import deque

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

        # 3 luồng Reward độc lập
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
# 2. ACTOR & MULTI-OBJECTIVE TWIN CRITIC
# ══════════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # Sửa lỗi log_prob của Tanh transform (chuẩn SAC)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class MultiObjectiveCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MultiObjectiveCritic, self).__init__()

        # --- Twin 1 ---
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_exp = nn.Linear(hidden_dim, 1)
        self.q1_len = nn.Linear(hidden_dim, 1)
        self.q1_feas = nn.Linear(hidden_dim, 1)

        # --- Twin 2 ---
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_exp = nn.Linear(hidden_dim, 1)
        self.q2_len = nn.Linear(hidden_dim, 1)
        self.q2_feas = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1
        feat1 = F.relu(self.q1_l2(F.relu(self.q1_l1(sa))))
        v1_exp = self.q1_exp(feat1)
        v1_len = self.q1_len(feat1)
        v1_feas = self.q1_feas(feat1)

        # Q2
        feat2 = F.relu(self.q2_l2(F.relu(self.q2_l1(sa))))
        v2_exp = self.q2_exp(feat2)
        v2_len = self.q2_len(feat2)
        v2_feas = self.q2_feas(feat2)

        return (v1_exp, v1_len, v1_feas), (v2_exp, v2_len, v2_feas)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MO-SAC ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
class MO_SAC:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 6
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        # Khớp State Dim = 16 (13 Base + 3 Weights) y hệt MO-PPO / MO-TD3
        self.base_state_dim = 13
        self.state_dim = 16
        self.action_dim = 1
        self.action_scale = kwargs.get('action_scale', 6.0)

        # [CẢI TIẾN 1]: Thêm biến max_expected_length
        self.max_expected_length = np.hypot(env.width, env.height)

        self.max_episodes = kwargs.get('n_generations', 1500)
        self.batch_size = kwargs.get('batch_size', 256)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic = MultiObjectiveCritic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target = MultiObjectiveCritic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Tự động điều chỉnh Alpha (Entropy Weight)
        self.target_entropy = -self.action_dim
        self.log_alpha = torch.tensor([np.log(0.2)], requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = MOReplayBuffer(self.state_dim, self.action_dim)
        self.EP = []

        self.current_actor_loss = 0.0
        self.current_critic_loss = 0.0
        self.current_reward_exp = 0.0
        self.current_reward_len = 0.0
        self.current_reward_feas = 0.0
        self.current_value = 0.0

        self.recent_successes = deque(maxlen=100)
        self.history_success_rate = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

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
            ShapelyPoint(x + look_ahead, y), ShapelyPoint(x + look_ahead, min(y + look_side, self.env.height)),
            ShapelyPoint(x + look_ahead, max(y - look_side, 0)), ShapelyPoint(x, min(y + look_side, self.env.height)),
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
                    if min(d, 20.0) / 20.0 < obs_features[i]: obs_features[i] = min(d, 20.0) / 20.0

        sensor_points = [Point(x + look_ahead, y), Point(x + look_ahead, min(y + look_side, self.env.height)),
                         Point(x + look_ahead, max(y - look_side, 0))]
        sensor_features = [0.0, 0.0, 0.0]
        for i, pt in enumerate(sensor_points):
            exp_val = sum(sensor.exposure_at(pt, obstacles=self.env.obstacles) for sensor in self.env.sensors)
            sensor_features[i] = min(exp_val / 2.0, 1.0)

        return np.array(base_features + obs_features + sensor_features, dtype=np.float32)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(state)
                return torch.tanh(mu).cpu().data.numpy().flatten()[0]
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()[0]

    def train(self):
        if self.replay_buffer.size < self.batch_size: return
        s, a, r_e, r_l, r_f, s2, d = self.replay_buffer.sample(self.batch_size)

        # Bóc tách Weights từ State 16 chiều
        w_exp = s[:, -3].unsqueeze(1)
        w_len = s[:, -2].unsqueeze(1)
        w_feas = s[:, -1].unsqueeze(1)

        # ---------------- CRITIC UPDATE ----------------
        with torch.no_grad():
            next_a, next_lp = self.actor.sample(s2)
            (q1_e_t, q1_l_t, q1_f_t), (q2_e_t, q2_l_t, q2_f_t) = self.critic_target(s2, next_a)

            # Tính Target Q cho từng objective (Có kèm Entropy Bonus của SAC)
            target_q_exp = r_e + (1 - d) * self.gamma * (torch.min(q1_e_t, q2_e_t) - self.alpha * next_lp)
            target_q_len = r_l + (1 - d) * self.gamma * (torch.min(q1_l_t, q2_l_t) - self.alpha * next_lp)
            target_q_feas = r_f + (1 - d) * self.gamma * (torch.min(q1_f_t, q2_f_t) - self.alpha * next_lp)

        (c1_e, c1_l, c1_f), (c2_e, c2_l, c2_f) = self.critic(s, a)

        critic_loss = (F.mse_loss(c1_e, target_q_exp) + F.mse_loss(c2_e, target_q_exp) +
                       F.mse_loss(c1_l, target_q_len) + F.mse_loss(c2_l, target_q_len) +
                       F.mse_loss(c1_f, target_q_feas) + F.mse_loss(c2_f, target_q_feas))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- ACTOR UPDATE ----------------
        new_a, lp = self.actor.sample(s)
        (q1_e_p, q1_l_p, q1_f_p), (q2_e_p, q2_l_p, q2_f_p) = self.critic(s, new_a)

        # Scalarization (Tổng hợp theo trọng số)
        q1_scalar = w_exp * q1_e_p + w_len * q1_l_p + w_feas * q1_f_p
        q2_scalar = w_exp * q2_e_p + w_len * q2_l_p + w_feas * q2_f_p

        self.current_value = q1_scalar.mean().item()

        # Tối ưu hóa điểm nhỏ nhất (Twin SAC logic) + Maximize Entropy
        actor_loss = (self.alpha * lp - torch.min(q1_scalar, q2_scalar)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------- ALPHA UPDATE ----------------
        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft Update Critic Target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.current_actor_loss = actor_loss.item()
        self.current_critic_loss = critic_loss.item()

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path): return float('inf'), float('inf')
        return -path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles), path.length()

    def update_ep(self, points, objs):
        if objs[0] == float('inf'): return
        if any(self._dominates(o, objs) for _, o in self.EP): return
        self.EP = [(p, o) for p, o in self.EP if not self._dominates(objs, o)]
        self.EP.append((points, objs))

    def run(self, verbose=True, callback=None):
        for episode in range(1, self.max_episodes + 1):

            # Sử dụng logic sample weights đã cải tiến
            w_exp, w_len = self._sample_weights()
            w_feas = max(0.2, 1.0 - 0.8 * (episode / self.max_episodes))

            sector = episode % 5
            target_y = [0.9, 0.7, 0.5, 0.3, 0.1][sector] * self.env.height
            state_y = self.get_safe_start_y_in_range(max(0, target_y - 15.0), min(self.env.height, target_y + 15.0),
                                                     self.xs[0])

            prev_x, prev_action = self.xs[0], 0.0
            current_points = [Point(prev_x, state_y)]
            crashed = False
            episode_transitions = []

            for i, x in enumerate(self.xs[1:]):
                base_state = self.get_base_state(prev_x, state_y, prev_action)
                state = np.append(base_state, [w_exp, w_len, w_feas]).astype(np.float32)

                action = self.select_action(state)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0.0, self.env.height)

                prev_point, next_point, next_shapely = Point(prev_x, state_y), Point(x, next_y), ShapelyPoint(x, next_y)

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
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0: r_feas -= (4.0 - dist) * 0.5

                next_base_state = self.get_base_state(x, next_y, action)
                next_state = np.append(next_base_state, [w_exp, w_len, w_feas]).astype(np.float32)

                if crashed:
                    progress_ratio = x / self.env.width
                    crash_penalty = -50.0 - (50.0 * progress_ratio)
                    episode_transitions.append((state, action, r_exp, r_len, r_feas + crash_penalty, next_state, done))
                    current_points.append(next_point)
                    break

                if i == len(self.xs[1:]) - 1: done = True

                episode_transitions.append((state, action, r_exp, r_len, r_feas, next_state, done))

                state_y, prev_x, prev_action = next_y, x, action
                current_points.append(next_point)

            # [CẢI TIẾN 3]: Đánh giá cuối đường đi & Add Terminal Bonus giống hệt MO-PPO / MO-TD3
            if not crashed:
                self.recent_successes.append(1) # [THÊM]
                objs = self.evaluate_path(current_points) # (Đối với PPO/TD3 là current_pts)
                self.update_ep(current_points, objs)

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
                self.recent_successes.append(0)
                objs = (float('inf'), float('inf'))

            curr_sr = (sum(self.recent_successes) / len(self.recent_successes)) * 100.0 if self.recent_successes else 0.0
            self.history_success_rate.append(curr_sr)

            if len(episode_transitions) > 0:
                self.current_reward_exp = sum([t[2] for t in episode_transitions])
                self.current_reward_len = sum([t[3] for t in episode_transitions])
                self.current_reward_feas = sum([t[4] for t in episode_transitions])

            for (s, a, r_e, r_l, r_f, ns, d) in episode_transitions:
                self.replay_buffer.add(s, np.array([a]), r_e, r_l, r_f, ns, d)
                self.train()

            if verbose and episode % 100 == 0:
                exp_str = f"{-objs[0]:.3f}" if objs[0] != float('inf') else "Crash"
                len_str = f"{objs[1]:.1f}" if objs[0] != float('inf') else "Crash"
                print(
                    f"Gen {episode:4d}/{self.max_episodes} | w=({w_exp:.2f},{w_len:.2f}) fw={w_feas:.2f} | Exp: {exp_str:>8} | Len: {len_str:>8} | Pareto: {len(self.EP):3d}")

            if callback: callback(self, episode)

    def pareto_front(self):
        return self.EP