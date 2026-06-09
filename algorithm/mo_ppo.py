import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math

from general.point import Point
from general.path import Path
from shapely.geometry import Point as ShapelyPoint, LineString


# ══════════════════════════════════════════════════════════════════════════════
# 1. ACTOR-CRITIC — SHARED BACKBONE + TRIPLE CRITIC HEAD
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        self.critic_exp  = nn.Linear(hidden_size, 1)
        self.critic_len  = nn.Linear(hidden_size, 1)
        self.critic_feas = nn.Linear(hidden_size, 1)

        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        feat = self.backbone(state)
        mu   = self.actor_head(feat)
        std  = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        v_exp  = self.critic_exp(feat).squeeze(-1)
        v_len  = self.critic_len(feat).squeeze(-1)
        v_feas = self.critic_feas(feat).squeeze(-1)

        return dist, v_exp, v_len, v_feas

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        dist, v_exp, v_len, v_feas = self.forward(state)
        action_logprobs = dist.log_prob(action).squeeze(-1)
        dist_entropy    = dist.entropy().squeeze(-1)

        return action_logprobs, v_exp, v_len, v_feas, dist_entropy


# ══════════════════════════════════════════════════════════════════════════════
# 2. MO-PPO ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

class PPO:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx  = 5
        self.xs  = list(np.arange(0, env.width + 1, self.dx))

        self.base_state_dim = 13
        self.state_dim      = 16  # State bao gồm w_exp, w_len, w_feas
        self.action_dim     = 1
        self.action_scale   = 8.0

        self.lr           = kwargs.get('lr',            3e-4)
        self.gamma        = kwargs.get('gamma',         0.99)
        self.eps_clip     = kwargs.get('clip_range',    0.2)
        self.K_epochs     = kwargs.get('n_epochs',      10)
        self.max_episodes = kwargs.get('n_generations', 1500)

        self.timesteps_per_batch = kwargs.get('timesteps_per_batch', 512)
        self.max_expected_length = np.hypot(env.width, env.height)

        self.policy     = ActorCritic(self.state_dim, self.action_dim)
        self.policy_old = ActorCritic(self.state_dim, self.action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss   = nn.MSELoss()

        self.EP: list = []

    def _sample_weights(self, episode: int) -> tuple[float, float]:
        """
        Tỷ lệ rơi vào điểm biên giảm dần:
        - Lúc đầu: 20% (Khám phá nhanh giới hạn cực đại/cực tiểu)
        - Lúc cuối: 5% (Chỉ để nhắc nhở mạng nơ-ron không bị quên)
        - Còn lại: Random liên tục để lấp đầy Pareto Front.
        """
        # Tính xác suất chọn điểm biên (từ 0.2 giảm mượt về 0.05)
        prob_boundary = max(0.05, 0.20 - 0.15 * (episode / self.max_episodes))

        if np.random.rand() < prob_boundary:
            # 50/50 chọn 1 trong 2 biên
            if np.random.rand() < 0.5:
                return 1.0, 0.0  # Ưu tiên Max Exposure tuyệt đối
            else:
                return 0.0, 1.0  # Ưu tiên Min Length tuyệt đối
        else:
            # Khám phá không gian giữa (Pareto Front)
            w = np.random.uniform(0.0, 1.0)
            return float(w), float(1.0 - w)

    def get_safe_start_y_in_range(self, low: float, high: float, x: float = 0.0) -> float:
        for _ in range(50):
            y  = np.random.uniform(low, high)
            pt = ShapelyPoint(x, y)
            if all(not obs.polygon.contains(pt) and obs.polygon.distance(pt) >= 10.0 for obs in self.env.obstacles):
                return y
        return self.env.height / 2.0

    def get_base_state(self, x: float, y: float, prev_action: float) -> np.ndarray:
        dist_top    = (self.env.height - y) / self.env.height
        dist_bottom = y / self.env.height
        base        = [x / self.env.width, y / self.env.height, prev_action, dist_top, dist_bottom]

        look_ahead = self.dx * 2.0
        look_side  = self.action_scale * 2.0

        check_pts = [
            ShapelyPoint(x + look_ahead, y),
            ShapelyPoint(x + look_ahead, min(y + look_side, self.env.height)),
            ShapelyPoint(x + look_ahead, max(y - look_side, 0.0)),
            ShapelyPoint(x,              min(y + look_side, self.env.height)),
            ShapelyPoint(x,              max(y - look_side, 0.0)),
        ]

        obs_feat = [1.0] * 5
        for i, pt in enumerate(check_pts):
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt):
                    obs_feat[i] = 0.0
                    break
                d = min(obs.polygon.distance(pt), 20.0) / 20.0
                obs_feat[i] = min(obs_feat[i], d)

        sensor_pts = [
            Point(x + look_ahead, y),
            Point(x + look_ahead, min(y + look_side, self.env.height)),
            Point(x + look_ahead, max(y - look_side, 0.0)),
        ]

        sensor_feat = []
        for pt in sensor_pts:
            exp_val = sum(s.exposure_at(pt, obstacles=self.env.obstacles) for s in self.env.sensors)
            sensor_feat.append(min(exp_val / 2.0, 1.0))

        return np.array(base + obs_feat + sensor_feat, dtype=np.float32)

    def select_action(self, state: np.ndarray):
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            dist, _, _, _ = self.policy_old(state_t)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    @staticmethod
    def _dominates(a, b) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def update_ep(self, points, objs):
        if objs[0] == float('inf'): return
        if any(self._dominates(o, objs) for _, o in self.EP): return
        self.EP = [(p, o) for p, o in self.EP if not self._dominates(objs, o)]
        self.EP.append((points, objs))

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path): return float('inf'), float('inf')
        exp    = path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles)
        length = path.length()
        return -exp, length

    @staticmethod
    def _discount(rewards: list, gamma: float) -> list:
        rets, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            rets.insert(0, G)
        return rets

    def update_ppo(self, states, actions, logprobs, returns_exp, returns_len, returns_feas, weights):
        if not returns_exp: return

        w_exp_t  = torch.tensor([w[0] for w in weights], dtype=torch.float32)
        w_len_t  = torch.tensor([w[1] for w in weights], dtype=torch.float32)
        w_feas_t = torch.tensor([w[2] for w in weights], dtype=torch.float32)

        ret_exp_t  = torch.tensor(returns_exp,  dtype=torch.float32)
        ret_len_t  = torch.tensor(returns_len,  dtype=torch.float32)
        ret_feas_t = torch.tensor(returns_feas, dtype=torch.float32)

        def normalize(tensor):
            if tensor.numel() > 1:
                return (tensor - tensor.mean()) / (tensor.std() + 1e-8)
            return tensor

        ret_exp_norm  = normalize(ret_exp_t)
        ret_len_norm  = normalize(ret_len_t)
        ret_feas_norm = normalize(ret_feas_t)

        scal_returns = w_exp_t * ret_exp_norm + w_len_t * ret_len_norm + w_feas_t * ret_feas_norm

        old_states   = torch.FloatTensor(np.array(states))
        old_actions  = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        old_logprobs = torch.FloatTensor(np.array(logprobs))

        with torch.no_grad():
            _, old_v_exp, old_v_len, old_v_feas, _ = self.policy_old.evaluate(old_states, old_actions)
            old_v_scalar = w_exp_t * old_v_exp + w_len_t * old_v_len + w_feas_t * old_v_feas

        advantages = scal_returns - old_v_scalar
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            logprobs_new, v_exp, v_len, v_feas, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs_new - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (self.MseLoss(v_exp,  ret_exp_norm) +
                           self.MseLoss(v_len,  ret_len_norm) +
                           self.MseLoss(v_feas, ret_feas_norm))

            loss = actor_loss + 0.5 * critic_loss - 0.015 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def run(self, verbose: bool = True, callback=None):
        batch_states, batch_actions, batch_logprobs = [], [], []
        batch_ret_exp, batch_ret_len, batch_ret_feas, batch_weights = [], [], [], []
        total_timesteps = 0

        for episode in range(1, self.max_episodes + 1):
            w_exp, w_len = self._sample_weights(episode)

            # Smooth Decay: w_feas giảm từ 1.0 -> 0.2 trên toàn bộ quá trình training
            w_feas = max(0.2, 1.0 - 0.8 * (episode / self.max_episodes))

            sector   = episode % 5
            target_y = [0.9, 0.7, 0.5, 0.3, 0.1][sector] * self.env.height
            low      = max(0.0, target_y - 5.0)
            high     = min(self.env.height, target_y + 5.0)
            state_y  = self.get_safe_start_y_in_range(low, high, self.xs[0])

            states, actions, log_probs = [], [], []
            rewards_exp, rewards_len, rewards_feas = [], [], []

            prev_x, prev_action = self.xs[0], 0.0
            current_pts = [Point(prev_x, state_y)]
            crashed = False

            for x in self.xs[1:]:
                base_state = self.get_base_state(prev_x, state_y, prev_action)

                # Append toàn bộ 3 loại weights vào State (State_dim = 16)
                state = np.append(base_state, [w_exp, w_len, w_feas]).astype(np.float32)

                action, log_prob = self.select_action(state)
                raw_next_y = state_y + action * self.action_scale
                next_y     = np.clip(raw_next_y, 0.0, self.env.height)

                prev_point, next_point = Point(prev_x, state_y), Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                # Step Rewards (Tích luỹ từng đoạn)
                r_exp = sum(s.exposure_on_segment(prev_point, next_point, step=1.0, obstacles=self.env.obstacles) for s in self.env.sensors) * 2.0
                r_len = -0.05 - abs(action - prev_action) * 0.05
                r_feas = 0.1

                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = True
                        break
                    else:
                        dist_obs = obs.polygon.distance(next_shapely)
                        if dist_obs < 4.0:
                            r_feas -= (4.0 - dist_obs) * 0.5

                if raw_next_y <= 0.0 or raw_next_y >= self.env.height:
                    crashed = True

                # CHÚ Ý LUỒNG CODE:
                # Append state/action TRƯỚC KHI xử lý Crash break loop.
                # Điều này ĐÚNG CHUẨN RL: Agent bắt buộc phải lưu lại bước đi sai lầm
                # để Critic học được Penalty và Actor né hành vi đó ra.
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)

                if crashed:
                    progress_ratio = x / self.env.width
                    crash_penalty  = -50.0 - 50.0 * progress_ratio

                    rewards_exp.append(r_exp)
                    rewards_len.append(r_len)
                    rewards_feas.append(r_feas + crash_penalty)
                    current_pts.append(next_point)
                    break

                rewards_exp.append(r_exp)
                rewards_len.append(r_len)
                rewards_feas.append(r_feas)

                state_y, prev_x, prev_action = next_y, x, action
                current_pts.append(next_point)

            # Terminal Bonus: Phần thưởng hoàn thành lộ trình (Chấm dứt Double-Counting)
            if not crashed:
                objs = self.evaluate_path(current_pts)
                self.update_ep(current_pts, objs)

                if objs[0] != float('inf') and len(rewards_exp) > 0:
                    # Gán thưởng thuần tuý (Flat bonus) làm mỏ neo vạch đích.
                    # Không dùng actual_exposure hay actual_length để nhân hệ số nữa.
                    rewards_exp[-1]  += 10.0
                    rewards_len[-1]  += 5.0
                    rewards_feas[-1] += 50.0
            else:
                objs = (float('inf'), float('inf'))

            ret_exp  = self._discount(rewards_exp,  self.gamma)
            ret_len  = self._discount(rewards_len,  self.gamma)
            ret_feas = self._discount(rewards_feas, self.gamma)

            ep_len = len(states)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_logprobs.extend(log_probs)
            batch_ret_exp.extend(ret_exp)
            batch_ret_len.extend(ret_len)
            batch_ret_feas.extend(ret_feas)
            batch_weights.extend([(w_exp, w_len, w_feas)] * ep_len)

            total_timesteps += ep_len

            if total_timesteps >= self.timesteps_per_batch or episode == self.max_episodes:
                self.update_ppo(batch_states, batch_actions, batch_logprobs, batch_ret_exp, batch_ret_len, batch_ret_feas, batch_weights)
                batch_states, batch_actions, batch_logprobs = [], [], []
                batch_ret_exp, batch_ret_len, batch_ret_feas, batch_weights = [], [], [], []
                total_timesteps = 0

            if verbose and episode % 100 == 0:
                exp_str = f"{-objs[0]:.3f}" if objs[0] != float('inf') else "Crash"
                len_str = f"{objs[1]:.1f}" if objs[0] != float('inf') else "Crash"
                print(f"Gen {episode:4d}/{self.max_episodes} | w=({w_exp:.2f},{w_len:.2f}) fw={w_feas:.2f} | Exp: {exp_str:>8} | Len: {len_str:>8} | Pareto: {len(self.EP):3d}")

            if callback: callback(self, episode)

    def pareto_front(self):
        return self.EP