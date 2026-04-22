import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from general.point import Point
from general.path import Path

import math
from shapely.geometry import Point as ShapelyPoint, LineString


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, value

    def evaluate(self, state, action):
        dist, value = self.forward(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy


class PPO:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 5
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        # Đổi state_dim thành 10 để chứa thêm thông tin cảm biến radar
        self.state_dim = 10
        self.action_dim = 1
        self.action_scale = 8.0

        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_clip = kwargs.get('clip_range', 0.2)
        self.K_epochs = kwargs.get('n_epochs', 10)
        self.max_episodes = kwargs.get('n_generations', 500)

        self.policy = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCritic(self.state_dim, self.action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.EP = []

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

        # fallback nếu xui quá
        return self.env.height / 2



    def get_state(self, x, y, prev_action):
        """
        Tạo state có tính năng 'Radar' để phát hiện chướng ngại vật
        """
        # 1. Các thông số cơ bản (Tọa độ chuẩn hóa)
        dist_top = (self.env.height - y) / self.env.height
        dist_bottom = y / self.env.height
        base_features = [x / self.env.width, y / self.env.height, prev_action, dist_top, dist_bottom]

        # 2. Phát tia Radar (kiểm tra 5 điểm xung quanh phía trước)
        look_ahead = self.dx * 2.0
        look_side = self.action_scale * 2.0

        points_to_check = [
            ShapelyPoint(x + look_ahead, y),  # Phía trước
            ShapelyPoint(x + look_ahead, min(y + look_side, self.env.height)),  # Chéo lên
            ShapelyPoint(x + look_ahead, max(y - look_side, 0)),  # Chéo xuống
            ShapelyPoint(x, min(y + look_side, self.env.height)),  # Ngay phía trên
            ShapelyPoint(x, max(y - look_side, 0))  # Ngay phía dưới
        ]

        obs_features = [1.0] * 5  # 1.0 nghĩa là an toàn tuyệt đối
        for i, pt in enumerate(points_to_check):
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt):
                    obs_features[i] = 0.0  # Vật cản ngay tại đây
                    break
                else:
                    # Tính khoảng cách tới vật cản, chuẩn hóa về 0 -> 1 (0 là sát vách, 1 là cách xa)
                    d = obs.polygon.distance(pt)
                    norm_d = min(d, 20.0) / 20.0
                    if norm_d < obs_features[i]:
                        obs_features[i] = norm_d

        return np.array(base_features + obs_features, dtype=np.float32)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            dist, _ = self.policy_old(state)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update_ppo(self, states, actions, logprobs, rewards):
        # Nếu episode kết thúc sớm do va chạm, hàm này vẫn hoạt động bình thường
        if len(rewards) == 0: return

        returns = []
        discounted_reward = 0

        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        # Chuẩn hóa lợi tức (giúp huấn luyện ổn định)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.FloatTensor(np.array(states))
        old_actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        old_logprobs = torch.FloatTensor(np.array(logprobs)).unsqueeze(1)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            state_values = state_values.squeeze()

            # Nếu state_values có size 0 (do batch=1), cần reshape lại để ko lỗi
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()

            surr1 = ratios.squeeze() * advantages
            surr2 = torch.clamp(
                ratios.squeeze(),
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path):
            return float('inf'), float('inf')

        exp = path.exposure(
            self.env.sensors,
            step=1.0,
            obstacles=self.env.obstacles
        )
        length = path.length()
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
            states, actions, log_probs, rewards = [], [], [], []

            # --- Curriculum Learning (2-Phase) ---
            progress = episode / self.max_episodes

            if progress < 0.4:
                # Phase 1: Tập trung vùng giữa để học né vật cản cơ bản (0 - 40% quá trình)
                center = self.env.height / 2
                range_scale = progress * 1.5  # Mở rộng chậm
            else:
                # Phase 2: Đã cứng cáp, bắt đầu thả random khắp bản đồ để phá bias vùng giữa
                center = np.random.uniform(0.1 * self.env.height, 0.9 * self.env.height)
                range_scale = 1.0  # Mở full tầm nhìn quanh center mới

            half_range = (self.env.height / 2) * min(1.0, range_scale)

            # Đảm bảo không bị văng ra khỏi map
            low = max(0, center - half_range)
            high = min(self.env.height, center + half_range)

            # --- Safe Spawn ---
            state_y = self.get_safe_start_y_in_range(low, high, self.xs[0])

            prev_x = self.xs[0]
            prev_action = 0.0
            current_points = [Point(prev_x, state_y)]

            crashed = False

            for x in self.xs[1:]:
                # 1. Lấy State có Radar
                state = self.get_state(prev_x, state_y, prev_action)

                action, log_prob = self.select_action(state)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0, self.env.height)

                prev_point = Point(prev_x, state_y)
                next_point = Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                # --- REWARD FUNCTION CHUẨN ---
                step_reward = 1.0  # Thưởng +1 cho mỗi bước tiến tới thành công (Giúp hạn chế crash)

                # Phạt di chuyển zig-zag
                step_reward -= abs(action - prev_action) * 0.1
                step_reward -= abs(action) * 0.05

                # Kiểm tra đâm vào viền bản đồ
                if raw_next_y <= 0 or raw_next_y >= self.env.height:
                    crashed = True

                # Kiểm tra va chạm với vật cản
                for obs in self.env.obstacles:
                    # Kiểm tra xem đường đi (LineString) có cắt qua vật cản không (Chuẩn xác nhất)
                    if obs.intersects(prev_point, next_point):
                        crashed = True
                        break
                    else:
                        # Thêm hình phạt nếu đi QUÁ GẦN vật cản (Proximity Penalty)
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0:
                            step_reward -= (4.0 - dist) * 0.5  # Càng sát càng trừ nhiều

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)

                # --- ĐIỀU KIỆN DỪNG SỚM NẾU CRASH ---
                if crashed:
                    progress_ratio = x / self.env.width  # 0 → 1

                    # Chết sớm → phạt nhẹ hơn
                    # Chết gần đích → phạt nặng hơn
                    crash_penalty = -50.0 - (50.0 * progress_ratio)

                    rewards.append(step_reward + crash_penalty)
                    current_points.append(next_point)
                    break

                rewards.append(step_reward)

                # Chuyển qua step sau
                state_y = next_y
                prev_x = x
                prev_action = action
                current_points.append(next_point)

            # --- TÍNH PHẦN THƯỞNG KẾT THÚC (Nếu đến đích thành công) ---
            if not crashed:
                objs = self.evaluate_path(current_points)
                self.update_ep(current_points, objs)

                if objs[0] != float('inf'):
                    actual_exposure = -objs[0]
                    actual_length = objs[1]

                    # Thưởng đậm +50 vì đã đến tới đích thành công, và tối ưu hóa Exp/Len
                    terminal_reward = 50.0 + (actual_exposure * 1.5) - (actual_length * 0.05)
                    rewards[-1] += terminal_reward
            else:
                objs = [float('inf'), float('inf')]

            # Cập nhật Neural Network
            self.update_ppo(states, actions, log_probs, rewards)

            if verbose and episode % 100 == 0:
                total_reward = sum(rewards)
                best_exp = -objs[0] if objs[0] != float('inf') else "Crash"
                # Tính % quãng đường đi được
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