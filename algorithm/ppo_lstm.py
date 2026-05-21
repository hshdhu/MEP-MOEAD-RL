import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from general.point import Point
from general.path import Path

import math
from shapely.geometry import Point as ShapelyPoint, LineString


class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCriticLSTM, self).__init__()

        self.hidden_size = hidden_size

        # LSTM layer đóng vai trò là "Bộ nhớ ngắn hạn"
        # batch_first=True nghĩa là input có dạng (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)

        # Actor quyết định hành động dựa trên bộ nhớ
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        # Critic đánh giá giá trị trạng thái dựa trên bộ nhớ
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state, hidden):
        # Truyền qua LSTM
        lstm_out, hidden_out = self.lstm(state, hidden)

        # Lấy output của LSTM đưa vào Actor và Critic
        value = self.critic(lstm_out)
        mu = self.actor(lstm_out)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, value, hidden_out

    def evaluate(self, state, action, hidden):
        dist, value, _ = self.forward(state, hidden)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy


class PPO_LSTM:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 5
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        # 13 tính năng: 4 cơ bản + 5 Lidar vật cản + 3 Lidar sóng + 1 hành động trước
        self.state_dim = 13
        self.action_dim = 1
        self.action_scale = 8.0

        # [QUAN TRỌNG] Tầm nhìn của UAV bị giới hạn (Local Exploration)
        self.sensor_range = 25.0

        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_clip = kwargs.get('clip_range', 0.2)
        self.K_epochs = kwargs.get('n_epochs', 10)
        self.max_episodes = kwargs.get('n_generations', 500)
        self.hidden_size = kwargs.get('hidden_size', 128)

        self.policy = ActorCriticLSTM(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCriticLSTM(self.state_dim, self.action_dim, self.hidden_size)
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
        return self.env.height / 2

    def get_state(self, x, y, prev_action):
        """ Tạo state với tầm nhìn giới hạn (bị mù ở xa). LSTM sẽ lo việc ghi nhớ. """
        norm_x = x / self.env.width
        norm_y = y / self.env.height
        dist_top = (self.env.height - y) / self.env.height
        dist_bottom = y / self.env.height
        base_features = [norm_x, norm_y, prev_action, dist_top, dist_bottom]

        # Tia quét xung quanh, độ dài tối đa là sensor_range
        points_to_check = [
            ShapelyPoint(x + self.sensor_range, y),
            ShapelyPoint(x + self.sensor_range, min(y + self.sensor_range, self.env.height)),
            ShapelyPoint(x + self.sensor_range, max(y - self.sensor_range, 0)),
            ShapelyPoint(x, min(y + self.sensor_range, self.env.height)),
            ShapelyPoint(x, max(y - self.sensor_range, 0))
        ]

        # Mặc định là 1.0 (Không thấy gì)
        obs_features = [1.0] * 5
        curr_pt = ShapelyPoint(x, y)

        for i, pt in enumerate(points_to_check):
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt):
                    obs_features[i] = 0.0
                    break
                else:
                    d = obs.polygon.distance(curr_pt)
                    # Chỉ phát hiện nếu vật cản nằm TRONG tầm nhìn
                    if d < self.sensor_range:
                        norm_d = d / self.sensor_range
                        if norm_d < obs_features[i]:
                            obs_features[i] = norm_d

        # Ngửi sóng trong khu vực gần
        sensor_points = [
            Point(x + self.dx, y),
            Point(x + self.dx, min(y + self.action_scale, self.env.height)),
            Point(x + self.dx, max(y - self.action_scale, 0))
        ]
        sensor_features = [0.0, 0.0, 0.0]
        for i, pt in enumerate(sensor_points):
            exp_val = 0.0
            for sensor in self.env.sensors:
                exp_val += sensor.exposure_at(pt, obstacles=self.env.obstacles)
            sensor_features[i] = min(exp_val / 2.0, 1.0)

        return np.array(base_features + obs_features + sensor_features, dtype=np.float32)

    def select_action(self, state, hidden):
        with torch.no_grad():
            # Chuyển state thành dạng (batch=1, seq_len=1, features) cho LSTM
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            dist, _, hidden = self.policy_old(state, hidden)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), hidden

    def update_ppo(self, states, actions, logprobs, rewards):
        if len(rewards) == 0: return

        # 1. Tính Discounted Returns
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # Shape: (1, seq_len)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)
        if returns.size(1) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 2. Chuẩn bị Batch dạng Sequence cho LSTM
        # Shape yêu cầu: (batch=1, seq_len, dim)
        old_states = torch.FloatTensor(np.array(states)).unsqueeze(0)
        old_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0).unsqueeze(2)
        old_logprobs = torch.FloatTensor(np.array(logprobs)).unsqueeze(0).unsqueeze(2)

        for _ in range(self.K_epochs):
            # Khởi tạo bộ nhớ trống ở đầu mỗi quá trình học
            h0 = torch.zeros(1, 1, self.hidden_size)
            c0 = torch.zeros(1, 1, self.hidden_size)
            hidden = (h0, c0)

            # Đưa toàn bộ chuỗi bay qua mạng cùng lúc
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, hidden)

            # state_values trả về có dạng (1, seq_len, 1), ta bóp lại thành (1, seq_len)
            state_values = state_values.squeeze(2)

            # Tính Advantage
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()

            # PPO Loss
            surr1 = ratios.squeeze(2) * advantages
            surr2 = torch.clamp(ratios.squeeze(2), 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns)

            # Khuyến khích khám phá (Entropy)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            # Tối ưu hóa
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path):
            return float('inf'), float('inf')

        exp = path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles)
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

            # Lựa chọn cổng xuất phát (Trên, Giữa, Dưới)
            sector = episode % 3
            if sector == 0:
                target_y = 0.8 * self.env.height
            elif sector == 1:
                target_y = 0.5 * self.env.height
            else:
                target_y = 0.2 * self.env.height

            low = max(0, target_y - 5.0)
            high = min(self.env.height, target_y + 5.0)
            state_y = self.get_safe_start_y_in_range(low, high, self.xs[0])

            prev_x = self.xs[0]
            prev_action = 0.0
            current_points = [Point(prev_x, state_y)]
            crashed = False

            # [QUAN TRỌNG] Khởi tạo bộ nhớ (Hidden State) đầu mỗi chặng bay
            hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

            for x in self.xs[1:]:
                # Agent chỉ thấy vùng xung quanh
                state = self.get_state(prev_x, state_y, prev_action)

                # Truyền state + bộ nhớ cũ -> nhận action + bộ nhớ mới
                action, log_prob, hidden = self.select_action(state, hidden)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0, self.env.height)

                prev_point = Point(prev_x, state_y)
                next_point = Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                # Thưởng cơ bản
                step_reward = 1.0
                step_reward -= abs(action - prev_action) * 0.05  # Phạt bẻ lái gắt

                # Thưởng tìm thấy sóng
                step_exposure = 0.0
                for sensor in self.env.sensors:
                    step_exposure += sensor.exposure_on_segment(prev_point, next_point, step=1.0,
                                                                obstacles=self.env.obstacles)
                step_reward += (step_exposure * 2.0)

                # Kiểm tra va chạm
                if raw_next_y <= 0 or raw_next_y >= self.env.height:
                    crashed = True
                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = True
                        break
                    else:
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0:
                            step_reward -= (4.0 - dist) * 0.5  # Cảnh báo nếu bay quá gần vật cản

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)

                if crashed:
                    progress_ratio = x / self.env.width
                    crash_penalty = -50.0 - (50.0 * progress_ratio)
                    rewards.append(step_reward + crash_penalty)
                    current_points.append(next_point)
                    break

                rewards.append(step_reward)
                state_y = next_y
                prev_x = x
                prev_action = action
                current_points.append(next_point)

            if not crashed:
                objs = self.evaluate_path(current_points)
                self.update_ep(current_points, objs)

                if objs[0] != float('inf'):
                    actual_exposure = -objs[0]
                    actual_length = objs[1]
                    terminal_reward = 50.0 + (actual_exposure * 1.5) - (actual_length * 0.05)
                    rewards[-1] += terminal_reward
            else:
                objs = [float('inf'), float('inf')]

            # Huấn luyện chuỗi thời gian (Sequence)
            self.update_ppo(states, actions, log_probs, rewards)

            if verbose and episode % 100 == 0:
                total_reward = sum(rewards)
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