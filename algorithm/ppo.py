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

        self.state_dim = 13
        self.action_dim = 1
        self.action_scale = 8.0

        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_clip = kwargs.get('clip_range', 0.2)
        self.K_epochs = kwargs.get('n_epochs', 10)
        self.max_episodes = kwargs.get('n_generations', 500)

        self.update_episodes = 1

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

        return self.env.height / 2

    def get_state(self, x, y, prev_action):
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
                    if norm_d < obs_features[i]:
                        obs_features[i] = norm_d

        sensor_points = [
            Point(x + look_ahead, y),
            Point(x + look_ahead, min(y + look_side, self.env.height)),
            Point(x + look_ahead, max(y - look_side, 0))
        ]

        sensor_features = [0.0, 0.0, 0.0]
        for i, pt in enumerate(sensor_points):
            exp_val = 0.0
            for sensor in self.env.sensors:
                exp_val += sensor.exposure_at(pt, obstacles=self.env.obstacles)
            sensor_features[i] = min(exp_val / 2.0, 1.0)

        return np.array(base_features + obs_features + sensor_features, dtype=np.float32)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            dist, _ = self.policy_old(state)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update_ppo(self, states, actions, logprobs, returns):
        if len(returns) == 0: return

        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        old_states = torch.FloatTensor(np.array(states))
        old_actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        old_logprobs = torch.FloatTensor(np.array(logprobs)).unsqueeze(1)

        # ✅ FIX CUỐI CÙNG: Tính Advantages 1 lần duy nhất ở ngoài vòng lặp
        with torch.no_grad():
            _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
            old_state_values = old_state_values.squeeze()
            if old_state_values.dim() == 0:
                old_state_values = old_state_values.unsqueeze(0)

        advantages = returns_tensor - old_state_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.K_epochs):
            # Tính lại giá trị mới để update mạng
            logprobs_new, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            state_values = state_values.squeeze()

            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            ratios = torch.exp(logprobs_new - old_logprobs.detach())

            # Dùng Advantages CỐ ĐỊNH đã tính ở trên
            surr1 = ratios.squeeze() * advantages
            surr2 = torch.clamp(
                ratios.squeeze(),
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            actor_loss = -torch.min(surr1, surr2)

            # Critic học raw returns
            critic_loss = self.MseLoss(state_values, returns_tensor)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()

            # Gradient clipping 0.5
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

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
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_returns = []

        for episode in range(1, self.max_episodes + 1):
            states, actions, log_probs, rewards = [], [], [], []

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

            for x in self.xs[1:]:
                state = self.get_state(prev_x, state_y, prev_action)

                action, log_prob = self.select_action(state)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0, self.env.height)

                prev_point = Point(prev_x, state_y)
                next_point = Point(x, next_y)
                next_shapely = ShapelyPoint(x, next_y)

                step_reward = 1.0

                step_reward -= abs(action - prev_action) * 0.05

                step_exposure = 0.0
                for sensor in self.env.sensors:
                    step_exposure += sensor.exposure_on_segment(prev_point, next_point, step=1.0,
                                                                obstacles=self.env.obstacles)

                step_reward += (step_exposure * 2.0)

                if raw_next_y <= 0 or raw_next_y >= self.env.height:
                    crashed = True

                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = True
                        break
                    else:
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0:
                            step_reward -= (4.0 - dist) * 0.5

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

            returns = []
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

            if episode % self.update_episodes == 0 or episode == self.max_episodes:
                self.update_ppo(batch_states, batch_actions, batch_log_probs, batch_returns)

                batch_states = []
                batch_actions = []
                batch_log_probs = []
                batch_returns = []

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