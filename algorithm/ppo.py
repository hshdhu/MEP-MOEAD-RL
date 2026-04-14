import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from general.point import Point
from general.path import Path


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

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

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

        self.state_dim = 3
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

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            dist, _ = self.policy_old(state)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item()

    def update_ppo(self, states, actions, logprobs, rewards):
        returns = []
        discounted_reward = 0

        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.FloatTensor(np.array(states))
        old_actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        old_logprobs = torch.FloatTensor(np.array(logprobs)).unsqueeze(1)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            state_values = state_values.squeeze()

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

            state_y = self.env.height / 2
            prev_action = 0.0
            current_points = [Point(self.xs[0], state_y)]

            for x in self.xs[1:]:
                state = np.array([
                    x / self.env.width,
                    state_y / self.env.height,
                    prev_action
                ])

                action, log_prob = self.select_action(state)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0, self.env.height)

                step_reward = 0.0

                smoothness_penalty = abs(action - prev_action) * 0.5
                step_reward -= smoothness_penalty

                margin = self.env.height * 0.15

                if next_y < margin:
                    step_reward -= ((margin - next_y) / margin) * 1.5
                elif next_y > self.env.height - margin:
                    step_reward -= ((next_y - (self.env.height - margin)) / margin) * 1.5

                if raw_next_y < 0 or raw_next_y > self.env.height:
                    step_reward -= 2.0

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(step_reward)

                state_y = next_y
                prev_action = action
                current_points.append(Point(x, state_y))

            objs = self.evaluate_path(current_points)
            self.update_ep(current_points, objs)

            if objs[0] == float('inf'):
                terminal_reward = -50.0
            else:
                terminal_reward = (objs[0] * 1.0) - (objs[1] * 0.05)

            rewards[-1] += terminal_reward

            self.update_ppo(states, actions, log_probs, rewards)

            if verbose and episode % 10 == 0:
                total_reward = sum(rewards)
                best_exp = -objs[0] if objs[0] != float('inf') else "Crash"

                print(
                    f"Gen {episode}/{self.max_episodes} | "
                    f"Total Reward: {total_reward:.2f} | Exp: {best_exp}"
                )

            if callback:
                callback(self, episode)

    def pareto_front(self):
        return self.EP