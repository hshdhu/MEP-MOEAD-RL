import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from general.point import Point
from general.path import Path
from shapely.geometry import Point as ShapelyPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
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
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

class SAC:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = 5
        self.xs = list(np.arange(0, env.width + 1, self.dx))

        self.state_dim = 13
        self.action_dim = 1
        self.action_scale = kwargs.get('action_scale', 8.0)

        self.max_episodes = kwargs.get('n_generations', 500)
        self.batch_size = kwargs.get('batch_size', 128)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.lr = kwargs.get('lr', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.target_entropy = -self.action_dim
        self.log_alpha = torch.tensor([np.log(0.2)], requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.EP = []

    @property
    def alpha(self): return self.log_alpha.exp()

    def get_safe_start_y_in_range(self, low, high, x=0.0):
        for _ in range(50):
            y = np.random.uniform(low, high)
            pt = ShapelyPoint(x, y)
            is_safe = True
            for obs in self.env.obstacles:
                if obs.polygon.contains(pt) or obs.polygon.distance(pt) < 10.0:
                    is_safe = False
                    break
            if is_safe: return y
        return self.env.height / 2

    def get_state(self, x, y, prev_action):
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

        sensor_points = [Point(x + look_ahead, y), Point(x + look_ahead, min(y + look_side, self.env.height)), Point(x + look_ahead, max(y - look_side, 0))]
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
        s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_a, next_lp = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, next_a)
            target_q = r + (1 - d) * self.gamma * (torch.min(q1_t, q2_t) - self.alpha * next_lp)

        c1, c2 = self.critic(s, a)
        critic_loss = F.mse_loss(c1, target_q) + F.mse_loss(c2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_a, lp = self.actor.sample(s)
        q1_p, q2_p = self.critic(s, new_a)
        actor_loss = (self.alpha * lp - torch.min(q1_p, q2_p)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def evaluate_path(self, points):
        path = Path(points)
        if not self.env.is_valid_path(path): return float('inf'), float('inf')
        return -path.exposure(self.env.sensors, step=1.0, obstacles=self.env.obstacles), path.length()

    def update_ep(self, points, objs):
        if objs[0] == float('inf'): return
        new_ep = []
        is_dominated = False
        for sol_p, sol_o in self.EP:
            if sol_o[0] <= objs[0] and sol_o[1] <= objs[1]:
                is_dominated = True
                new_ep.append((sol_p, sol_o))
            elif not (objs[0] <= sol_o[0] and objs[1] <= sol_o[1]):
                new_ep.append((sol_p, sol_o))
        if not is_dominated: new_ep.append((points, objs))
        self.EP = new_ep

    def run(self, verbose=True, callback=None):
        for episode in range(1, self.max_episodes + 1):
            episode_transitions = []
            sector = episode % 3
            target_y = [0.8, 0.5, 0.2][sector] * self.env.height
            state_y = self.get_safe_start_y_in_range(max(0, target_y - 5.0), min(self.env.height, target_y + 5.0), self.xs[0])

            prev_x, prev_action = self.xs[0], 0.0
            current_points = [Point(prev_x, state_y)]
            crashed, total_reward = False, 0

            for i, x in enumerate(self.xs[1:]):
                state = self.get_state(prev_x, state_y, prev_action)
                action = self.select_action(state)

                raw_next_y = state_y + action * self.action_scale
                next_y = np.clip(raw_next_y, 0, self.env.height)

                prev_point, next_point, next_shapely = Point(prev_x, state_y), Point(x, next_y), ShapelyPoint(x, next_y)

                step_reward = 1.0 - abs(action - prev_action) * 0.05
                step_exposure = sum(s.exposure_on_segment(prev_point, next_point, 1.0, self.env.obstacles) for s in self.env.sensors)
                step_reward += (step_exposure * 2.0)

                done = False
                if raw_next_y <= 0 or raw_next_y >= self.env.height: crashed = done = True

                for obs in self.env.obstacles:
                    if obs.intersects(prev_point, next_point):
                        crashed = done = True; break
                    else:
                        dist = obs.polygon.distance(next_shapely)
                        if dist < 4.0: step_reward -= (4.0 - dist) * 0.5

                next_state = self.get_state(x, next_y, action)

                if crashed:
                    step_reward += -50.0 - (50.0 * (x / self.env.width))
                    episode_transitions.append((state, action, step_reward, next_state, done))
                    current_points.append(next_point)
                    total_reward += step_reward
                    break

                if i == len(self.xs[1:]) - 1: done = True

                episode_transitions.append((state, action, step_reward, next_state, done))
                total_reward += step_reward

                state_y, prev_x, prev_action = next_y, x, action
                current_points.append(next_point)

            if not crashed:
                objs = self.evaluate_path(current_points)
                self.update_ep(current_points, objs)
                if objs[0] != float('inf'):
                    terminal_reward = 50.0 + (-objs[0] * 1.5) - (objs[1] * 0.05)
                    total_reward += terminal_reward
                    s, a, r, ns, d = episode_transitions[-1]
                    episode_transitions[-1] = (s, a, r + terminal_reward, ns, d)
            else: objs = [float('inf'), float('inf')]

            for (s, a, r, ns, d) in episode_transitions:
                self.replay_buffer.add(s, np.array([a]), r, ns, d)
                self.train()

            if verbose and episode % 100 == 0:
                print(f"Gen {episode:3d}/{self.max_episodes} | Total Rwd: {total_reward:6.1f} | Progress: {(prev_x / self.env.width) * 100:5.1f}% | Exp: {-objs[0] if objs[0] != float('inf') else 'Crash'}")

            if callback: callback(self, episode)

    def pareto_front(self): return self.EP