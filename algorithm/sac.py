import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from typing import List, Tuple

from general.point import Point
from general.path import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helpers ---
def ylist_to_path(xs: List[float], ys: List[float]) -> Path:
    return Path([Point(x, y) for x, y in zip(xs, ys)])


# --- Neural Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_scale):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

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
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        return action * self.action_scale, log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + 1, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + 1, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa));
        q1 = F.relu(self.l2(q1));
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa));
        q2 = F.relu(self.l5(q2));
        q2 = self.l6(q2)
        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)).to(device),
                torch.FloatTensor(np.array(a)).to(device),
                torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(s2)).to(device),
                torch.FloatTensor(np.array(d)).unsqueeze(1).to(device))

    def __len__(self): return len(self.buffer)


class PathEnv:
    def __init__(self, env, xs, action_scale, step_exposure, length_max, w1=0.8, w2=0.2, window_size=5):
        self.env = env
        self.xs = xs
        self.n = len(xs)
        self.action_scale = action_scale
        self.step_exp = step_exposure
        self.length_max = length_max
        self.w1, self.w2 = w1, w2  # w1 ưu tiên Exposure, w2 cho Length
        self.window_size = window_size
        self.state_dim = 2 + window_size
        self._ys = []
        self._step = 0

    def reset(self, start_y=None):
        if start_y is None:
            start_y = random.uniform(self.env.height * 0.2, self.env.height * 0.8)
        self._ys = [start_y]
        self._step = 0
        return self._make_state()

    def _make_state(self):
        norm_x = self._step / max(self.n - 1, 1)
        norm_y = self._ys[-1] / self.env.height if self.env.height > 0 else 0.0
        pad = self._ys[0]
        history = (self._ys[-self.window_size:] if len(self._ys) >= self.window_size
                   else [pad] * (self.window_size - len(self._ys)) + self._ys)
        norm_h = [v / self.env.height for v in history]
        return np.array([norm_x, norm_y] + norm_h, dtype=np.float32)

    def step(self, action):
        current_y = self._ys[-1]
        next_y = float(np.clip(current_y + action, 0, self.env.height))
        x_next = self.xs[self._step + 1] if self._step + 1 < self.n else self.xs[-1]

        current_point = Point(x_next, next_y)
        collision = not self.env.is_valid_point(current_point)

        total_reward = 0.0

        # 1. Phạt va chạm vật cản
        if collision:
            total_reward -= 2.0
        else:
            # 2. Thưởng Exposure tức thời (nhử Agent vào vùng đậm)
            local_exp = 0.0
            prev_p = Point(self.xs[self._step], self._ys[-1])
            curr_p = current_point

            for s in self.env.sensors:
                local_exp += s.exposure_on_segment(prev_p, curr_p, self.step_exp, self.env.obstacles)

            total_reward += (local_exp * 1.0)

        # 3. Phạt bám biên (Đẩy Agent ra khỏi mép 0 và 200)
        if next_y > self.env.height - 10 or next_y < 10:
            total_reward -= 1.0

        self._ys.append(next_y)
        self._step += 1
        done = (self._step >= self.n - 1)

        if done:
            path = ylist_to_path(self.xs, self._ys)
            if self.env.is_valid_path(path):
                # Tính tổng Exposure thực tế của cả đường
                final_exp = path.exposure(self.env.sensors, step=self.step_exp, obstacles=self.env.obstacles)

                # Reward cuối: Ưu tiên cực cao cho Exposure (nhân hệ số lớn)
                # Trừ đi phạt độ dài nhưng không để nó lấn át Exposure
                exp_reward = self.w1 * final_exp * 5.0
                len_penalty = self.w2 * (path.length() / self.length_max) * 10.0

                total_reward += (exp_reward - len_penalty + 50.0)
            else:
                total_reward -= 50.0  # Phạt cực nặng nếu không về đích hợp lệ

        return self._make_state(), total_reward, done

    def get_ylist(self):
        return self._ys.copy()


# --- SAC Algorithm Class ---
class SAC:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dx = kwargs.get('dx', 5)
        self.xs = list(np.arange(0, env.width + 1, self.dx))
        self.pop_size = kwargs.get('pop_size', 40)
        self.n_generations = kwargs.get('n_generations', 100)
        self.length_max = kwargs.get('length_max', 2000.0)
        self.batch_size = kwargs.get('batch_size', 128)
        self.updates_per_ep = kwargs.get('updates_per_ep', 5)

        self.path_env = PathEnv(env, self.xs, kwargs.get('action_scale', 15.0),
                                kwargs.get('step_exposure', 1.0), self.length_max,
                                kwargs.get('w1', 0.7), kwargs.get('w2', 0.3), kwargs.get('window_size', 5))

        self.agent = SACAgent(self.path_env.state_dim, kwargs.get('action_scale', 15.0),
                              hidden=kwargs.get('hidden_size', 256), lr=kwargs.get('lr', 5e-4))

        self.buffer = ReplayBuffer(100000)
        self.EP = []  # External Population (Pareto Front)
        self.population = []
        self.hv_ref_point = (1.0, self.length_max)
        self.hypervolume_history, self.pareto_size_history, self.pareto_front_history = [], [], []

    def evaluate_solution(self, ylist):
        path = ylist_to_path(self.xs, ylist)
        if not self.env.is_valid_path(path): return (float('inf'), float('inf'))
        exp = path.exposure(self.env.sensors, step=self.path_env.step_exp, obstacles=self.env.obstacles)
        length = path.length()
        return (-exp, length) if length <= self.length_max else (float('inf'), float('inf'))

    def update_external_population(self, ylist, objs):
        if objs[0] == float('inf'): return
        new_EP = []
        is_dominated = False
        for sol_y, sol_o in self.EP:
            if sol_o[0] <= objs[0] and sol_o[1] <= objs[1] and (sol_o[0] < objs[0] or sol_o[1] < objs[1]):
                is_dominated = True
                new_EP.append((sol_y, sol_o))
            elif not (objs[0] <= sol_o[0] and objs[1] <= sol_o[1]):
                new_EP.append((sol_y, sol_o))
        if not is_dominated:
            if not any(np.allclose(ylist, ey, atol=0.1) for ey, _ in new_EP):
                new_EP.append((ylist, objs))
        self.EP = new_EP

    def calculate_hypervolume(self, front, ref):
        if not front: return 0.0
        valid = sorted([o for o in front if o[0] <= ref[0] and o[1] <= ref[1]], key=lambda x: x[0], reverse=True)
        hv, prev = 0.0, ref[0]
        for f1, f2 in valid:
            hv += (prev - f1) * (ref[1] - f2)
            prev = f1
        return hv

    def _run_episode(self, deterministic=False):
        state = self.path_env.reset()
        done = False
        while not done:
            action = self.agent.select_action(state, deterministic)
            next_state, reward, done = self.path_env.step(action)
            self.buffer.push(state, [action], reward, next_state, done)
            state = next_state
        return self.path_env.get_ylist()

    def initialize_population(self):
        print("[SAC-Torch] Warming up buffer & Initial training...")
        for _ in range(self.pop_size):
            ylist = self._run_episode(False)
            obj = self.evaluate_solution(ylist)
            self.update_external_population(ylist, obj)
            self.population.append(ylist)
        for _ in range(500):
            self.agent.update(self.buffer, self.batch_size)

    def run(self, verbose=True, callback=None):
        if not self.population: self.initialize_population()
        for gen in range(1, self.n_generations + 1):
            for _ in range(self.pop_size):
                ylist = self._run_episode(False)
                obj = self.evaluate_solution(ylist)
                self.update_external_population(ylist, obj)

            for _ in range(self.updates_per_ep * self.pop_size):
                self.agent.update(self.buffer, self.batch_size)

            if callback: callback(self, gen)

            curr_front = [o for _, o in self.EP]
            hv = self.calculate_hypervolume(curr_front, self.hv_ref_point)
            self.hypervolume_history.append(hv)
            self.pareto_size_history.append(len(self.EP))

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:4d} | EP: {len(self.EP):3d} | HV: {hv:.2f} | Alpha: {self.agent.alpha.item():.4f}")

    def pareto_front(self) -> List[Tuple[Path, Tuple[float, float]]]:
        return [(ylist_to_path(self.xs, y), o) for y, o in self.EP]


class SACAgent:
    def __init__(self, state_dim, action_scale, hidden=256, lr=5e-4, gamma=0.98, tau=0.005):
        self.gamma, self.tau, self.action_scale = gamma, tau, action_scale
        self.actor = Actor(state_dim, hidden, action_scale).to(device)
        self.critic = Critic(state_dim, hidden).to(device)
        self.critic_target = Critic(state_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -3.0
        self.log_alpha = torch.tensor([np.log(0.5)], requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if deterministic:
            mu, _ = self.actor(state)
            return torch.tanh(mu).item() * self.action_scale
        action, _ = self.actor.sample(state)
        return action.item()

    def update(self, buffer, batch_size=64):
        if len(buffer) < batch_size: return
        s, a, r, s2, d = buffer.sample(batch_size)

        with torch.no_grad():
            next_a, next_lp = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, next_a)
            target_q = r + (1 - d) * self.gamma * (torch.min(q1_t, q2_t) - self.alpha * next_lp)

        c1, c2 = self.critic(s, a)
        critic_loss = F.mse_loss(c1, target_q) + F.mse_loss(c2, target_q)
        self.critic_opt.zero_grad();
        critic_loss.backward();
        self.critic_opt.step()

        new_a, lp = self.actor.sample(s)
        q1_p, q2_p = self.critic(s, new_a)
        actor_loss = (self.alpha * lp - torch.min(q1_p, q2_p)).mean()
        self.actor_opt.zero_grad();
        actor_loss.backward();
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad();
        alpha_loss.backward();
        self.alpha_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)