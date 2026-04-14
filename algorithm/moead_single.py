import random
import numpy as np
from typing import List, Tuple
from general.point import Point
from general.path import Path


# --- Helper functions ---
def path_to_ylist(path: Path) -> List[float]:
    """Extracts y-coordinates from a Path object."""
    return [p.y for p in path.points]


def ylist_to_path(xs: List[float], ys: List[float]) -> Path:
    """Constructs a Path object from lists of x and y coordinates."""
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    return Path(pts)


# --- Class MOEAD Single Objective (Optimized for Path Planning) ---
class MOEAD_Single:
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
    adapted for Single-Objective Path Planning optimization.
    """

    def __init__(self, env, dx=10, pop_size=50, n_generations=100,
                 neighborhood_size=10, crossover_prob=0.9, mutation_prob=None,
                 eta_c=20, eta_m=20, step_exposure=1.0, length_max=2000.0,
                 nr=2):

        self.env = env
        self.dx = dx
        self.xs = list(np.arange(0, env.width + 1, dx))
        self.gene_length = len(self.xs)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.T = min(neighborhood_size, pop_size)  # Size of the neighborhood
        self.nr = nr  # Maximum number of solutions replaced by a single offspring

        self.pc = crossover_prob
        self.pm = mutation_prob if mutation_prob is not None else 1.0 / self.gene_length
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.step_exposure = step_exposure
        self.length_max = length_max

        # Population and objective value storage
        self.population = []  # List[List[float]]
        self.fitness_values = []  # Minimized value: -Exposure

        # Neighborhood topology based on index proximity (Ring topology)
        self.neighbors = self.generate_neighbors_topology()

        # Best solution tracking (Elite solution)
        self.best_solution = None  # Tuple: (ylist, fitness_value)
        self.best_history = []

    def generate_neighbors_topology(self) -> List[List[int]]:
        """Defines the neighborhood for each individual to maintain local search diversity."""
        neighbors = []
        for i in range(self.pop_size):
            # Circular indexing for ring topology: from i-T/2 to i+T/2
            indices = [k % self.pop_size for k in range(i - self.T // 2, i + self.T // 2 + 1)]
            neighbors.append(indices)
        return neighbors

    def evaluate_solution(self, ylist: List[float]) -> float:
        """
        Objective function: Returns negative Exposure for minimization.
        Handles hard constraints (path length) via penalty (float('inf')).
        """
        path = ylist_to_path(self.xs, ylist)

        # 1. Hard Constraint: Maximum Length
        length = path.length()
        if length > self.length_max:
            return float('inf')

        # 2. Hard Constraint: Boundary and Obstacles
        # Assumes the environment is valid within defined bounds
        if not self.env.is_valid_path(path):
            return float('inf')

        # 3. Exposure Calculation
        # Higher exposure is worse; we minimize (-Exposure)
        exp = path.exposure(self.env.sensors, step=self.step_exposure, obstacles=[])
        return -exp

    def sbx_crossover(self, y1, y2):
        """Simulated Binary Crossover (SBX) for continuous search space."""
        n = len(y1)
        c1, c2 = y1.copy(), y2.copy()
        for i in range(n):
            if abs(y1[i] - y2[i]) > 1e-12:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))

                c1[i] = 0.5 * ((1 + beta) * y1[i] + (1 - beta) * y2[i])
                c2[i] = 0.5 * ((1 - beta) * y1[i] + (1 + beta) * y2[i])

        # Boundary clipping
        c1 = [min(max(0.0, v), self.env.height) for v in c1]
        c2 = [min(max(0.0, v), self.env.height) for v in c2]
        return c1, c2

    def mutate(self, ys):
        """Hybrid mutation: Polynomial perturbation and block-based movement."""
        new_ys = ys.copy()
        gene_len = len(new_ys)

        # Standard Polynomial/Gaussian mutation
        for i in range(gene_len):
            if random.random() <= self.pm:
                delta = random.uniform(-10, 10)  # Stochastic perturbation
                new_ys[i] = np.clip(new_ys[i] + delta, 0, self.env.height)

        # Block Mutation: Shifts a segment of the path to improve local structures
        if random.random() < 0.2:
            block_size = random.randint(int(gene_len * 0.1), int(gene_len * 0.3))
            start_idx = random.randint(0, gene_len - block_size)
            shift = random.uniform(-20, 20)

            # Cosine-based smoothing for the block shift to maintain path continuity
            x_range = np.linspace(-np.pi, np.pi, block_size)
            weights = (np.cos(x_range) + 1) / 2

            for k in range(block_size):
                idx = start_idx + k
                new_ys[idx] += shift * weights[k]
                new_ys[idx] = np.clip(new_ys[idx], 0, self.env.height)
        return new_ys

    def repair_boundary(self, ys):
        """Ensures all coordinates remain within the environment height boundaries."""
        return [np.clip(y, 0, self.env.height) for y in ys]

    def initialize_population(self):
        """Initializes population using stratified sampling across the map height."""
        print(f"[MOEAD-Single] Initializing population...")
        self.population = []
        self.fitness_values = []

        # Stratified centers to ensure uniform coverage of the search space
        strata_centers = np.linspace(0, self.env.height, self.pop_size)
        random.shuffle(strata_centers)

        for i in range(self.pop_size):
            target_y = strata_centers[i]
            # Generate horizontal paths with Gaussian noise
            base = np.full(len(self.xs), target_y)
            noise = np.random.normal(0, 10, len(self.xs))
            candidate = np.clip(base + noise, 0, self.env.height).tolist()

            self.population.append(candidate)
            self.fitness_values.append(self.evaluate_solution(candidate))

        self.update_global_best()
        print(f"[MOEAD-Single] Init done. Best Exposure: {-self.best_solution[1] if self.best_solution else 0}")

    def update_global_best(self):
        """Tracks the overall best solution found during the evolution."""
        valid_indices = [i for i, f in enumerate(self.fitness_values) if f != float('inf')]
        if not valid_indices:
            return

        current_best_idx = min(valid_indices, key=lambda i: self.fitness_values[i])
        current_best_val = self.fitness_values[current_best_idx]

        if self.best_solution is None or current_best_val < self.best_solution[1]:
            # Use deep copy to prevent reference leakage
            self.best_solution = (list(self.population[current_best_idx]), current_best_val)

    def run(self, verbose=True):
        """Executes the MOEA/D-Single optimization loop."""
        if not self.population:
            self.initialize_population()

        for gen in range(1, self.n_generations + 1):
            # Update individuals in randomized order
            order = list(range(self.pop_size))
            random.shuffle(order)

            for i in order:
                # 1. Parent Selection from Neighborhood
                neighbors_idx = self.neighbors[i]
                k, l = random.sample(neighbors_idx, 2)
                p1, p2 = self.population[k], self.population[l]

                # 2. Reproduction Operators
                if random.random() < self.pc:
                    c, _ = self.sbx_crossover(p1, p2)
                else:
                    c = p1.copy()

                c = self.mutate(c)
                c = self.repair_boundary(c)

                # 3. Evaluation
                f_c = self.evaluate_solution(c)
                if f_c == float('inf'):
                    continue  # Discard infeasible offspring

                # 4. Replacement (Neighborhood Update)
                # Limited by self.nr to prevent premature convergence and maintain diversity
                check_list = list(neighbors_idx)
                random.shuffle(check_list)

                replaced_count = 0
                for j in check_list:
                    if replaced_count >= self.nr:
                        break

                    # Greedy replacement based on objective value
                    if f_c < self.fitness_values[j]:
                        self.population[j] = c[:]
                        self.fitness_values[j] = f_c
                        replaced_count += 1

            self.update_global_best()
            best_exp = -self.best_solution[1] if self.best_solution else 0
            self.best_history.append(best_exp)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen} | Best Exposure: {best_exp:.4f}")

    def get_best_path(self):
        """Returns the optimized path as a Path object."""
        if self.best_solution:
            return ylist_to_path(self.xs, self.best_solution[0])
        return None