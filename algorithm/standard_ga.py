import random
import numpy as np
from typing import List, Tuple
from general.point import Point
from general.path import Path


# --- Helper functions (Shared with MOEAD) ---
def path_to_ylist(path: Path) -> List[float]:
    """Extracts y-coordinates from a Path object for genotype representation."""
    return [p.y for p in path.points]


def ylist_to_path(xs: List[float], ys: List[float]) -> Path:
    """Reconstructs a Path object from x and y coordinate lists."""
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    return Path(pts)


# --- Class Standard Genetic Algorithm (GA) ---
class StandardGA:
    """
    Implementation of a Standard Genetic Algorithm for Path Planning optimization.
    Features Tournament Selection and Elitism to maintain selection pressure.
    """

    def __init__(self, env, dx=10, pop_size=50, n_generations=100,
                 crossover_prob=0.9, mutation_prob=None,
                 eta_c=20, eta_m=20, step_exposure=1.0, length_max=2000.0,
                 tournament_size=3, n_elites=2):

        self.env = env
        self.dx = dx
        self.xs = list(np.arange(0, env.width + 1, dx))
        self.gene_length = len(self.xs)
        self.pop_size = pop_size
        self.n_generations = n_generations

        # GA-specific hyperparameters
        self.tournament_size = tournament_size
        self.n_elites = n_elites  # Number of top individuals preserved across generations

        # Evolutionary operators parameters
        self.pc = crossover_prob
        self.pm = mutation_prob if mutation_prob is not None else 1.0 / self.gene_length
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.step_exposure = step_exposure
        self.length_max = length_max

        # Population and fitness tracking
        self.population = []
        self.fitness_values = []

        self.best_solution = None  # Elite solution (ylist, fitness)
        self.best_history = []

    def evaluate_solution(self, ylist: List[float]) -> float:
        """
        Objective function evaluation.
        Maintains logic consistency with MOEAD_Single for fair benchmarking.
        Returns negative Exposure (minimization objective).
        """
        path = ylist_to_path(self.xs, ylist)

        # Constraint handling: Path length limit
        length = path.length()
        if length > self.length_max:
            return float('inf')

        # Constraint handling: Boundary and obstacle validity
        if not self.env.is_valid_path(path):
            return float('inf')

        # Calculate exposure (assuming obstacles are handled by environment validity)
        exp = path.exposure(self.env.sensors, step=self.step_exposure, obstacles=[])
        return -exp

    def sbx_crossover(self, y1, y2):
        """Simulated Binary Crossover (SBX) operator."""
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

        c1 = [min(max(0.0, v), self.env.height) for v in c1]
        c2 = [min(max(0.0, v), self.env.height) for v in c2]
        return c1, c2

    def mutate(self, ys):
        """Mutation operator including point-wise perturbation and smooth block shifts."""
        new_ys = ys.copy()
        gene_len = len(new_ys)

        for i in range(gene_len):
            if random.random() <= self.pm:
                delta = random.uniform(-10, 10)
                new_ys[i] = np.clip(new_ys[i] + delta, 0, self.env.height)

        # Structural Block Mutation for spatial continuity
        if random.random() < 0.2:
            block_size = random.randint(int(gene_len * 0.1), int(gene_len * 0.3))
            start_idx = random.randint(0, gene_len - block_size)
            shift = random.uniform(-20, 20)

            x_range = np.linspace(-np.pi, np.pi, block_size)
            weights = (np.cos(x_range) + 1) / 2
            for k in range(block_size):
                idx = start_idx + k
                new_ys[idx] += shift * weights[k]
                new_ys[idx] = np.clip(new_ys[idx], 0, self.env.height)
        return new_ys

    def repair_boundary(self, ys):
        """Clips coordinates to environment bounds."""
        return [np.clip(y, 0, self.env.height) for y in ys]

    def tournament_selection(self):
        """Selects the best individual among a random subset of the population."""
        candidates_idx = random.sample(range(self.pop_size), self.tournament_size)

        # Minimize objective value (Negative Exposure)
        best_idx = min(candidates_idx, key=lambda idx: self.fitness_values[idx])
        return self.population[best_idx]

    def initialize_population(self):
        """Initializes population using stratified horizontal paths with noise."""
        print(f"[Standard GA] Initializing population...")
        self.population = []
        self.fitness_values = []

        strata_centers = np.linspace(0, self.env.height, self.pop_size)
        random.shuffle(strata_centers)

        for i in range(self.pop_size):
            target_y = strata_centers[i]
            base = np.full(len(self.xs), target_y)
            noise = np.random.normal(0, 10, len(self.xs))
            candidate = np.clip(base + noise, 0, self.env.height).tolist()

            self.population.append(candidate)
            self.fitness_values.append(self.evaluate_solution(candidate))

        self.update_global_best()
        print(f"[Standard GA] Init done. Best Exposure: {-self.best_solution[1] if self.best_solution else 0}")

    def update_global_best(self):
        """Identifies and stores the best feasible individual found so far."""
        valid_indices = [i for i, f in enumerate(self.fitness_values) if f != float('inf')]
        if not valid_indices: return

        current_best_idx = min(valid_indices, key=lambda i: self.fitness_values[i])
        current_best_val = self.fitness_values[current_best_idx]

        if self.best_solution is None or current_best_val < self.best_solution[1]:
            self.best_solution = (list(self.population[current_best_idx]), current_best_val)

    def run(self, verbose=True):
        """Executes the Genetic Algorithm generational loop."""
        if not self.population:
            self.initialize_population()

        for gen in range(1, self.n_generations + 1):
            new_population = []
            new_fitness = []

            # 1. Elitism: Preserve top-tier feasible individuals
            sorted_indices = np.argsort(self.fitness_values)
            valid_sorted = [i for i in sorted_indices if self.fitness_values[i] != float('inf')]

            elites_idx = valid_sorted[:self.n_elites]
            for idx in elites_idx:
                new_population.append(self.population[idx][:])
                new_fitness.append(self.fitness_values[idx])

            # 2. Reproduction: Selection, Crossover, and Mutation
            while len(new_population) < self.pop_size:
                # Parent Selection
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                # Crossover
                if random.random() < self.pc:
                    offspring = self.sbx_crossover(p1, p2)
                else:
                    offspring = (p1.copy(), p2.copy())

                for child in offspring:
                    if len(new_population) >= self.pop_size:
                        break

                    # Mutation and Repair
                    child = self.mutate(child)
                    child = self.repair_boundary(child)

                    # Feasibility check for population inclusion
                    f_child = self.evaluate_solution(child)
                    if f_child != float('inf'):
                        new_population.append(child)
                        new_fitness.append(f_child)

            # 3. Generational Replacement
            self.population = new_population
            self.fitness_values = new_fitness

            # Record metrics
            self.update_global_best()
            best_exp = -self.best_solution[1] if self.best_solution else 0
            self.best_history.append(best_exp)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen} [GA] | Best Exposure: {best_exp:.4f}")

    def get_best_path(self):
        """Returns the optimal path found as a Path object."""
        if self.best_solution:
            return ylist_to_path(self.xs, self.best_solution[0])
        return None