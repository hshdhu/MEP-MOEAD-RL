import random
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from general.point import Point
from general.path import Path
from utils.generator import generate_random_path


# --- Helper functions ---
def path_to_ylist(path: Path) -> List[float]:
    return [p.y for p in path.points]


def ylist_to_path(xs: List[float], ys: List[float]) -> Path:
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    return Path(pts)


# --- Class MOEAD ---
class MOEAD:
    def __init__(self, env, dx=10, pop_size=50, n_generations=100,
                 neighborhood_size=10, crossover_prob=0.9, mutation_prob=None,
                 eta_c=20, eta_m=20, step_exposure=1.0, repair_attempts=50, length_max=500.0):

        self.env = env
        self.dx = dx
        self.xs = list(np.arange(0, env.width + 1, dx))
        self.gene_length = len(self.xs)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.T = min(neighborhood_size, pop_size)
        self.pc = crossover_prob
        self.pm = mutation_prob if mutation_prob is not None else 1.0 / self.gene_length
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.step_exposure = step_exposure
        self.repair_attempts = repair_attempts
        self.length_max = length_max

        # Internal storage
        self.population = []
        self.objectives = []
        self.lambdas = self.generate_lambdas()
        self.neighbors = self.find_neighbors()
        self.ideal_point = None
        self.nadir_point = None

        # External Population (EP)
        self.EP = []

        # History tracking
        self.hypervolume_history = []
        self.pareto_size_history = []
        self.pareto_front_history = []
        self.hv_ref_point = (1.0, self.length_max)

    def generate_lambdas(self) -> np.ndarray:
        lambdas = np.zeros((self.pop_size, 2))
        for i in range(self.pop_size):
            l1 = i / (self.pop_size - 1)
            lambdas[i, 0] = l1
            lambdas[i, 1] = 1.0 - l1
        return lambdas

    def find_neighbors(self) -> np.ndarray:
        nbrs = NearestNeighbors(n_neighbors=self.T, algorithm='auto').fit(self.lambdas)
        distances, indices = nbrs.kneighbors(self.lambdas)
        return indices

    def update_ideal_nadir(self, objs):
        if objs[0] == float('inf') or objs[1] == float('inf'): return
        objs_arr = np.array(objs)
        if self.ideal_point is None:
            self.ideal_point = objs_arr.copy()
            self.nadir_point = objs_arr.copy()
        else:
            self.ideal_point = np.minimum(self.ideal_point, objs_arr)
            self.nadir_point = np.maximum(self.nadir_point, objs_arr)

    def scalar_tchebycheff(self, objs, lambda_vec):
        if objs[0] == float('inf') or self.ideal_point is None:
            return float('inf')

        scale = np.ones_like(self.ideal_point)
        if self.nadir_point is not None:
            scale = self.nadir_point - self.ideal_point
            scale[scale < 1e-6] = 1.0

        diff = np.abs(np.array(objs) - self.ideal_point)
        normalized_diff = diff / scale

        return np.max(lambda_vec * normalized_diff)

    def update_external_population(self, ylist, objs):
        if objs[0] == float('inf'): return

        new_EP = []
        is_dominated = False

        for sol_y, sol_o in self.EP:
            if self.dominates(sol_o, objs):
                is_dominated = True
                new_EP.append((sol_y, sol_o))
            elif not self.dominates(objs, sol_o):
                new_EP.append((sol_y, sol_o))

        if not is_dominated:
            is_physically_duplicate = False
            curr_path_arr = np.array(ylist)
            for exist_y, _ in new_EP:
                exist_arr = np.array(exist_y)
                diff = np.mean(np.abs(curr_path_arr - exist_arr))
                if diff < 0.1:
                    is_physically_duplicate = True
                    break

            if not is_physically_duplicate:
                new_EP.append((ylist, objs))

        self.EP = new_EP

    def dominates(self, a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def evaluate_solution(self, ylist: List[float]) -> Tuple[float, float]:
        path = ylist_to_path(self.xs, ylist)
        if not self.env.is_valid_path(path):
            return (float('inf'), float('inf'))

        exp = path.exposure(self.env.sensors, step=self.step_exposure, obstacles=self.env.obstacles)
        length = path.length()
        if length > self.length_max:
            return (float('inf'), float('inf'))

        return (-exp, length)

    def sbx_crossover(self, y1, y2):
        n = len(y1)
        c1, c2 = y1.copy(), y2.copy()
        for i in range(n):
            if abs(y1[i] - y2[i]) > 1e-12:
                u = random.random()
                beta = (2 * u) ** (1 / (self.eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                c1[i] = 0.5 * ((1 + beta) * y1[i] + (1 - beta) * y2[i])
                c2[i] = 0.5 * ((1 - beta) * y1[i] + (1 + beta) * y2[i])
        c1 = [min(max(0.0, v), self.env.height) for v in c1]
        c2 = [min(max(0.0, v), self.env.height) for v in c2]
        return c1, c2

    def mutate(self, ys):
        new_ys = ys.copy()
        gene_len = len(new_ys)

        # Single-point mutation
        for i in range(gene_len):
            if random.random() <= (1.0 / gene_len):
                delta = random.uniform(-10, 10)
                new_ys[i] = np.clip(new_ys[i] + delta, 0, self.env.height)

        # Block mutation
        if random.random() < 0.3:
            block_size = random.randint(int(gene_len * 0.1), int(gene_len * 0.3))
            start_idx = random.randint(0, gene_len - block_size)
            shift = random.uniform(-30, 30)
            x_range = np.linspace(-np.pi, np.pi, block_size)
            weights = (np.cos(x_range) + 1) / 2

            for k in range(block_size):
                idx = start_idx + k
                new_ys[idx] += shift * weights[k]
                new_ys[idx] = np.clip(new_ys[idx], 0, self.env.height)
        return new_ys

    def repair_path(self, ys, max_tries_per_point=30):
        ys_copy = ys.copy()


        safe_min = 0.0
        safe_max = self.env.height

        for i, (x, y) in enumerate(zip(self.xs, ys_copy)):
            is_in_obstacle = not self.env.is_valid_point(Point(x, y))

            if is_in_obstacle:
                found = False
                search_radii = [2, 5, 10, 20, 40, 80, self.env.height]

                for r in search_radii:
                    limit_tries = 10 if r < self.env.height else max_tries_per_point
                    for _ in range(limit_tries):
                        offset = random.uniform(-r, r)
                        ny = np.clip(y + offset, safe_min, safe_max)
                        if self.env.is_valid_point(Point(x, ny)):
                            ys_copy[i] = ny
                            found = True
                            break
                    if found: break

                if not found:
                    ys_copy[i] = random.uniform(safe_min, safe_max)

        final_path = ylist_to_path(self.xs, ys_copy)
        if self.env.is_valid_path(final_path):
            return ys_copy
        return None

    def initialize_population(self):
        print(f"[MOEAD] Initializing population...")
        pop_ylists = []

        strata_centers = np.linspace(0, self.env.height, self.pop_size)
        random.shuffle(strata_centers)

        for i in range(self.pop_size):
            target_y = strata_centers[i]
            created = False

            for _ in range(50):
                # Linear Noise Strategy
                y1 = np.clip(target_y + random.uniform(-15, 15), 0, self.env.height)
                y2 = np.clip(target_y + random.uniform(-40, 40), 0, self.env.height)

                base = np.linspace(y1, y2, len(self.xs))
                noise = np.random.normal(0, 5, len(self.xs))
                candidate = np.clip(base + noise, 0, self.env.height).tolist()

                repaired = self.repair_path(candidate, max_tries_per_point=20)
                if repaired:
                    if len(pop_ylists) > 0:
                        diff = np.mean(np.abs(np.array(repaired) - np.array(pop_ylists[-1])))
                        if diff < 5.0: continue

                    pop_ylists.append(repaired)
                    created = True
                    break

            if not created:
                # Fallback: try using generate random path to get a valid path, then convert to y-list
                rand_path = generate_random_path(
                    self.env.width,
                    self.env.height,
                    dx=self.dx,
                    obstacles=self.env.obstacles
                )
                if rand_path is not None and len(rand_path.points) == len(self.xs):
                    pop_ylists.append(path_to_ylist(rand_path))
                else:
                    # Final fallback: random y-list across full height
                    random_ylist = [random.uniform(0, self.env.height) for _ in self.xs]
                    pop_ylists.append(random_ylist)

        self.population = pop_ylists
        print(f"[MOEAD] Init done. Population size: {len(self.population)}")

        self.objectives = []
        valid_cnt = 0
        for y in self.population:
            obj = self.evaluate_solution(y)
            self.objectives.append(obj)
            if obj[0] != float('inf'):
                self.update_ideal_nadir(obj)
                valid_cnt += 1
            self.update_external_population(y, obj)
        print(f"[MOEAD] Valid initial solutions: {valid_cnt}/{self.pop_size}")

    def calculate_hypervolume(self, front, ref):
        if not front: return 0.0
        ref_f1, ref_f2 = ref
        valid = [o for o in front if o[0] <= ref_f1 and o[1] <= ref_f2]
        if not valid: return 0.0
        valid.sort(key=lambda x: x[0], reverse=True)
        hv = 0.0
        prev_f1 = ref_f1
        for f1, f2 in valid:
            width = prev_f1 - f1
            height = ref_f2 - f2
            if width > 0 and height > 0: hv += width * height
            prev_f1 = f1
        return hv

    def run(self, verbose=True, callback=None):
        if not self.population:
            self.initialize_population()

        current_front = [o for y, o in self.EP]
        hv = self.calculate_hypervolume(current_front, self.hv_ref_point)
        self.hypervolume_history = [hv]
        self.pareto_size_history = [len(self.EP)]
        self.pareto_front_history = [current_front]

        for gen in range(1, self.n_generations + 1):
            for i in range(self.pop_size):
                n_idx = random.choice(self.neighbors[i])
                p1, p2 = self.population[i], self.population[n_idx]

                if random.random() < self.pc:
                    c1, _ = self.sbx_crossover(p1, p2)
                else:
                    c1 = p1.copy()

                c1 = self.mutate(c1)
                c1_repaired = self.repair_path(c1, self.repair_attempts)

                if c1_repaired is None: continue

                c_obj = self.evaluate_solution(c1_repaired)
                if c_obj[0] == float('inf'): continue

                self.update_ideal_nadir(c_obj)
                self.update_external_population(c1_repaired, c_obj)

                for j in self.neighbors[i]:
                    if self.scalar_tchebycheff(c_obj, self.lambdas[j]) < \
                            self.scalar_tchebycheff(self.objectives[j], self.lambdas[j]):
                        self.population[j] = c1_repaired
                        self.objectives[j] = c_obj

            if callback:
                callback(self, gen)

            curr_front = [o for y, o in self.EP]
            hv = self.calculate_hypervolume(curr_front, self.hv_ref_point)
            self.hypervolume_history.append(hv)
            self.pareto_size_history.append(len(self.EP))
            self.pareto_front_history.append(curr_front)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen} | EP Size: {len(self.EP)} | HV: {hv:.4f}")

    def pareto_front(self):
        return [(ylist_to_path(self.xs, y), o) for y, o in self.EP]