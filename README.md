# Path Planning Optimization: MOEA/D & Genetic Algorithms

This project implements evolutionary algorithms to solve path planning problems in environments with sensors and obstacles. The system supports two main operating modes:

**Multi-Objective Optimization (MOEA/D):** Finds a set of Pareto-optimal paths that balance between path length (shortest) and exposure level (safe/low detection). Typically used for environments with obstacles.

**Single-Objective Benchmark:** Compares performance between MOEA/D (Single) and Standard GA for the Maximal Exposure problem (e.g., patrol or data collection tasks), typically used in obstacle-free environments.

## 📂 Project Structure

```
├── algorithm/
│   ├── moead.py           # Original MOEA/D algorithm (Multi-objective)
│   ├── moead_single.py    # Single-objective MOEA/D variant
│   └── standard_ga.py     # Standard Genetic Algorithm (GA) for comparison
├── utils/
│   ├── config_loader.py   # YAML configuration file reader
│   ├── draw.py            # Drawing utilities (Map, Pareto Front, Convergence)
│   └── generator.py       # Environment and random path generator
├── data/                  # Directory containing generated environment files (JSON)
├── result/                # Multi-objective optimization results (from run.py)
├── result_benchmark/      # Benchmark comparison results (from run_benchmark.py)
├── config.yaml            # Central configuration file
├── generate_env.py        # Script to generate new environments
├── run.py                 # Script to run Multi-objective optimization
└── run_benchmark.py       # Script to run Single-objective benchmark comparison
```

## ⚙️ Configuration (config.yaml)

All parameters for environment, robot, and algorithms are managed in `config.yaml`.

### Key Settings:

**Environment:**
- `num_sensors`: Number of sensors.
- `num_obstacles`: Set to `0` for Benchmark mode, set `>0` for Multi-objective mode (obstacle avoidance).

**Path:** `dx` adjusts the step size granularity.

**Algorithm:** Adjust `pop_size` (population size), `n_generations` (number of generations), etc.

```yaml
environment:
  num_sensors: 50
  num_obstacles: 0  # Set = 0 for benchmark, > 0 for obstacle avoidance
  # ...

path:
  dx: 5
  length_max: 2000.0

algorithm:
  moead:
    pop_size: 100
    n_generations: 100
    # ...
```

## 📦 Installation Requirements

This project requires Python 3.8+ and the following libraries:

```bash
pip install numpy matplotlib pyyaml shapely
```

## 🚀 Usage Guide

### Step 1: Generate Environment (Map)

Before running any algorithm, you need to create a map file (JSON).

```bash
python generate_env.py
```

**Output:** JSON file will be saved to `data/{num_sensors} sensors/env_{timestamp}.json`.

**Note:** Directories will be automatically organized based on the number of sensors in config (e.g., 50 sensors, 100 sensors).

### Step 2: Run Algorithm (Choose 1 of 2 modes)

#### Mode A: Multi-Objective Optimization (run.py)

Use this mode when you want to find obstacle-avoiding paths that balance between distance and safety.

1. Ensure `num_obstacles > 0` in `config.yaml` (and regenerate env if needed).
2. Run command:

```bash
python run.py "data/50 sensors/env_20260124_120000.json"
```

**Results (result/):**
- `pareto_fronts.png`: Trade-off visualization between path length and exposure.
- `solutions_map.png`: Visual representation of best paths on the map.
- `pareto_solutions.json`: Detailed solution data.

#### Mode B: Single-Objective Benchmark (run_benchmark.py)

Use this mode to compare the performance between MOEA/D and GA. The problem is typically Maximal Exposure (finding paths through maximum coverage areas) in open environments.

1. Ensure `num_obstacles: 0` in `config.yaml` (and regenerate new env).
2. Run command:

```bash
python run_benchmark.py "data/50 sensors/env_20260124_120000.json"
```

**Results (result_benchmark/):**
- `convergence_comparison.png`: Convergence speed comparison chart (higher is better).
- `paths_comparison.png`: Visual comparison of best paths from both algorithms on the map.
- `summary_report.txt`: Summary report of runtime and scores.

## 📊 Results Interpretation

**Multi-Objective:** Find the "Knee Solution" (knee point) on the Pareto chart. This is the best balanced solution where exposure level is low while the path is not excessively long.

**Benchmark:**
- **Higher is Better** chart: The goal is to maximize Exposure.
- The curve that rises higher and faster indicates a more effective algorithm in exploring the solution space.