import random
import yaml
from pathlib import Path
from typing import Dict, Any
from general.environment import Environment
from general.sensor import Sensor
from general.obstacle import Obstacle
from general.point import Point
from utils.generator import generate_random_obstacles


class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_environment(self, load_from_file: str = None) -> Environment:
        if load_from_file:
            env, _ = Environment.load(load_from_file)
            return env

        env_config = self.config['environment']
        env = Environment(
            width=env_config['width'],
            height=env_config['height'],
            obstacles=[],
            sensors=[]
        )

        # Generate obstacles first
        obstacle_config = self.config.get('obstacle', {})
        env.obstacles = generate_random_obstacles(
            num_obs=env_config['num_obstacles'],
            width=env_config['width'],
            height=env_config['height'],
            coverage_ratio=env_config['coverage_ratio'],
            min_area=obstacle_config.get('min_area', 5),
            radius_min=obstacle_config.get('radius_min', 10),
            radius_max=obstacle_config.get('radius_max', 30),
            vertices_min=obstacle_config.get('vertices_min', 4),
            vertices_max=obstacle_config.get('vertices_max', 8),
            min_distance=obstacle_config.get('min_distance', 5)
        )

        # Generate sensors
        sensor_config = self.config.get('sensor', {})
        radius_min = sensor_config.get('radius_min', 5)
        radius_max = sensor_config.get('radius_max', 40)
        power = sensor_config.get('power', 1.0)

        sensors = []
        max_attempts = 1000  # Maximum attempts per sensor
        for i in range(env_config['num_sensors']):
            attempts = 0
            placed = False
            while attempts < max_attempts and not placed:
                attempts += 1
                x = random.uniform(0, env_config['width'])
                y = random.uniform(0, env_config['height'])
                r = random.uniform(radius_min, radius_max)
                sensor_pos = Point(x, y)
                
                # Check if sensor position is outside all obstacles
                is_valid = True
                for obs in env.obstacles:
                    if obs.contains(sensor_pos):
                        is_valid = False
                        break
                
                if is_valid:
                    sensors.append(Sensor(i, sensor_pos, r, power=power))
                    placed = True
            
            if not placed:
                print(f"Warning: could not place sensor {i} after {max_attempts} attempts")

        env.sensors = sensors

        return env

    def get_moead_params(self) -> Dict[str, Any]:
        path_config = self.config['path']
        exposure_config = self.config.get('exposure', {})
        repair_config = self.config.get('repair', {})

        try:
            alg_config = self.config['algorithm']['moead']
        except KeyError:
            raise KeyError("Config file error: 'algorithm.moead' section not found.")

        return {
            'dx': path_config['dx'],
            'length_max': path_config.get('length_max', 500.0),
            'step_exposure': exposure_config.get('step', 1.0),
            'repair_attempts': repair_config.get('attempts', 50),

            'pop_size': alg_config.get('pop_size', 50),
            'n_generations': alg_config.get('n_generations', 100),
            'neighborhood_size': alg_config.get('neighborhood_size', 10), 
            'crossover_prob': alg_config.get('crossover_prob', 0.9),
            'mutation_prob': alg_config.get('mutation_prob', 0.2),
            'eta_c': alg_config.get('eta_c', 20),
            'eta_m': alg_config.get('eta_m', 20),
        }

    def get_config_value(self, key_path: str):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Config key not found: {key_path}")
        return value

    def save_environment(self, env: Environment, folder: str = "data", filename: str = None):
        if filename is None:
            env_config = self.config['environment']
            filename = f"env_{env_config['width']}x{env_config['height']}_" \
                      f"s{env_config['num_sensors']}_o{env_config['num_obstacles']}.json"

        env.save(folder, filename, path=None)

    def print_config(self):
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)

        # General info
        env_config = self.config.get('environment', {})
        print(f"Environment: {env_config.get('width', 0)}x{env_config.get('height', 0)}")
        print(f"  - Sensors: {env_config.get('num_sensors', 0)}")
        print(f"  - Obstacles: {env_config.get('num_obstacles', 0)}")
        print(f"  - Coverage: {env_config.get('coverage_ratio', 0)*100:.1f}%")

        path_config = self.config.get('path', {})
        print(f"\nPath dx: {path_config.get('dx', 'N/A')}")

        # MOEA/D info
        try:
            moead_conf = self.config['algorithm']['moead']
            print("\n[MOEA/D]")
            print(f"  Population: {moead_conf.get('pop_size', 0)}")
            print(f"  Generations: {moead_conf.get('n_generations', 0)}")
            print(f"  Neighborhood: {moead_conf.get('neighborhood_size', 0)}")
        except KeyError:
            print("\n[MOEA/D] section not found in config.")

        print("=" * 50)


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    return ConfigLoader(config_path)