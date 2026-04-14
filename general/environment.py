from pathlib import Path as FilePath
import math
import json
import os
from general.point import Point
from general.path import Path
from general.obstacle import Obstacle
from general.sensor import Sensor

class Environment:
    def __init__(self, width=200, height=200, obstacles=None, sensors=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles else []
        self.sensors = sensors if sensors else []

    def __repr__(self):
        return f"Environment(size=({self.width}, {self.height}), " \
               f"sensors={len(self.sensors)}, obstacles={len(self.obstacles)})"

    def add_sensor(self, sensor, min_sensor_distance = 10):
        if not (0 <= sensor.position.x <= self.width and 0 <= sensor.position.y <= self.height):
            return False

        for s in self.sensors:
            d = math.dist(sensor.position.to_tuple(), s.position.to_tuple())
            if d < min_sensor_distance:
                return False

        self.sensors.append(sensor)
        return True

    def add_obstacle(self, obstacle: Obstacle, min_distance=2.0):
        new_poly = obstacle.polygon

        for existing in self.obstacles:
            existing_poly = existing.polygon
            if new_poly.intersects(existing_poly):
                # Intersects or touches the boundary
                return False
            if new_poly.distance(existing_poly) < min_distance:
                # Distance between two obstacles is too small
                return False

        self.obstacles.append(obstacle)
        return True

    def is_valid_point(self, point: Point):
        if not (0 <= point.x <= self.width and 0 <= point.y <= self.height):
            return False
        for obs in self.obstacles:
            if obs.contains(point):
                return False
        return True

    def is_valid_path(self, path: Path):
        for i in range(1, len(path.points)):
            p1, p2 = path.points[i-1], path.points[i]
            if not (self.is_valid_point(p1) and self.is_valid_point(p2)):
                return False
            for obs in self.obstacles:
                if obs.intersects(p1, p2):
                    return False
        return True

    def to_dict(self, path: Path | None = None):
        return {
            "width": self.width,
            "height": self.height,
            "sensors": [
                {
                    "id": s.id,
                    "x": s.position.x,
                    "y": s.position.y,
                    "radius": s.radius,
                    "power": s.power
                } for s in self.sensors
            ],
            "obstacles": [
                [[p.x, p.y] for p in obs.vertices] for obs in self.obstacles
            ],
            "path": [[p.x, p.y] for p in path.points] if path else None
        }

    @classmethod
    def from_dict(cls, data):
        obstacles = [Obstacle([Point(x, y) for x, y in verts]) for verts in data.get("obstacles", [])]
        sensors = [
            Sensor(s["id"], Point(s["x"], s["y"]), s["radius"], s["power"])
            for s in data.get("sensors", [])
        ]
        env = cls(data["width"], data["height"], obstacles, sensors)

        path_data = data.get("path")
        path = None
        if path_data:
            path = Path([Point(x, y) for x, y in path_data])

        return env, path

    def save(self, folder: str, filename: str, path: Path | None = None):
        save_dir = FilePath(folder)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        data = self.to_dict(path)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Environment saved to {save_path}")

    @classmethod
    def load(cls, filepath: str):
        with open(FilePath(filepath), "r", encoding="utf-8") as f:
            data = json.load(f)

        env, path = cls.from_dict(data)
        print(f"Environment loaded from {filepath}")
        return env, path