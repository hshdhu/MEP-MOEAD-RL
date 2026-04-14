import random
import math
import numpy as np
from shapely.geometry import Polygon, LineString
from general.path import Path
from general.point import Point
from general.sensor import Sensor
from general.obstacle import Obstacle


def generate_random_obstacles(num_obs, width, height, coverage_ratio=0.1, min_area=5,
                              radius_min=10, radius_max=30, vertices_min=4, vertices_max=8, min_distance=5):

    target_area = coverage_ratio * width * height
    obstacles = []
    current_area = 0.0
    attempts = 0
    max_attempts = 5000

    while current_area < target_area and len(obstacles) < num_obs and attempts < max_attempts:
        attempts += 1

        remaining_area = target_area - current_area
        remaining_slots = num_obs - len(obstacles)

        if remaining_slots > 0:
            current_needed_area = remaining_area / remaining_slots
            current_needed_radius = math.sqrt(current_needed_area / math.pi)
            scale = random.uniform(0.8, 1.2)
            r = current_needed_radius * scale
        else:
            break

        r = min(r, min(width, height) / 3)
        r = max(r, radius_min)

        cx = random.uniform(r, width - r)
        cy = random.uniform(r, height - r)
        num_vertices = random.randint(vertices_min, vertices_max)

        angles = np.sort(np.random.rand(num_vertices) * 2 * np.pi)
        dists = r * (0.6 + 0.4 * np.random.rand(num_vertices)) 

        points_tuples = []
        for angle, dist in zip(angles, dists):
            px = cx + dist * np.cos(angle)
            py = cy + dist * np.sin(angle)
            points_tuples.append((px, py))

        try:
            poly_shape = Polygon(points_tuples).convex_hull
        except:
            continue

        if poly_shape.area < min_area: continue

        min_x, min_y, max_x, max_y = poly_shape.bounds
        if min_x < 0 or min_y < 0 or max_x > width or max_y > height: continue

        # Collision check
        is_overlapping = False
        current_buffer = min_distance if remaining_slots > 2 else min_distance * 0.5
        new_obs_expanded = poly_shape.buffer(current_buffer)

        for existing_obs in obstacles:
            if new_obs_expanded.intersects(existing_obs.polygon):
                is_overlapping = True
                break

        if is_overlapping: continue

        obs_points = [Point(x, y) for x, y in poly_shape.exterior.coords[:-1]]
        new_obstacle = Obstacle(obs_points)
        obstacles.append(new_obstacle)
        current_area += new_obstacle.area()

        if attempts > 1000: attempts = 0

    real_ratio = (current_area / (width * height)) * 100
    print(f"   Done. Created {len(obstacles)}/{num_obs} obstacles. Coverage: {real_ratio:.2f}%")

    return obstacles

def generate_random_path(width, height, dx=10, obstacles=None, max_attempts=1000):
    xs = np.arange(0, width + 1, dx)
    path_points = []
    prev_point = None

    for x in xs:
        attempt = 0
        found_valid = False

        while attempt < max_attempts:
            attempt += 1
            y = random.uniform(0, height)
            p = Point(x, y)

            # Ensure the point is not inside any obstacle
            if obstacles and any(obs.contains(p) for obs in obstacles):
                continue

            # Ensure the connecting segment does not intersect obstacles
            if prev_point and obstacles:
                line = LineString([prev_point.to_tuple(), p.to_tuple()])
                if any(line.intersects(obs.to_shapely()) for obs in obstacles):
                    continue

            # Valid
            path_points.append(p)
            prev_point = p
            found_valid = True
            break

        if not found_valid:
            print(f"Warning: cannot create a valid point at x={x}")
            return None

    return Path(path_points)