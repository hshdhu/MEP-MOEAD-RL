from math import sqrt
from shapely.geometry import LineString
from general.point import Point

class Sensor:
    def __init__(self, sensor_id, position, radius, power=1.0):
        self.id = sensor_id
        self.position = position  # Point object
        self.radius = radius
        self.power = power

    def __repr__(self):
        return f"Sensor(id={self.id}, pos={self.position}, R={self.radius})"

    def distance_to(self, point):
        return sqrt((self.position.x - point.x)**2 + (self.position.y - point.y)**2)

    def is_in_range(self, point):
        return self.distance_to(point) <= self.radius

    def is_visible(self, point, obstacles=None):
        if obstacles is None or len(obstacles) == 0:
            return True

        # Build a line segment from the sensor to the point
        line = LineString([self.position.to_tuple(), point.to_tuple()])
        
        # Check whether the segment intersects any obstacle
        for obs in obstacles:
            if line.intersects(obs.polygon):
                # If it intersects an obstacle, the point is considered not visible.
                # If the point is inside an obstacle, it is definitely not visible.
                if obs.contains(point):
                    return False
                # Even if the point is not inside, the obstacle still blocks the signal.
                return False
        
        return True

    def exposure_at(self, point, obstacles=None):
        d = self.distance_to(point)
        if d > self.radius:
            return 0.0
        
        # Line-of-sight check
        if not self.is_visible(point, obstacles):
            return 0.0
        
        return self.power / (d**2 + 1)

    def exposure_on_segment(self, p1, p2, step=1.0, obstacles=None):
        total_exp = 0.0
        length = p1.distance_to(p2)
        n = max(1, int(length / step))
        for i in range(n+1):
            t = i / n
            x = p1.x + t * (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            total_exp += self.exposure_at(Point(x, y), obstacles)
        return total_exp * (length / (n+1))  
