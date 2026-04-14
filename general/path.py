from general.point import Point

class Path:
    def __init__(self, points=None):
        self.points = points if points else []

    def __repr__(self):
        return f"Path(len={len(self.points)})"

    def add_point(self, point: Point):
        self.points.append(point)

    def length(self):
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i-1].distance_to(self.points[i])
        return total

    def exposure(self, sensors, step=1.0, obstacles=None):
        total_exp = 0.0
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i-1], self.points[i]
            # Accumulate exposure from all sensors
            for sensor in sensors:
                total_exp += sensor.exposure_on_segment(p1, p2, step, obstacles)
        return total_exp

    def to_tuples(self):
        return [p.to_tuple() for p in self.points]
