import math

class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"

    def distance_to(self, other: "Point") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)

    def to_tuple(self) -> tuple[float, float]:
        return self.x, self.y
