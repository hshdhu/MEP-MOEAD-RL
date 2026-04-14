from shapely.geometry import Point as ShapelyPoint, Polygon, LineString
from general.point import Point

class Obstacle:
    def __init__(self, vertices):
        self.vertices = vertices
        self.polygon = Polygon([p.to_tuple() for p in vertices])

    def __repr__(self):
        return f"Obstacle(num_vertices={len(self.vertices)}, area={self.area():.2f})"

    def area(self):
        return self.polygon.area

    def contains(self, point: Point):
        return self.polygon.contains(ShapelyPoint(point.x, point.y))

    def to_shapely(self):
        return self.polygon

    def intersects(self, p1: Point, p2: Point):
        line = LineString([p1.to_tuple(), p2.to_tuple()])
        return self.polygon.intersects(line)

    def to_tuples(self):
        return [p.to_tuple() for p in self.vertices]
