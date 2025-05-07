
import numpy as np


class Ball:
    def __init__(self, points):
        self.points = points
        self.center = np.mean(points, axis=0)
        self.radius = max(np.linalg.norm(point - self.center) for point in points)


class BallTree:
    def __init__(self, ball):
        self.ball = ball
        self.left = None
        self.right = None

    def split(self):
        if len(self.ball.points) <= 20:
            return
        # Find the two points furthest apart
        p1 = max(self.ball.points, key=lambda p: np.linalg.norm(p - self.ball.center))
        p2 = max(self.ball.points, key=lambda p: np.linalg.norm(p - p1))
        # Partition points into two sets
        points_left = [p for p in self.ball.points if np.linalg.norm(p - p1) < np.linalg.norm(p - p2)]
        points_right = [p for p in self.ball.points if np.linalg.norm(p - p1) >= np.linalg.norm(p - p2)]
        # Create left and right children
        self.left = BallTree(Ball(points_left))
        self.right = BallTree(Ball(points_right))
        # Recursively split children
        self.left.split()
        self.right.split()


def create_ball_tree(points):
    root = BallTree(Ball(points))
    root.split()
    return root


def print_tree(tree, level=0):
    if tree is not None:
        print('  ' * level + f'Center: {tree.ball.center}, Radius: {tree.ball.radius}')
        print_tree(tree.left, level + 1)
        print_tree(tree.right, level + 1)


def test_construction():
    np.random.seed(0)
    X = np.random.random((40, 2))
    tree = create_ball_tree(X)

    print_tree(tree, level=0)


if __name__ == '__main__':
    test_construction()
