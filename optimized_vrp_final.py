
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def sweep_sort(points, center):
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]

def two_opt_limited_adaptive(route, base_window=5, max_successes=20):
    best = route.copy()
    best_dist = np.sum(np.linalg.norm(best[1:] - best[:-1], axis=1))
    improved = True
    successes = 0
    n = len(route)
    window = min(base_window + n // 20, n - 1)
    while improved and successes < max_successes:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 2, min(i + window, len(best))):
                new = np.concatenate((best[:i], best[i:j][::-1], best[j:]))
                new_dist = np.sum(np.linalg.norm(new[1:] - new[:-1], axis=1))
                if new_dist < best_dist:
                    best = new
                    best_dist = new_dist
                    improved = True
                    successes += 1
                    break
            if improved:
                break
    return best

def optimized_vrp_all(num_points=1000, max_quadrant_size=20, start_point=(0.5, 0.5)):
    start_time = time.time()
    coords = np.random.rand(num_points, 2).astype(np.float32)

    grid_size = int(np.ceil(np.sqrt(num_points / max_quadrant_size)))
    step = 1.0 / grid_size
    quadrants = {}

    for point in coords:
        gx, gy = int(point[0] / step), int(point[1] / step)
        quadrants.setdefault((gx, gy), []).append(point)

    quadrant_routes = []
    quadrant_centers = []

    for (gx, gy), points in quadrants.items():
        points = np.array(points, dtype=np.float32)
        center = np.array([gx * step + step / 2, gy * step + step / 2], dtype=np.float32)
        quadrant_centers.append(center)
        if len(points) > 1:
            route = sweep_sort(points, center)
            if len(route) > 3:
                route = two_opt_limited_adaptive(route)
            quadrant_routes.append(route)
        else:
            quadrant_routes.append(points)

    quadrant_centers = np.array(quadrant_centers, dtype=np.float32)
    used = np.zeros(len(quadrant_centers), dtype=bool)
    tree = KDTree(quadrant_centers)
    path_order = []

    start = np.array(start_point, dtype=np.float32)
    _, current_idx = tree.query(start)
    path_order.append(current_idx)
    used[current_idx] = True

    for _ in range(1, len(quadrant_centers)):
        dist, idx = tree.query(quadrant_centers[current_idx], k=len(quadrant_centers))
        for i in idx:
            if not used[i]:
                current_idx = i
                used[i] = True
                path_order.append(i)
                break

    full_path = []
    for idx in path_order:
        full_path.extend(quadrant_routes[idx])

    duration = round(time.time() - start_time, 2)

    if num_points <= 1000:
        plt.figure(figsize=(10, 10))
        plt.scatter(coords[:, 0], coords[:, 1], s=5, label='Punkte')
        path = np.array(full_path)
        plt.plot(path[:, 0], path[:, 1], 'r-', label='Tour')
        plt.scatter(*start_point, color='blue', s=40, label='Startpunkt')
        plt.title("Optimierter VRP-Algorithmus mit Startpunkt & adaptivem 2-opt")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "Algorithmus": "Voll optimiert (adaptives 2-opt, dynamisches Grid, Startpunkt)",
        "Punkte": num_points,
        "Quadranten": len(quadrant_routes),
        "Rechenzeit (s)": duration
    }

if __name__ == '__main__':
    result = optimized_vrp_all(num_points=1000)
    print(result)
