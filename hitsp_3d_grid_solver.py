
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import time

def hitsp_3d_grid_solver(num_points=10000, grid_size=10, max_cluster_size=100):
    start_time = time.time()

    coords_3d = np.random.rand(num_points, 3).astype(np.float32) * 100
    grid_indices_3d = (coords_3d // grid_size).astype(int)

    grid_clusters_3d = defaultdict(list)
    for i, (gx, gy, gz) in enumerate(grid_indices_3d):
        grid_clusters_3d[(gx, gy, gz)].append(i)

    final_clusters_3d = []
    for cluster in grid_clusters_3d.values():
        for i in range(0, len(cluster), max_cluster_size):
            final_clusters_3d.append(cluster[i:i + max_cluster_size])

    cluster_centers_3d = np.array([np.mean(coords_3d[cluster], axis=0) for cluster in final_clusters_3d], dtype=np.float32)

    dist_matrix = cdist(cluster_centers_3d, cluster_centers_3d)
    visited = set()
    current = 0
    order = [current]
    visited.add(current)

    while len(visited) < len(cluster_centers_3d):
        next_cid = np.argmin([dist_matrix[current][j] if j not in visited else np.inf for j in range(len(cluster_centers_3d))])
        order.append(next_cid)
        visited.add(next_cid)
        current = next_cid

    total_time = round(time.time() - start_time, 2)

    return {
        "Modus": "HiTSP mit 3D Grid-Partitionierung",
        "Punkte": num_points,
        "Cluster": len(final_clusters_3d),
        "Max. Clustergröße": max_cluster_size,
        "Reihenfolge-Auszug": order[:10],
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hitsp_3d_grid_solver()
    for k, v in result.items():
        print(f"{k}: {v}")
