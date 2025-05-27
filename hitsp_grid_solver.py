
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import time

def hitsp_grid_solver(num_cities=500_000, grid_size=10, max_cluster_size=100):
    start_time = time.time()

    coords = np.random.rand(num_cities, 2).astype(np.float32) * 100
    grid_indices = (coords // grid_size).astype(int)

    grid_clusters = defaultdict(list)
    for i, (gx, gy) in enumerate(grid_indices):
        grid_clusters[(gx, gy)].append(i)

    final_clusters = []
    for cluster in grid_clusters.values():
        for i in range(0, len(cluster), max_cluster_size):
            final_clusters.append(cluster[i:i + max_cluster_size])

    cluster_centers = np.array([np.mean(coords[cluster], axis=0) for cluster in final_clusters], dtype=np.float32)
    dist_matrix = cdist(cluster_centers, cluster_centers)
    visited = set()
    current = 0
    order = [current]
    visited.add(current)
    while len(visited) < len(cluster_centers):
        next_cid = np.argmin([dist_matrix[current][j] if j not in visited else np.inf
                              for j in range(len(cluster_centers))])
        order.append(next_cid)
        visited.add(next_cid)
        current = next_cid

    total_time = round(time.time() - start_time, 2)

    return {
        "Modus": "HiTSP mit Grid-Partitionierung",
        "Städte": num_cities,
        "Grid-Zellen": len(grid_clusters),
        "Cluster nach Begrenzung": len(final_clusters),
        "Max. Clustergröße": max_cluster_size,
        "Reihenfolge-Auszug": order[:10],
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hitsp_grid_solver()
    for k, v in result.items():
        print(f"{k}: {v}")
