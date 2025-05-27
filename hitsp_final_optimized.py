
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from scipy.spatial.distance import cdist
import time

def hitsp_combined_solver(num_cities=10_000_000, num_clusters=10000, num_groups=500, max_cluster_size=100):
    start_time = time.time()

    coords = np.random.rand(num_cities, 2).astype(np.float32)

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0,
                              init='random', batch_size=10000, max_iter=50).fit(coords)
    cluster_labels = kmeans.labels_

    clusters = defaultdict(list)
    for city_idx, label in enumerate(cluster_labels):
        clusters[label].append(city_idx)

    cluster_centroids = {}
    for cid, city_list in clusters.items():
        limited_subset = city_list[:max_cluster_size] if len(city_list) > max_cluster_size else city_list
        cluster_centroids[cid] = np.mean(coords[limited_subset], axis=0)

    cluster_center_array = np.array(list(cluster_centroids.values()), dtype=np.float32)
    group_kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=1,
                                   init='random', batch_size=1000, max_iter=50).fit(cluster_center_array)
    group_labels = group_kmeans.labels_

    cluster_to_group = {cid: group_labels[i] for i, cid in enumerate(cluster_centroids)}
    groups = defaultdict(list)
    for cid, gid in cluster_to_group.items():
        groups[gid].append(cid)

    group_centroids = np.array([
        np.mean(cluster_center_array[groups[gid]], axis=0) for gid in range(num_groups)
    ], dtype=np.float32)

    dist_matrix = cdist(group_centroids, group_centroids)
    visited = set()
    current = 0
    group_order = [current]
    visited.add(current)
    while len(visited) < num_groups:
        next_gid = np.argmin([dist_matrix[current][j] if j not in visited else np.inf
                              for j in range(num_groups)])
        group_order.append(next_gid)
        visited.add(next_gid)
        current = next_gid

    final_city_count = sum(min(len(clusters[cid]), max_cluster_size)
                           for gid in group_order for cid in groups[gid])

    total_time = round(time.time() - start_time, 2)
    return {
        "Algorithmus": "HiTSP (Optimiert + Clusterlimit)",
        "Städte": num_cities,
        "Cluster": num_clusters,
        "Gruppen": num_groups,
        "Max. Clustergröße": max_cluster_size,
        "Berechnete Städte": final_city_count,
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hitsp_combined_solver()
    for k, v in result.items():
        print(f"{k}: {v}")
