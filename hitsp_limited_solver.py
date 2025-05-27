
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import time

def hitsp_limited_solver(num_cities=1_000_000, num_clusters=5000, num_groups=200, max_cluster_size=20):
    start_time = time.time()

    coords = np.random.rand(num_cities, 2) * 100
    cities = {i: tuple(coords[i]) for i in range(num_cities)}

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=5000).fit(coords)
    cluster_labels = kmeans.labels_

    clusters = defaultdict(list)
    for city_idx, label in enumerate(cluster_labels):
        clusters[label].append(city_idx)

    cluster_centroids = {}
    for cid, city_list in clusters.items():
        limited_subset = city_list[:max_cluster_size] if len(city_list) > max_cluster_size else city_list
        cluster_centroids[cid] = tuple(np.mean([cities[i] for i in limited_subset], axis=0))

    cluster_center_array = np.array(list(cluster_centroids.values()))
    group_kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=1, batch_size=500).fit(cluster_center_array)
    group_labels = group_kmeans.labels_

    cluster_to_group = {cid: group_labels[i] for i, cid in enumerate(cluster_centroids)}
    groups = defaultdict(list)
    for cid, gid in cluster_to_group.items():
        groups[gid].append(cid)

    def solve_meta_nn(centers):
        remaining = set(centers.keys())
        current = list(remaining)[0]
        route = [current]
        remaining.remove(current)
        while remaining:
            next_node = min(remaining, key=lambda x: np.linalg.norm(np.array(centers[current]) - np.array(centers[x])))
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        return route

    group_routes = {}
    group_centroids = {}

    for group_id, cluster_ids in groups.items():
        route = solve_meta_nn({cid: cluster_centroids[cid] for cid in cluster_ids})
        group_routes[group_id] = route
        group_center = np.mean([cluster_centroids[cid] for cid in cluster_ids], axis=0)
        group_centroids[group_id] = tuple(group_center)

    final_group_order = solve_meta_nn(group_centroids)

    final_city_count = sum(min(len(clusters[cid]), max_cluster_size)
                           for gid in final_group_order for cid in group_routes[gid])

    total_time = round(time.time() - start_time, 2)
    return {
        "Algorithmus": "HiTSP mit Clustergrößenlimit",
        "Städte": num_cities,
        "Max. Clustergröße": max_cluster_size,
        "Berechnete Städte": final_city_count,
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hitsp_limited_solver()
    for k, v in result.items():
        print(f"{k}: {v}")
