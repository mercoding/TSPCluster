
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import time

def scalable_hierarchical_tsp_runtime(num_cities=100000, num_clusters=1000, num_groups=100):
    start_time = time.time()

    coords = np.random.rand(num_cities, 2) * 100
    cities = {i: tuple(coords[i]) for i in range(num_cities)}

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000).fit(coords)
    cluster_labels = kmeans.labels_

    clusters = defaultdict(list)
    for city_idx, label in enumerate(cluster_labels):
        clusters[label].append(city_idx)

    cluster_centers = np.array([np.mean([cities[i] for i in clusters[cid]], axis=0) for cid in clusters])
    group_kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=1, batch_size=100).fit(cluster_centers)
    group_labels = group_kmeans.labels_

    cluster_to_group = {cid: group_labels[i] for i, cid in enumerate(clusters)}
    groups = defaultdict(list)
    for cid, gid in cluster_to_group.items():
        groups[gid].append(cid)

    def solve_tsp_nn_limited(subset):
        if len(subset) > 200:
            return subset
        unvisited = set(subset)
        current = subset[0]
        route = [current]
        unvisited.remove(current)
        while unvisited:
            next_city = min(unvisited, key=lambda city: np.linalg.norm(np.array(cities[current]) - np.array(cities[city])))
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return route

    cluster_routes = {}
    cluster_centroids = {}

    for cluster_id, city_list in clusters.items():
        route = solve_tsp_nn_limited(city_list)
        cluster_routes[cluster_id] = route
        center = np.mean([coords[i] for i in route], axis=0)
        cluster_centroids[cluster_id] = tuple(center)

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

    final_route_length = sum(len(cluster_routes[cid]) for gid in final_group_order for cid in group_routes[gid])
    total_time = round(time.time() - start_time, 2)

    return {
        "Städte": num_cities,
        "Cluster": num_clusters,
        "Gruppen": num_groups,
        "Berechnete Städte": final_route_length,
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = scalable_hierarchical_tsp_runtime()
    for k, v in result.items():
        print(f"{k}: {v}")
