
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import time

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def solve_tsp_nearest_neighbor(cities, subset):
    unvisited = set(subset)
    current = subset[0]
    route = [current]
    unvisited.remove(current)

    while unvisited:
        next_city = min(unvisited, key=lambda city: distance(cities[current], cities[city]))
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return route

def two_opt(route, cities):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route, cities) < route_distance(best, cities):
                    best = new_route
                    improved = True
        if not improved:
            break
    return best

def solve_meta_tsp_nearest_neighbor(centers):
    remaining = set(centers.keys())
    current = list(remaining)[0]
    route = [current]
    remaining.remove(current)

    while remaining:
        next_cluster = min(remaining, key=lambda cid: distance(centers[current], centers[cid]))
        route.append(next_cluster)
        remaining.remove(next_cluster)
        current = next_cluster

    return route

def route_distance(route, cities):
    return sum(distance(cities[route[i]], cities[route[i+1]]) for i in range(len(route) - 1))

def hierarchical_tsp_solver(num_cities=10000, num_clusters=200, num_groups=20):
    start_time = time.time()

    np.random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100
    cities = {i: tuple(coords[i]) for i in range(num_cities)}

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    clusters = defaultdict(list)
    for city_idx, label in enumerate(cluster_labels):
        clusters[label].append(city_idx)

    kmeans_group = KMeans(n_clusters=num_groups, random_state=1).fit(cluster_centers)
    group_labels = kmeans_group.labels_
    cluster_to_group = {cid: group_labels[cid] for cid in range(num_clusters)}

    groups = defaultdict(list)
    for cluster_id, group_id in cluster_to_group.items():
        groups[group_id].append(cluster_id)

    group_city_map = defaultdict(list)
    for group_id, cluster_ids in groups.items():
        for cid in cluster_ids:
            group_city_map[group_id].extend(clusters[cid])

    cluster_routes = {}
    cluster_centroids = {}

    for cluster_id, city_list in clusters.items():
        base = solve_tsp_nearest_neighbor(cities, city_list)
        optimized = two_opt(base, cities)
        cluster_routes[cluster_id] = optimized
        center = np.mean([coords[i] for i in optimized], axis=0)
        cluster_centroids[cluster_id] = tuple(center)

    group_routes = {}
    group_centroids = {}

    for group_id, cluster_ids in groups.items():
        route = solve_meta_tsp_nearest_neighbor({cid: cluster_centroids[cid] for cid in cluster_ids})
        group_routes[group_id] = route
        group_center = np.mean([cluster_centroids[cid] for cid in cluster_ids], axis=0)
        group_centroids[group_id] = tuple(group_center)

    final_group_order = solve_meta_tsp_nearest_neighbor(group_centroids)

    final_route = []
    for group_id in final_group_order:
        for cluster_id in group_routes[group_id]:
            final_route.extend(cluster_routes[cluster_id])

    total_distance = route_distance(final_route, cities)
    total_time = round(time.time() - start_time, 3)

    return {
        "Anzahl Städte": num_cities,
        "Anzahl Cluster": num_clusters,
        "Anzahl Gruppen": num_groups,
        "Gesamtdistanz": round(total_distance, 2),
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hierarchical_tsp_solver()
    for key, value in result.items():
        print(f"{key}: {value}")
