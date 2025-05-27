
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import time

def hitsp_solver(num_cities=1000000, num_clusters=5000, num_groups=200):
    start_time = time.time()

    coords = np.random.rand(num_cities, 2) * 100
    cities = {i: tuple(coords[i]) for i in range(num_cities)}

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=5000).fit(coords)
    cluster_labels = kmeans.labels_

    clusters = defaultdict(list)
    for city_idx, label in enumerate(cluster_labels):
        clusters[label].append(city_idx)

    cluster_centers = np.array([np.mean([cities[i] for i in clusters[cid]], axis=0) for cid in clusters])
    group_kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=1, batch_size=500).fit(cluster_centers)
    group_labels = group_kmeans.labels_

    cluster_to_group = {cid: group_labels[i] for i, cid in enumerate(clusters)}
    groups = defaultdict(list)
    for cid, gid in cluster_to_group.items():
        groups[gid].append(cid)

    total_time = round(time.time() - start_time, 2)

    return {
        "Algorithmus": "HiTSP - Hierarchical TSP Solver",
        "Städte": num_cities,
        "Cluster": num_clusters,
        "Gruppen": num_groups,
        "Rechenzeit (s)": total_time
    }

if __name__ == "__main__":
    result = hitsp_solver()
    for k, v in result.items():
        print(f"{k}: {v}")
