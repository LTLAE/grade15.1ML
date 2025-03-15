import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('seeds_data.csv', header=None)
data = data.values

def calculate_distance_matrix(data):
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:  # set distance to itself as inf
                distance_matrix[i, j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))
            else:
                distance_matrix[i, j] = np.inf
    return distance_matrix

distance_matrix = calculate_distance_matrix(data)

# hierarchical clustering
def hierarchical_clustering(distance_matrix):
    n = distance_matrix.shape[0]
    clusters = {i: [i] for i in range(n)}  # every point is set a cluster
    merge_history = []
    max_cluster = 3

    while len(clusters) > max_cluster:
        # find clusters with the smallest distance
        min_dist = np.inf
        closest_pair = (None, None)
        for i in clusters:
            for j in clusters:
                if i != j:
                    dist = np.min([distance_matrix[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (i, j)

        # combine closest clusters
        cluster1, cluster2 = closest_pair
        clusters[cluster1].extend(clusters[cluster2])
        del clusters[cluster2]
        merge_history.append((cluster1, cluster2, min_dist, len(clusters[cluster1])))

        # update distance matrix
        for i in clusters:
            if i != cluster1:
                dist = np.min([distance_matrix[p1, p2] for p1 in clusters[cluster1] for p2 in clusters[i]])
                distance_matrix[cluster1, i] = dist
                distance_matrix[i, cluster1] = dist

        # remove cluster2 from distance matrix
        for p in clusters[cluster2] if cluster2 in clusters else []:
            distance_matrix[p, :] = np.inf
            distance_matrix[:, p] = np.inf

    return clusters, merge_history


final_clusters, merge_history = hierarchical_clustering(distance_matrix)
print("Final clusters:", final_clusters)
print("Merge history:")
for step in merge_history:
    print(f"Merge clusters {step[0]} and {step[1]} with distance {step[2]:.2f}. New cluster size: {step[3]}")

# draw dendrogram
def draw_dendrogram(merge_history):
    plt.figure(figsize=(10, 5))
    for i, (c1, c2, dist, size) in enumerate(merge_history):
        x = [c1, c1, c2, c2]
        y = [0, dist, dist, 0]
        plt.plot(x, y, color='black')
        plt.text((c1 + c2) * 0.5, dist, f"{size}", ha='center', va='bottom')
    plt.show()

draw_dendrogram(merge_history)
