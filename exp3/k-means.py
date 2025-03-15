import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('seeds_data.csv', header=None)

def init_cluster_centroids(data, k):
    # select random points to init
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters, k):
    return np.array([data[clusters == i].mean(axis=0) for i in range(k)])

def k_means(data, k):
    centroids = init_cluster_centroids(data.values, k)
    while True:
        labels = assign_clusters(data.values, centroids)
        new_centroids = update_centroids(data.values, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


k = 3
labels, centroids = k_means(data, k)

print("Labels: ", labels)
print("Centroids: ", centroids)

# draw clusters
plt.scatter(data[0], data[1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means clustering')

plt.show()

# Elbow Method
# max_k = 10
# def calculate_wcss(data, max_k):
#     wcss = []
#     for k in range(1, max_k + 1):
#         labels, centroids = k_means(data, k)
#         # calculate within-cluster sum of squares
#         wcss.append(np.sum((data.values - centroids[labels])**2))
#     return wcss
#
# wcss = calculate_wcss(data, max_k)
#
# plt.plot(range(1, max_k + 1), wcss, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('WCSS')
# plt.title('Elbow Method')
# plt.show()
