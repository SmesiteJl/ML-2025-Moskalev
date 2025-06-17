import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X, visualize=False, dimension_pairs=None):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for iteration in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            if visualize and dimension_pairs:
                self._plot_iteration(X, iteration, dimension_pairs)
                plt.show(block=False)
                plt.pause(2)
                plt.close()

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def _plot_iteration(self, X, iteration, dimension_pairs):
        plt.figure(figsize=(15, 10))
        for i, (d1, d2) in enumerate(dimension_pairs, 1):
            plt.subplot(2, 3, i)
            plt.scatter(X[:, d1], X[:, d2], c=self.labels, cmap='tab10', s=50)
            plt.scatter(self.centroids[:, d1], self.centroids[:, d2],
                        c='red', marker='X', s=200, linewidths=2)
            plt.xlabel(iris.feature_names[d1])
            plt.ylabel(iris.feature_names[d2])
            plt.title(f'Iteration {iteration + 1}: Dimensions {d1} & {d2}')
        plt.tight_layout()

def show_elbow_plot(X, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data

    print("Displaying elbow plot...")
    show_elbow_plot(X)

    optimal_k = 3  # manually set
    print(f"Using number of clusters: {optimal_k}")

    dimension_pairs = list(combinations(range(X.shape[1]), 2))

    print("Starting custom KMeans clustering with visualization...")
    kmeans = CustomKMeans(n_clusters=optimal_k, max_iter=20)
    kmeans.fit(X, visualize=True, dimension_pairs=dimension_pairs)

    print("Clustering complete. Displaying final result...")

    plt.figure(figsize=(15, 10))
    for i, (d1, d2) in enumerate(dimension_pairs, 1):
        plt.subplot(2, 3, i)
        plt.scatter(X[:, d1], X[:, d2], c=kmeans.labels, cmap='tab10', s=50)
        plt.scatter(kmeans.centroids[:, d1], kmeans.centroids[:, d2],
                    c='red', marker='X', s=200, linewidths=2)
        plt.xlabel(iris.feature_names[d1])
        plt.ylabel(iris.feature_names[d2])
        plt.title(f'Final Clusters: Dimensions {d1} & {d2}')
    plt.tight_layout()
    plt.show()

    print("Finished")
