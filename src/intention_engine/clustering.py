import numpy as np
from dataclasses import dataclass


@dataclass
class Cluster:
    indices: np.ndarray  # Indices into the active node array
    centroid: np.ndarray  # Centroid in utility space


def cluster_utility_space(
    U: np.ndarray,
    min_cluster_size: int = 2,
    max_cluster_size: int = 50,
) -> list[Cluster]:
    """
    Cluster nodes in utility space using agglomerative clustering
    (no external dependency on hdbscan).
    """
    m, k = U.shape
    if m < min_cluster_size:
        return []

    # Normalize for cosine-based clustering
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    U_norm = U / norms

    # Compute pairwise cosine similarity matrix
    sim_matrix = U_norm @ U_norm.T

    # Simple agglomerative: merge closest pairs until threshold
    # Use single-linkage with cosine distance threshold
    n = m
    labels = np.arange(n)  # Each point starts in its own cluster
    threshold = 0.5  # Cosine similarity threshold for merging

    # Convert to distance
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, np.inf)

    # Greedy agglomerative
    while True:
        # Find closest pair among different clusters
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        if dist_matrix[i, j] > (1.0 - threshold):
            break  # No more pairs close enough

        # Only merge if they're in different clusters (safety check)
        label_j = labels[j]
        label_i = labels[i]
        if label_i == label_j:
            # Same cluster — invalidate this distance entry
            dist_matrix[i, j] = np.inf
            dist_matrix[j, i] = np.inf
            continue

        # Merge j's cluster into i's cluster
        mask_j = labels == label_j
        labels[mask_j] = label_i

        # Update distance matrix: use average linkage for all members of the new merged cluster
        new_members = np.where(labels == label_i)[0]
        for idx in range(n):
            if labels[idx] == label_i:
                continue
            # Average distance from idx to all members of merged cluster
            dists = [
                dist_matrix[idx, mem]
                for mem in new_members
                if dist_matrix[idx, mem] < np.inf
            ]
            if dists:
                avg_dist = float(np.mean(dists))
            else:
                avg_dist = np.inf
            dist_matrix[idx, i] = avg_dist
            dist_matrix[i, idx] = avg_dist

        # Invalidate rows/cols for all old j-cluster members except representative i
        for mem in np.where(mask_j)[0]:
            dist_matrix[mem, :] = np.inf
            dist_matrix[:, mem] = np.inf

    # Extract clusters
    unique_labels = set(labels)
    clusters = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if min_cluster_size <= len(indices) <= max_cluster_size:
            centroid = U[indices].mean(axis=0)
            clusters.append(Cluster(indices=indices, centroid=centroid))

    return clusters
