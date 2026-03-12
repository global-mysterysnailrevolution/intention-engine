import numpy as np


class UtilityProjector:
    """Projects node embeddings through intention predicates into utility space."""

    def project(
        self,
        node_embeddings: np.ndarray,  # (n, d)
        predicate_embeddings: np.ndarray,  # (k, d)
        predicate_weights: np.ndarray,  # (k,)
    ) -> np.ndarray:
        """
        Compute utility matrix U where U[i,j] = weighted cosine similarity
        between node i and predicate j.

        Returns: (n, k) utility matrix
        """
        # Normalize node embeddings
        n_norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        n_norms = np.maximum(n_norms, 1e-10)
        F_norm = node_embeddings / n_norms

        # Normalize predicate embeddings
        p_norms = np.linalg.norm(predicate_embeddings, axis=1, keepdims=True)
        p_norms = np.maximum(p_norms, 1e-10)
        E_norm = predicate_embeddings / p_norms

        # Cosine similarity matrix: (n, k)
        U = F_norm @ E_norm.T

        # Apply predicate weights: broadcast (n, k) * (k,)
        U_weighted = U * predicate_weights

        return U_weighted

    def utility_magnitudes(self, utility_matrix: np.ndarray) -> np.ndarray:
        """Compute per-node utility magnitude (L2 norm in utility space)."""
        return np.linalg.norm(utility_matrix, axis=1)

    def filter_by_threshold(
        self,
        utility_matrix: np.ndarray,
        percentile: float = 80.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter to top nodes by utility magnitude.
        Returns (filtered_utility_matrix, active_indices).
        """
        magnitudes = self.utility_magnitudes(utility_matrix)
        threshold = np.percentile(magnitudes, percentile)
        active = magnitudes >= threshold
        return utility_matrix[active], np.where(active)[0]
