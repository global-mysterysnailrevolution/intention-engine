import numpy as np
from scipy.spatial.distance import pdist, squareform


class CoherenceScorer:
    """Scores candidate node groups for multi-way coherence in utility space."""

    def __init__(self, weights: tuple[float, float, float] = (0.5, 0.4, 0.1)):
        self.w_compact, self.w_distinct, self.w_diverse = weights

    def score(
        self,
        utility_vectors: np.ndarray,  # (m, k) utility vectors for group
        ontologies: list[str] | None = None,  # ontology tags per node
    ) -> float:
        """
        Compute multi-way coherence score for a candidate group.
        Combines compactness, distinctiveness, and ontology diversity.
        Returns score in [0, 1].
        """
        m = utility_vectors.shape[0]
        if m < 2:
            return 0.0

        compactness = self._compactness(utility_vectors)
        distinctiveness = self._distinctiveness(utility_vectors)
        diversity_bonus = self._diversity_bonus(ontologies) if ontologies else 0.0

        score = (
            self.w_compact * compactness
            + self.w_distinct * distinctiveness
            + self.w_diverse * diversity_bonus
        )
        return float(np.clip(score, 0.0, 1.0))

    def _compactness(self, U: np.ndarray) -> float:
        """How tightly grouped in utility space. 1 = perfectly tight."""
        centroid = U.mean(axis=0)
        # Cosine distances from centroid
        norms_u = np.linalg.norm(U, axis=1, keepdims=True)
        norms_u = np.maximum(norms_u, 1e-10)
        U_norm = U / norms_u

        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-10:
            return 0.0
        centroid_norm = centroid / c_norm

        cos_sims = U_norm @ centroid_norm
        return float(np.mean(cos_sims))

    def _distinctiveness(self, U: np.ndarray) -> float:
        """How distinct from random spread. 1 = very tight cluster."""
        if U.shape[0] < 2:
            return 1.0
        # Normalize rows
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        U_norm = U / norms
        # Pairwise cosine similarities
        sim_matrix = U_norm @ U_norm.T
        # Get upper triangle (excluding diagonal)
        mask = np.triu_indices_from(sim_matrix, k=1)
        pairwise_sims = sim_matrix[mask]
        if len(pairwise_sims) == 0:
            return 1.0
        min_sim = float(np.min(pairwise_sims))
        diameter = 1.0 - min_sim
        return float(1.0 - diameter)

    def _diversity_bonus(self, ontologies: list[str]) -> float:
        """Bonus for cross-ontology groups. Only positive if >50% unique ontologies."""
        unique = len(set(ontologies))
        total = len(ontologies)
        if total == 0:
            return 0.0
        diversity = unique / total
        return max(0.0, diversity - 0.5) * 2  # Scale [0.5, 1.0] -> [0.0, 1.0]
