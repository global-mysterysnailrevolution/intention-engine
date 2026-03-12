"""
Manufacturing Demo: Intention-driven cross-ontology discovery.

Demonstrates how the Intention Engine discovers connections across
equipment, process, material, and quality ontologies.

Uses mock embeddings so no ML dependencies are needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from intention_engine import IntentionEngine, EngineConfig


# === Mock Encoder ===
# Creates consistent embeddings from text using hash-based projection.
# Deterministic — same text always gives the same embedding.

class MockEncoder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        # Hash-based deterministic base vector
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self.dim).astype(np.float32)

        # Add semantic signal: words map to specific dimensions
        words = text.lower().split()
        semantic_dims: dict[str, int] = {
            # Contamination-related
            "contamination": 0, "contaminant": 0, "clean": 0, "sterile": 0,
            "purge": 1, "filter": 1, "barrier": 1, "hepa": 1,
            # Material-related
            "titanium": 10, "metal": 10, "alloy": 10,
            "oxidation": 11, "corrosion": 11,
            # Process-related
            "mill": 20, "cnc": 20, "machine": 20, "lathe": 20,
            "print": 21, "3d": 21, "additive": 21, "fdm": 21, "sls": 21,
            "anneal": 22, "heat": 22, "treatment": 22,
            # Quality-related
            "inspect": 30, "quality": 30, "qc": 30, "measure": 30,
            "tolerance": 31, "precision": 31, "accuracy": 31,
            "defect": 32, "reduce": 32, "prevent": 32,
            # Environment-related
            "vacuum": 40, "atmosphere": 40, "inert": 40, "argon": 40,
            "cleanroom": 41, "controlled": 41, "environment": 41,
        }
        for word in words:
            if word in semantic_dims:
                dim_idx = semantic_dims[word]
                vec[dim_idx] += 3.0  # Strong semantic signal
                if dim_idx + 1 < self.dim:
                    vec[dim_idx + 1] += 1.5  # Fuzzy spread to neighbour

        vec = vec / (np.linalg.norm(vec) + 1e-10)
        self._cache[text] = vec
        return vec


def main():
    print("=" * 70)
    print("INTENTION ENGINE — Manufacturing Cross-Ontology Discovery Demo")
    print("=" * 70)

    # Create engine with mock encoder
    config = EngineConfig(
        min_coherence=0.15,            # Lower threshold for demo
        explore_budget=50,
        utility_threshold_percentile=60.0,  # Include more nodes
    )
    engine = IntentionEngine(config=config)
    engine.set_encoder(MockEncoder(dim=384))

    # === Populate nodes from 4 different ontologies ===

    print("\n[1] Adding nodes across 4 ontologies...\n")

    equipment = [
        {"id": "eq_cnc_haas",   "description": "Haas VF-2 CNC mill 5-axis machining center",          "ontology": "equipment"},
        {"id": "eq_printer_sls","description": "SLS 3D printer selective laser sintering",             "ontology": "equipment"},
        {"id": "eq_printer_fdm","description": "Prusa MK4 FDM 3D print additive manufacturing",       "ontology": "equipment"},
        {"id": "eq_furnace",    "description": "Vacuum heat treatment anneal furnace",                  "ontology": "equipment"},
        {"id": "eq_hepa",       "description": "HEPA filter air purge contamination barrier",          "ontology": "equipment"},
        {"id": "eq_cmm",        "description": "CMM coordinate measure machine quality inspect tolerance", "ontology": "equipment"},
        {"id": "eq_lathe",      "description": "CNC lathe turning machine precision",                  "ontology": "equipment"},
        {"id": "eq_vacuum",     "description": "Vacuum chamber inert atmosphere argon purge system",   "ontology": "equipment"},
    ]

    processes = [
        {"id": "proc_fdm",      "description": "FDM additive 3D print layer deposition",              "ontology": "process"},
        {"id": "proc_5axis",    "description": "5-axis CNC mill subtractive machining",               "ontology": "process"},
        {"id": "proc_anneal",   "description": "Stress relief anneal heat treatment cycle",           "ontology": "process"},
        {"id": "proc_inspect",  "description": "Quality inspect dimensional tolerance check measure", "ontology": "process"},
        {"id": "proc_decontam", "description": "Contamination clean purge decontamination protocol sterile", "ontology": "process"},
        {"id": "proc_sls",      "description": "SLS selective laser sintering additive print",        "ontology": "process"},
    ]

    materials = [
        {"id": "mat_ti64",      "description": "Titanium Ti-6Al-4V alloy aerospace metal",            "ontology": "material"},
        {"id": "mat_inconel",   "description": "Inconel 718 nickel alloy high-temperature metal corrosion", "ontology": "material"},
        {"id": "mat_pla",       "description": "PLA polylactic acid thermoplastic 3D print filament", "ontology": "material"},
        {"id": "mat_al7075",    "description": "Aluminum 7075-T6 alloy aerospace metal machine",      "ontology": "material"},
    ]

    quality = [
        {"id": "qc_xrf",        "description": "XRF scanner contamination element analysis inspect quality", "ontology": "quality"},
        {"id": "qc_spc",        "description": "Statistical process control defect reduce prevent quality",  "ontology": "quality"},
        {"id": "qc_ct_scan",    "description": "CT scan internal defect inspect non-destructive quality",   "ontology": "quality"},
        {"id": "qc_surface",    "description": "Surface roughness measure quality precision tolerance",      "ontology": "quality"},
    ]

    all_nodes = equipment + processes + materials + quality
    engine.add_nodes_batch(all_nodes)
    print(f"   Added {len(all_nodes)} nodes across 4 ontologies")
    print(f"   {engine.stats()}")

    # === Search 1: Cross-ontology discovery ===

    print("\n" + "=" * 70)
    print("[2] Search: 'reduce contamination in titanium processing'")
    print("=" * 70)

    result = engine.search("reduce contamination in titanium processing")

    print(f"\n   Exploit: {result.explanation.exploit_stats.nodes_reached} nodes reached")
    print(f"   Explore: {result.explanation.explore_stats.clusters_found} clusters, "
          f"{result.explanation.explore_stats.edges_minted} edges minted")

    print("\n   Top results:")
    for i, sn in enumerate(result.nodes[:10]):
        print(f"   {i+1}. [{sn.source:7s}] {sn.node.id:20s} ({sn.node.ontology:10s}) score={sn.score:.3f}")

    if result.minted_edges:
        print(f"\n   Minted {len(result.minted_edges)} new hyperedges:")
        for edge in result.minted_edges:
            print(f"   - {edge.label}")
            print(f"     Members: {sorted(edge.members)}")
            print(f"     Coherence: {edge.coherence_score:.3f}")

    # === Search 2: Same topic — should exploit existing structure ===

    print("\n" + "=" * 70)
    print("[3] Search: 'prevent contamination in metal parts' (related — should exploit)")
    print("=" * 70)

    result2 = engine.search("prevent contamination in metal parts")

    print(f"\n   Exploit: {result2.explanation.exploit_stats.edges_activated} edges activated")
    print(f"   Explore: {result2.explanation.explore_stats.edges_minted} new edges minted")
    print(f"   (Previous minted edges are being reused by exploit phase!)")

    print("\n   Top results:")
    for i, sn in enumerate(result2.nodes[:8]):
        print(f"   {i+1}. [{sn.source:7s}] {sn.node.id:20s} ({sn.node.ontology:10s}) score={sn.score:.3f}")

    # === Search 3: Completely different intention ===

    print("\n" + "=" * 70)
    print("[4] Search: 'achieve tight tolerance precision machining'")
    print("=" * 70)

    result3 = engine.search("achieve tight tolerance precision machining")

    print(f"\n   Exploit: {result3.explanation.exploit_stats.edges_activated} edges activated")
    print(f"   Explore: {result3.explanation.explore_stats.edges_minted} new edges minted")

    print("\n   Top results:")
    for i, sn in enumerate(result3.nodes[:8]):
        print(f"   {i+1}. [{sn.source:7s}] {sn.node.id:20s} ({sn.node.ontology:10s}) score={sn.score:.3f}")

    # === Final stats ===

    print("\n" + "=" * 70)
    print("[5] Final hypergraph state")
    print("=" * 70)

    stats = engine.stats()
    print(f"\n   Nodes: {stats['nodes']}")
    print(f"   Total edges: {stats['edges']}")
    print(f"   Minted by intention: {stats['minted_edges']}")
    print(f"   Manual: {stats['manual_edges']}")

    print(
        f"\n   The hypergraph grew from 0 edges to {stats['edges']} edges purely through"
    )
    print("   intention-driven search. Each query enriched the structure for future queries.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
