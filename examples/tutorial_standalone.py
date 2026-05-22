"""Minimal end-to-end example — drop this into a Jupyter cell or run as a script.

Demonstrates the standalone pysccdc pipeline (detect -> quantify -> correct)
on a small clustered scRNA-seq dataset. scCDC works on a *filtered, clustered*
count matrix, so any AnnData with raw integer counts in ``.X`` and a
categorical cluster label in ``.obs`` will do.

Two data sources are shown:
  * the bundled clustered PBMC 3k dataset (``data/pbmc3k_clustered.h5ad``);
  * the synthetic spiked-contamination dataset from ``pysccdc.datasets``.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad

import pysccdc as cd


def run(adata, cluster_key: str = "cluster") -> None:
    """Detect GCGs, quantify contamination, correct only the GCGs."""
    # 1) detect Global Contamination-causing Genes (GCGs)
    detection = cd.ContaminationDetection(
        adata, cluster_key=cluster_key, min_cell=50, random_state=0,
    )
    gcgs = list(detection.index)
    print(f"detected {len(gcgs)} GCGs: {gcgs[:10]}")

    # 2) dataset-level contamination ratio
    ratio = cd.ContaminationQuantification(adata, detection,
                                           cluster_key=cluster_key)
    print(f"contamination ratio: {ratio:.3e}")

    # 3) gene-specific correction — only the GCGs are touched
    corrected = cd.ContaminationCorrection(adata, detection,
                                           cluster_key=cluster_key)
    before = adata.X.sum()
    after = corrected.layers["Corrected"].sum()
    print(f"total counts {before:.0f} -> {after:.0f} "
          f"({100 * (before - after) / before:.2f}% removed)")
    print("per-GCG thresholds:", corrected.uns["sccdc"]["thresholds"])


def main() -> None:
    # --- bundled real dataset (clustered PBMC 3k) --------------------
    h5ad = Path(__file__).parent.parent / "data" / "pbmc3k_clustered.h5ad"
    if h5ad.exists():
        print(f"== clustered PBMC 3k ({h5ad.name}) ==")
        run(ad.read_h5ad(h5ad), cluster_key="leiden")
        print()

    # --- synthetic spiked-contamination dataset ---------------------
    print("== synthetic spiked-contamination dataset ==")
    sim = cd.datasets.simulate_contaminated(random_state=0)
    run(sim, cluster_key="cluster")
    print("true spiked GCGs:", sim.uns["true_GCGs"])


if __name__ == "__main__":
    main()
