"""
Standalone clustering utility for anomaly windows.

Reads cluster_members.json/csv (E_norm, D_norm, A_norm) and reruns HDBSCAN
with user-specified parameters to avoid retraining the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_members(path: Path) -> pd.DataFrame:
    """Load cluster members from CSV or JSON."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return pd.DataFrame(data)
    raise ValueError("cluster_members file must be .csv or .json")


def build_histograms(indices: np.ndarray, codebook_size: int, eps: float = 1e-8) -> np.ndarray:
    """Build L1-normalized code histograms from VQ indices."""
    histograms = np.zeros((indices.shape[0], codebook_size), dtype=np.float32)
    for i, row in enumerate(indices):
        hist = np.bincount(row, minlength=codebook_size).astype(np.float32)
        histograms[i] = hist
    histograms = histograms + eps
    histograms = histograms / histograms.sum(axis=1, keepdims=True)
    return histograms


def js_distance_matrix(histograms: np.ndarray) -> np.ndarray:
    """Compute Jensen–Shannon distance matrix for histogram features."""
    n, _ = histograms.shape
    distance = np.zeros((n, n), dtype=np.float32)
    log_h = np.log(histograms)
    for i in range(n):
        p = histograms[i]
        log_p = log_h[i]
        m = 0.5 * (histograms + p)
        log_m = np.log(m)
        kl_p = np.sum(p * (log_p - log_m), axis=1)
        kl_q = np.sum(histograms * (log_h - log_m), axis=1)
        distance[i] = np.sqrt(0.5 * (kl_p + kl_q))
    return distance


def build_cluster_summary(
    members: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    clusterer: "hdbscan.HDBSCAN",
    min_cluster_size: int,
    min_samples: int,
    output_dir: Path,
    prefix: str = "",
) -> None:
    """Write cluster summaries and members to disk."""
    anomaly_features = members[["E_norm", "D_norm", "A_norm"]].to_numpy()
    anomaly_indices = members["window_index"].to_numpy()

    cluster_members: List[dict] = []
    cluster_summary: List[dict] = []

    for idx, label, prob, feats in zip(anomaly_indices, labels, probs, anomaly_features):
        cluster_members.append(
            {
                "window_index": int(idx),
                "cluster_id": int(label),
                "E_norm": float(feats[0]),
                "D_norm": float(feats[1]),
                "A_norm": float(feats[2]),
                "membership_probability": float(prob),
                "is_noise": bool(label == -1),
            }
        )

    unique_labels = sorted(set(labels) - {-1})
    for label in unique_labels:
        member_mask = labels == label
        member_indices = anomaly_indices[member_mask]
        member_feats = anomaly_features[member_mask]
        center = member_feats.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(member_feats - center, axis=1)
        rep_order = np.argsort(distances)
        representative_indices = [int(member_indices[i]) for i in rep_order[:5]]

        cluster_summary.append(
            {
                "cluster_id": int(label),
                "size": int(member_mask.sum()),
                "proportion": float(member_mask.mean()),
                "mean_E_norm": float(member_feats[:, 0].mean()),
                "mean_D_norm": float(member_feats[:, 1].mean()),
                "mean_A_norm": float(member_feats[:, 2].mean()),
                "representative_indices": representative_indices,
                "member_indices": [int(idx) for idx in member_indices],
                "stability": float(clusterer.cluster_persistence_[label]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}cluster_summary.json"
    members_path = output_dir / f"{prefix}cluster_members.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(cluster_summary, handle, ensure_ascii=False, indent=2)
    with members_path.open("w", encoding="utf-8") as handle:
        json.dump(cluster_members, handle, ensure_ascii=False, indent=2)

    cluster_summary_sorted = sorted(cluster_summary, key=lambda item: item["size"], reverse=True)
    stability_sorted = sorted(cluster_summary, key=lambda item: item["stability"], reverse=True)

    with (output_dir / f"{prefix}cluster_summary_by_size.json").open("w", encoding="utf-8") as handle:
        json.dump(cluster_summary_sorted, handle, ensure_ascii=False, indent=2)
    with (output_dir / f"{prefix}cluster_summary_by_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(stability_sorted, handle, ensure_ascii=False, indent=2)

    noise_ratio = float((labels == -1).mean())
    clustered_mask = labels != -1
    silhouette = None
    if clustered_mask.sum() > 1 and len(np.unique(labels[clustered_mask])) > 1:
        silhouette = float(silhouette_score(anomaly_features[clustered_mask], labels[clustered_mask]))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        anomaly_features[:, 0],
        anomaly_features[:, 1],
        anomaly_features[:, 2],
        c=labels,
        s=8,
        cmap="tab20",
    )
    ax.set_xlabel("E_norm")
    ax.set_ylabel("D_norm")
    ax.set_zlabel("A_norm")
    fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}cluster_3d.png", dpi=200)
    plt.close(fig)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(anomaly_features)
    fig = plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        pca_coords[:, 0],
        pca_coords[:, 1],
        c=labels,
        s=8,
        cmap="tab20",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}cluster_pca_2d.png", dpi=200)
    plt.close(fig)

    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_coords = reducer.fit_transform(anomaly_features)
        fig = plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=labels,
            s=8,
            cmap="tab20",
        )
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.colorbar(scatter, shrink=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}cluster_umap_2d.png", dpi=200)
        plt.close(fig)

    print("\n" + "=" * 70)
    print(f"Phase 5: HDBSCAN clustering ({prefix.rstrip('_') or 'from cluster_members'})")
    print("=" * 70)
    print(f"Anomalies clustered: {len(anomaly_indices)}")
    print(f"Clusters found: {len(cluster_summary)}")
    print(f"Noise ratio: {noise_ratio:.4f}")
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.4f}")
    print(f"HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"HDBSCAN min_samples: {min_samples}")


def require_columns(members: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    """Ensure required columns exist before clustering."""
    missing = [col for col in columns if col not in members.columns]
    if missing:
        missing_list = ", ".join(missing)
        required_list = ", ".join(columns)
        raise SystemExit(
            f"{context} requires columns: {required_list}. Missing: {missing_list}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune HDBSCAN on anomaly features.")
    parser.add_argument(
        "--members",
        required=True,
        type=Path,
        help="Path to cluster_members.json or cluster_members.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vqtransae_results") / "evaluation" / "cluster_tuning",
        help="Output directory for clustering results",
    )
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--use-vq-hist",
        action="store_true",
        help="Cluster by VQ histogram using JS distance (expects 'indices' column).",
    )
    args = parser.parse_args()

    if hdbscan is None:
        raise SystemExit("hdbscan is required. Install with: pip install \"hdbscan>=0.8.33\"")

    members = load_members(args.members)
    if not args.use_vq_hist:
        require_columns(members, ("E_norm", "D_norm", "A_norm"), "E/D/A clustering")
        features = members[["E_norm", "D_norm", "A_norm"]].to_numpy()

    min_cluster_size = max(2, args.min_cluster_size)
    min_samples = max(1, args.min_samples)

    if not args.use_vq_hist:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.fit_predict(features)
        probs = clusterer.probabilities_

        build_cluster_summary(
            members,
            labels,
            probs,
            clusterer,
            min_cluster_size,
            min_samples,
            args.output_dir,
        )

    if args.use_vq_hist:
        if "indices" not in members.columns:
            raise SystemExit("VQ histogram mode requires an 'indices' column in cluster_members.")
        require_columns(
            members,
            ("E_norm", "D_norm", "A_norm"),
            "VQ histogram clustering summary",
        )

        raw_indices = members["indices"].to_list()
        parsed_indices = np.array([np.asarray(row, dtype=int) for row in raw_indices])
        codebook_size = int(parsed_indices.max()) + 1
        histograms = build_histograms(parsed_indices, codebook_size)
        histograms = normalize(histograms, norm="l1")
        js_dist = js_distance_matrix(histograms).astype(np.float64)

        hist_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        hist_labels = hist_clusterer.fit_predict(js_dist)
        hist_probs = hist_clusterer.probabilities_

        build_cluster_summary(
            members,
            hist_labels,
            hist_probs,
            hist_clusterer,
            min_cluster_size,
            min_samples,
            args.output_dir,
            prefix="vq_hist_",
        )


if __name__ == "__main__":
    main()
