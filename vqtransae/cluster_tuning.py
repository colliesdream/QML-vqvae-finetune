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

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None


def load_members(path: Path) -> pd.DataFrame:
    """Load cluster members from CSV or JSON."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return pd.DataFrame(data)
    raise ValueError("cluster_members file must be .csv or .json")


def build_cluster_summary(
    members: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    clusterer: "hdbscan.HDBSCAN",
    min_cluster_size: int,
    min_samples: int,
    output_dir: Path,
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
    summary_path = output_dir / "cluster_summary.json"
    members_path = output_dir / "cluster_members.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(cluster_summary, handle, ensure_ascii=False, indent=2)
    with members_path.open("w", encoding="utf-8") as handle:
        json.dump(cluster_members, handle, ensure_ascii=False, indent=2)

    cluster_summary_sorted = sorted(cluster_summary, key=lambda item: item["size"], reverse=True)
    stability_sorted = sorted(cluster_summary, key=lambda item: item["stability"], reverse=True)

    with (output_dir / "cluster_summary_by_size.json").open("w", encoding="utf-8") as handle:
        json.dump(cluster_summary_sorted, handle, ensure_ascii=False, indent=2)
    with (output_dir / "cluster_summary_by_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(stability_sorted, handle, ensure_ascii=False, indent=2)

    noise_ratio = float((labels == -1).mean())
    clustered_mask = labels != -1
    silhouette = None
    if clustered_mask.sum() > 1 and len(np.unique(labels[clustered_mask])) > 1:
        silhouette = float(silhouette_score(anomaly_features[clustered_mask], labels[clustered_mask]))

    print("\n" + "=" * 70)
    print("Phase 5: HDBSCAN clustering (from cluster_members)")
    print("=" * 70)
    print(f"Anomalies clustered: {len(anomaly_indices)}")
    print(f"Clusters found: {len(cluster_summary)}")
    print(f"Noise ratio: {noise_ratio:.4f}")
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.4f}")
    print(f"HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"HDBSCAN min_samples: {min_samples}")


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
    args = parser.parse_args()

    if hdbscan is None:
        raise SystemExit("hdbscan is required. Install with: pip install \"hdbscan>=0.8.33\"")

    members = load_members(args.members)
    features = members[["E_norm", "D_norm", "A_norm"]].to_numpy()

    min_cluster_size = max(2, args.min_cluster_size)
    min_samples = max(1, args.min_samples)

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


if __name__ == "__main__":
    main()
