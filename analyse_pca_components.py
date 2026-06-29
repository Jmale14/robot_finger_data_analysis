import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FEATURE_NAMES = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "press",
]


def load_pcas(pca_path):
    pcas = joblib.load(pca_path)
    if isinstance(pcas, list):
        return pcas
    return [pcas]


def component_summary(pca, feature_names):
    loadings = pca.components_
    explained_ratio = pca.explained_variance_ratio_
    n_components, n_features = loadings.shape

    summary_rows = []
    for comp_idx in range(n_components):
        comp_weights = loadings[comp_idx]
        abs_weights = np.abs(comp_weights)
        total_abs = abs_weights.sum()
        for feat_idx, feat_name in enumerate(feature_names):
            summary_rows.append(
                {
                    "component": f"PC{comp_idx + 1}",
                    "feature": feat_name,
                    "weight": comp_weights[feat_idx],
                    "abs_weight": abs_weights[feat_idx],
                    "relative_importance": abs_weights[feat_idx] / total_abs,
                    "explained_variance_ratio": explained_ratio[comp_idx],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["weighted_importance"] = summary_df["abs_weight"] * summary_df["explained_variance_ratio"]
    return summary_df


def aggregate_feature_importance(summary_df):
    agg = summary_df.groupby("feature")[["abs_weight", "weighted_importance"]].sum()
    agg = agg.assign(
        normalized_abs=lambda df: df["abs_weight"] / df["abs_weight"].sum(),
        normalized_weighted=lambda df: df["weighted_importance"] / df["weighted_importance"].sum(),
    )
    return agg.sort_values(by="normalized_weighted", ascending=False)


def plot_explained_variance(pcas, out_dir):
    plt.figure(figsize=(8, 5))
    for idx, pca in enumerate(pcas):
        ratios = pca.explained_variance_ratio_
        plt.plot(range(1, len(ratios) + 1), ratios, marker="o", label=f"Fold {idx + 1}")
    mean_ratios = np.mean([pca.explained_variance_ratio_ for pca in pcas], axis=0)
    plt.plot(range(1, len(mean_ratios) + 1), mean_ratios, marker="s", color="black", label="Mean")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance Ratio")
    plt.xticks(range(1, len(mean_ratios) + 1))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_explained_variance_ratio.png")
    plt.show()


def plot_loading_heatmap(summary_df, out_dir):
    pivot = summary_df.pivot_table(
        index="component",
        columns="feature",
        values="weight",
        aggfunc="mean",
    )
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".3f", linewidths=0.5)
    plt.title("PCA Component Loadings")
    plt.xlabel("Feature")
    plt.ylabel("Principal Component")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_loading_heatmap.png")
    plt.show()


def plot_top_feature_contributions(summary_df, out_dir):
    pivot_abs = summary_df.pivot_table(
        index="component",
        columns="feature",
        values="abs_weight",
        aggfunc="mean",
    )
    pivot_norm = pivot_abs.div(pivot_abs.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 5))
    pivot_norm.plot(kind="bar", stacked=True, colormap="tab20", width=0.8)
    plt.title("Relative Feature Contribution to Each Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Relative Contribution")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_feature_contributions.png")
    plt.show()


def analyze_pcas(pcas, feature_names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for fold_idx, pca in enumerate(pcas):
        print(f"\n=== PCA Fold {fold_idx + 1} ===")
        print("Explained variance ratio:")
        for comp_idx, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{comp_idx + 1}: {ratio:.4f}")

        summary_df = component_summary(pca, feature_names)
        all_summaries.append(summary_df.assign(fold=fold_idx + 1))

        print("\nTop feature weights for each component:")
        for comp_idx in range(pca.n_components_):
            comp_df = summary_df[summary_df["component"] == f"PC{comp_idx + 1}"]
            comp_df = comp_df.sort_values(by="abs_weight", ascending=False)
            print(f"  PC{comp_idx + 1}:")
            for _, row in comp_df.head(3).iterrows():
                print(
                    f"    {row['feature']:>8}: weight={row['weight']:+.4f}, abs={row['abs_weight']:.4f}, rel={row['relative_importance']:.3f}"
                )

    combined_df = pd.concat(all_summaries, ignore_index=True)
    agg = aggregate_feature_importance(combined_df)

    print("\n=== Aggregate Feature Importance Across PCA Components ===")
    print(agg[["normalized_abs", "normalized_weighted"]].round(4))

    plot_explained_variance(pcas, out_dir)
    plot_loading_heatmap(combined_df, out_dir)
    plot_top_feature_contributions(combined_df, out_dir)

    return agg


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PCA component importances for text&soft PCA model.")
    parser.add_argument(
        "--pca-path",
        default=os.path.join("processed_data", "text&soft", "pca_True", "pcas.pkl"),
        help="Path to the saved PCA model file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pca_analysis",
        help="Directory to save plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pca_path = Path(args.pca_path)
    out_dir = Path(args.output_dir)

    if not pca_path.exists():
        raise FileNotFoundError(f"PCA file not found: {pca_path}")

    pcas = load_pcas(pca_path)
    agg_df = analyze_pcas(pcas, FEATURE_NAMES, out_dir)

    print("\nSaved analysis plots to:", out_dir)
    print("Use the output above to determine which features most strongly influence the first 5 principal components.")
