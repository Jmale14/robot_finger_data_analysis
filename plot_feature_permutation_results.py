from config import set_env_opts

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

base_folder = os.path.join("results", "feat_importance_analysis")


def result_from_folder(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith("trial_summary"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            print(df)
            acc = df["accuracy"][0]
            f1 = df["f1_score"][0]
            return acc, f1

def plot_bar_chart(accuracies, f1_scores):
    # Convert to float (they may currently be object dtype)
    accuracies = accuracies.astype(float)
    f1_scores = f1_scores.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    metrics = [
        ("Accuracy", accuracies),
        ("F1 Score", f1_scores),
    ]

    bar_width = 0.2
    x = np.arange(len(accuracies.index))

    for ax, (title, df) in zip(axes, metrics):
        for i, col in enumerate(df.columns):
            ax.bar(
                x + i * bar_width,
                df[col],
                width=bar_width,
                label=col
            )

        ax.set_title(title)
        ax.set_xticks(x + bar_width * (len(df.columns) - 1) / 2)
        ax.set_xticklabels(df.index, rotation=20)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Feature Removed")

    plt.tight_layout()
    plt.show()

def plot_heatmap(accuracies, f1_scores):
    accuracies = accuracies.astype(float)
    f1_scores = f1_scores.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    for ax, df, title in zip(
        axes,
        [accuracies, f1_scores],
        ["Accuracy", "F1 Score"]
    ):
        im = ax.imshow(df.values, cmap="viridis", vmin=0, vmax=1)

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title(title)

        # Display values
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(
                    j, i,
                    f"{df.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white"
                )

    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.show()

def get_results():
    rows = ["texture", "softness", "TS_texture", "TS_softness"]
    columns = ["accel", "gyro", "press", "None"]
    accuracies = pd.DataFrame(index=rows, columns=columns)
    f1_scores = pd.DataFrame(index=rows, columns=columns)
    for recognition_type in ["texture", "softness"]:
        for text_soft in [False, True]:
            for feat_to_blank in ["accel", "gyro", "press"]:
                if text_soft:
                    folder = os.path.join(base_folder, "ind_feats", f"_text_soft_{recognition_type}", f"{feat_to_blank}")
                    row = f"TS_{recognition_type}"
                else:
                    folder = os.path.join(base_folder, "ind_feats", f"_{recognition_type}", f"{feat_to_blank}")
                    row = recognition_type

                acc, f1 = result_from_folder(folder)
                accuracies.loc[row, feat_to_blank] = acc
                f1_scores.loc[row, feat_to_blank] = f1
                        
            # Get result for all features
            if text_soft:
                folder = os.path.join(base_folder, "all_features", f"_text_soft_{recognition_type}")
                row = f"TS_{recognition_type}"
            else:
                folder = os.path.join(base_folder, "all_features", f"_{recognition_type}")
                row = recognition_type
            
            acc, f1 = result_from_folder(folder)
            accuracies.loc[row, "None"] = acc
            f1_scores.loc[row, "None"] = f1

    print("Accuracies:")
    print(accuracies)
    print("F1 Scores:")
    print(f1_scores)
    return accuracies, f1_scores

if __name__ == "__main__":
    accuracies, f1_scores = get_results()
    plot_bar_chart(accuracies, f1_scores)
    plot_heatmap(accuracies, f1_scores)

