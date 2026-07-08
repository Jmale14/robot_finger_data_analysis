import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X, y_true):
    """
    Returns accuracy and weighted F1.
    """

    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)

    # If y_true is one-hot encoded
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1

def permutation_importance_sensor(
    model,
    X,
    y,
    sensor_groups,
    n_repeats=20,
    random_state=None,
):
    """
    Parameters
    ----------
    model : trained keras model

    X : ndarray
        Shape = (samples, windows, timesteps, features)

    y : labels

    sensor_groups : dict
        Example:
        {
            "accel": [0,1,2],
            "gyro": [3,4,5],
            "press": [6]
        }

    Returns
    -------
    dict containing importances for accuracy and F1.
    """
    def depth(lst):
        if isinstance(lst, list):
            return 1 + max(depth(item) for item in lst)
        else:
            return 0

    rng = np.random.default_rng(random_state)

    baseline_acc, baseline_f1 = evaluate_model(model, X, y)

    results = {}

    n_samples = X.shape[0]

    for sensor_name, feature_idx in sensor_groups.items():
        sens_depth = depth(feature_idx) # In multi sensor perm case we will do each sensor type separately
        variants = 1
        if sens_depth > 1:
            variants = len(feature_idx)
        else:
            feature_idx = [feature_idx]
        acc_scores = []
        f1_scores = []

        for _ in range(n_repeats):

            X_perm = X.copy()
            rand_copy = X.copy()

            # Shuffle samples
            for v in range(variants):
                perm = rng.permutation(n_samples)
                rand_copy = X_perm[perm, :, :, :]

                # Permute every feature belonging to this sensor
                X_perm[:, :, :, feature_idx[v]] = rand_copy[:, :, :, feature_idx[v]]

            acc, f1 = evaluate_model(model, X_perm, y)

            acc_scores.append(acc)
            f1_scores.append(f1)

        acc_scores = np.array(acc_scores)
        f1_scores = np.array(f1_scores)

        results[sensor_name] = {
            "accuracy_mean": baseline_acc - acc_scores.mean(),
            "accuracy_std": acc_scores.std(),
            "f1_mean": baseline_f1 - f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "accuracy_scores": acc_scores,
            "f1_scores": f1_scores,
        }
    
    # Validate on no permutations
    for _ in range(n_repeats):
        acc_scores = []
        f1_scores = []
        X_perm = X.copy()
        acc, f1 = evaluate_model(model, X_perm, y)
        acc_scores.append(acc)
        f1_scores.append(f1)

    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)

    results["None"] = {
        "accuracy_mean": baseline_acc - acc_scores.mean(),
        "accuracy_std": acc_scores.std(),
        "f1_mean": baseline_f1 - f1_scores.mean(),
        "f1_std": f1_scores.std(),
        "accuracy_scores": acc_scores,
        "f1_scores": f1_scores,
    }
    
    return results

def print_perm_results(results):
    for sensor, r in results.items():
        print(sensor)
        print(
            f"Accuracy importance = "
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        )

        print(
            f"F1 importance = "
            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        )

        print()

def plot_perm_results(results):
    import matplotlib.pyplot as plt

    sensors = list(results.keys())

    acc_mean = [results[s]["accuracy_mean"] for s in sensors]
    acc_std  = [results[s]["accuracy_std"] for s in sensors]

    f1_mean = [results[s]["f1_mean"] for s in sensors]
    f1_std  = [results[s]["f1_std"] for s in sensors]

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].bar(
        sensors,
        acc_mean,
        yerr=acc_std,
        capsize=5
    )
    axes[0].set_title("Permutation Importance (Accuracy)")
    axes[0].set_ylabel("Decrease in Accuracy")

    axes[1].bar(
        sensors,
        f1_mean,
        yerr=f1_std,
        capsize=5
    )
    axes[1].set_title("Permutation Importance (F1)")
    axes[1].set_ylabel("Decrease in F1")

    plt.tight_layout()
    plt.show()


def save_permutation_results(all_fold_results, save_path):
    """
    Average permutation importance results across CV folds and save to CSV.

    Parameters
    ----------
    all_fold_results : list
        List of dictionaries returned by permutation_importance_sensor().

    save_path : str
        Path to output CSV.
    """

    sensors = all_fold_results[0].keys()

    rows = []

    for sensor in sensors:

        acc_mean = np.mean(
            [fold[sensor]["accuracy_mean"] for fold in all_fold_results]
        )

        acc_std = np.std(
            [fold[sensor]["accuracy_mean"] for fold in all_fold_results],
            ddof=1,
        )

        f1_mean = np.mean(
            [fold[sensor]["f1_mean"] for fold in all_fold_results]
        )

        f1_std = np.std(
            [fold[sensor]["f1_mean"] for fold in all_fold_results],
            ddof=1,
        )

        rows.append({
            "sensor": sensor,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    return df