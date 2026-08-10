from config import set_env_opts

"""Feature importance and ablation analysis for trained tactile recognition models.

This script evaluates preprocessed, PCA-transformed test data using trained CNN-LSTM models.
It supports feature blanking, permutation importance, and result export under
`results/feat_importance_analysis/`.
"""

import os
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import load_model
from utils.permutation_importance import permutation_importance_sensor, print_perm_results, plot_perm_results, save_permutation_results, plot_permutation_importance
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from utils.plot_confusion_matrix import plot_confusion_matrix
import datetime
from utils.model_training_utils import time_divide_data, save_results
import argparse

class KerasPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # Optional
    def predict_proba(self, X):
        return self.model.predict(X)

def blank_features(windows, feat_to_blank):
    if feat_to_blank == "None":
        return windows
    elif feat_to_blank == "permutation":
        return windows
    else:
        # Blank out specific feature (set to zero)
        feature_indices = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "press": [6]}
        if feat_to_blank not in feature_indices:
            raise ValueError(f"Invalid feature to blank: {feat_to_blank}. Must be one of {list(feature_indices.keys())}.")
        indices_to_blank = feature_indices[feat_to_blank]
        for i in range(len(windows)):
            windows[i][:, indices_to_blank] = 0

    return windows

def prepare_data_for_evaluation(recognition_type, text_soft=False, feat_to_blank="None", folds=1):

    if text_soft:
        base_folder = f"processed_data/text&soft/pca_True/pre_and_pca_data/"
    else:
        base_folder = f"processed_data/{recognition_type}/pca_True/pre_and_pca_data/"

    pcas = joblib.load(f"{base_folder}/pcas.pkl")
    data = joblib.load(f"{base_folder}/normalized_folds_orig.pkl")

    if recognition_type == "texture":
        encoder = joblib.load(f"{base_folder}/labelsencoder.pkl")
    elif recognition_type == "softness":
        encoder = joblib.load(f"{base_folder}/softnessencoder.pkl")
        
    test_win_folds = []
    test_lab_folds = []
    for f in range(folds):
        (_, _, test_windows, test_labels) = data[f]
        test_windows = blank_features(test_windows, feat_to_blank)

        if recognition_type == "texture":
            test_labels = test_labels[:, 0]
            test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1).astype(int))
        elif recognition_type == "softness":
            test_labels = test_labels[:, 1]
            test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))
        
        test_win_folds.append(test_windows)
        test_lab_folds.append(test_labels_encoded)

    return test_win_folds, test_lab_folds, encoder.categories_, pcas

def pca_transform(test_windows, pca):
    new_test_windows = np.empty((test_windows.shape[0], test_windows.shape[1], test_windows.shape[2], 5))
    for s in range(test_windows.shape[0]):
        for j in range(test_windows.shape[1]):
            new_test_windows[s, j, :, :] = pca.transform(test_windows[s, j, :, :])
    return new_test_windows


def run_trial(recognition_type, text_soft=False, feat_to_blank="None", folds=1):
    # Lists to store results
    accuracy_scores = []
    rec_scores = []
    prec_scores = []
    f1_scores = []
    fold_histories = [None]
    all_y_true = []
    all_y_pred = []
    all_perm_results = []

    test_windows_folds, test_labels_encoded_folds, label_categories, pcas = prepare_data_for_evaluation(recognition_type, text_soft, feat_to_blank, folds)
    
    for f in range(folds):
        test_windows = test_windows_folds[f]
        test_labels_encoded = test_labels_encoded_folds[f]

        # Time divide the data into smaller windows
        data_folds = time_divide_data([[np.empty(0), [None], test_windows, test_labels_encoded]])
        test_windows = np.array(data_folds[0][2])

        # Load model
        if text_soft:
            model_path = os.path.join(
                "results",
                "feat_importance_analysis",
                f"text&soft_{recognition_type}_feat_study_CNN-LSTM",
                f"text&soft_{recognition_type}_pcaTrue_CNN-LSTM_feat_study_Model_fold{f}.keras",
            )
        else:
            model_path = os.path.join(
                "results",
                "feat_importance_analysis",
                f"{recognition_type}_feat_study_CNN-LSTM",
                f"{recognition_type}_{recognition_type}_pcaTrue_CNN-LSTM_feat_study_Model_fold{f}.keras",
            )
        model = load_model(model_path, compile=False)

        pca_transformer = FunctionTransformer(pca_transform, kw_args={'pca': pcas[f]}, validate=False)

        pipeline = Pipeline([
            ("preprocess", pca_transformer),
            ("classifier", KerasPredictor(model))
        ])

        pipeline.fit(test_windows[:1], test_labels_encoded[:1])   # Doesn't train anything

        if feat_to_blank == "permutation":
            # Perform permutation importance
            sensor_groups = {"Accelerometer": [0,1,2], "Gyroscope": [3,4,5], "Pressure": [6], "AccelGyro": [[0,1,2],[3,4,5]], "AccelPress": [[0,1,2],[6]], "GyroPress": [[3,4,5],[6]], "All": [[0,1,2],[3,4,5],[6]]}
            results = permutation_importance_sensor(
                pipeline,
                test_windows,
                test_labels_encoded,
                sensor_groups,
                n_repeats=50,
                random_state=42,
            )
            # print_perm_results(results)
            # plot_perm_results(results)
            all_perm_results.append(results)
        else:
            # Evaluate on Test data
            print(f"Input test data shape: {test_windows.shape}")
            y_test_pred = pipeline.predict(test_windows)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            y_test_true = np.argmax(test_labels_encoded, axis=1)
            
            # Accumulate predictions and true labels for confusion matrix
            all_y_true.append(y_test_true)
            all_y_pred.append(y_test_pred)

            # Calculate scores for test
            f1 = f1_score(y_test_true, y_test_pred, average='macro')
            test_accuracy = accuracy_score(y_test_true, y_test_pred)
            test_prec = precision_score(y_test_true, y_test_pred, average='macro')
            test_rec = recall_score(y_test_true, y_test_pred, average='macro')

            accuracy_scores.append(test_accuracy)
            rec_scores.append(test_rec)
            prec_scores.append(test_prec)
            f1_scores.append(f1)

    if feat_to_blank == "permutation":
        return all_perm_results, label_categories
    else:
        # Print results
        print(f"Test Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        print(f"Test F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Test Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
        print(f"Test Recall: {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

        results = {"acc"  : np.mean(accuracy_scores),
                "f1"   : np.mean(f1_scores),
                "prec" : np.mean(prec_scores),
                "rec"  : np.mean(rec_scores),
                "std_acc": np.std(accuracy_scores),
                "std_f1": np.std(f1_scores),
                "std_prec": np.std(prec_scores),
                "std_rec": np.std(rec_scores),
                "yTrue": all_y_true, 
                "yPred": all_y_pred, 
                "hist" : fold_histories}

    return results, label_categories


def setup_and_run_trial(recognition_type, text_soft=False, feat_to_blank="None", plot_results=True, save_folder_app="", folds=1):
    save_folder = f"results/feat_importance_analysis/"
    
    if feat_to_blank == "None":
        save_folder = save_folder + "all_features/"
    elif feat_to_blank == "permutation":
        save_folder = save_folder + "permutation/"
    else:
        save_folder = save_folder + "ind_feats/"
    
    if text_soft:
        save_folder = save_folder + f"_text_soft_{recognition_type}"
    else:
        save_folder = save_folder + f"_{recognition_type}"

    if feat_to_blank in ["accel", "gyro", "press"]:
        save_folder = save_folder + f"/{feat_to_blank}"

    # Append optional save folder appendix
    if save_folder_app:
        save_folder = save_folder + "_" + save_folder_app

    os.makedirs(save_folder, exist_ok=True)
    results, categories = run_trial(recognition_type, text_soft, feat_to_blank, folds)
    
    if feat_to_blank == "permutation":
        if text_soft:
            df = save_permutation_results(results, f"{save_folder}/permutation_importance_TS_{recognition_type}.csv")
        else:
            df = save_permutation_results(results, f"{save_folder}/permutation_importance_{recognition_type}.csv")
        print(df)
        plot_permutation_importance(
            df,
            save_path=f"{save_folder}/permutation_importance.png",
            show=False,
        )
    else:
        # Plot confusion matrix
        if plot_results:
            if recognition_type == "texture":
                categories = [c+1 for c in categories]
            plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories, save_dir=save_folder)

        hparam_hist = []
        save_results(results, 1, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, save_dir=save_folder, file_appendix=f'')
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/evaluation with configurable options.")
    parser.add_argument("--recognition-type", default="texture", choices=["texture", "softness"], help="Recognition target for the analysis.")
    parser.add_argument("--no-plot", dest="plot_results", action="store_false", help="Disable plotting of results")
    parser.set_defaults(plot_results=True)
    parser.add_argument("--text-soft", dest="text_soft", action="store_false", help="Texture and Softness combined dataset vs unimodal dataset")
    parser.set_defaults(text_soft=True)
    parser.add_argument("--all-variants", dest="run_all_variants", action="store_true", help="Run all recognition targets and both combined/unimodal variants")
    parser.set_defaults(run_all_variants=False)
    parser.add_argument("--save-folder-app", dest="save_folder_app", default="", help="Optional appendix to append to the results save folder")
    parser.add_argument("--feat_to_blank", dest="feat_to_blank", default="permutation", choices=["accel", "gyro", "press", "permutation", "None"], help="Optional feature to blank out for ablation study")
    args = parser.parse_args()

    folds = 5
    run_all_variants = args.run_all_variants if hasattr(args, "run_all_variants") else False
    plot_results = args.plot_results if hasattr(args, "plot_results") else True
    save_folder_app = args.save_folder_app if hasattr(args, "save_folder_app") else ""
    
    if not run_all_variants:
        recognition_type = args.recognition_type
        text_soft = args.text_soft if hasattr(args, "text_soft") else False
        feat_to_blank = args.feat_to_blank if hasattr(args, "feat_to_blank") else "None"
        setup_and_run_trial(recognition_type, text_soft, feat_to_blank, plot_results, save_folder_app, folds)
    else:
        # Loop through all versions of the ablation study
        for recognition_type in ["texture", "softness"]:
            for text_soft in [False, True]:
                for feat_to_blank in ["permutation"]:
                # for feat_to_blank in ["accel", "gyro", "press"]:
                    print(f"Running trial for recognition_type={recognition_type}, text_soft={text_soft}, feat_to_blank={feat_to_blank}")
                    setup_and_run_trial(recognition_type, text_soft, feat_to_blank, plot_results, save_folder_app, folds)

    print("Done")