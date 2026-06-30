import set_env_opts

import os
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import load_model
from utils.plot_confusion_matrix import plot_confusion_matrix
import datetime
from utils.model_training_utils import time_divide_data, load_data, save_results
from utils.prepare_data_utils import load_csv_files, create_windows, normalize_windows
import argparse


def filter_texture_labels(windows, labels, categories):
    # Filter out labels not in the specified categories
    mask = np.isin(labels[:, 0], categories)
    return windows[mask], labels[mask]

def prepare_data_for_evaluation(recognition_type, text_soft=False):
    sampling_freq = 50 # Hz
    window_size = 2 # Window size in seconds

    if text_soft:
        data = load_csv_files([f"raw_data/{recognition_type}_only"], sampling_freq)
        pcas = joblib.load(f"processed_data/{recognition_type}/pca_True/pcas.pkl")
        #pcas = joblib.load(f"processed_data/text&soft/pca_True/pcas.pkl")
        scalers = joblib.load(f"processed_data/{recognition_type}/pca_True/scalers.pkl")
        #scalers = joblib.load(f"processed_data/text&soft/pca_True/scalers.pkl")
    
    else:
        data = load_csv_files(["raw_data/softness&texture"], sampling_freq)
        # pcas = joblib.load(f"processed_data/{recognition_type}/pca_True/pcas.pkl")
        pcas = joblib.load(f"processed_data/text&soft/pca_True/pcas.pkl")
        # scalers = joblib.load(f"processed_data/{recognition_type}/pca_True/scalers.pkl")
        scalers = joblib.load(f"processed_data/text&soft/pca_True/scalers.pkl")

    windows, labels = create_windows(data, window_size*sampling_freq, overlap=0)

    if text_soft and recognition_type == "texture":
        windows, labels = filter_texture_labels(windows, labels, categories=['1', '3', '8', '11', '14']) 

    # Normalise data and apply PCA
    pca = pcas[0]
    scaler = scalers[0]

    test_windows, _ = normalize_windows(windows, scaler=scaler)
    test_windows = [pca.transform(window) for window in test_windows]

    if recognition_type == "texture":
        if text_soft:
            encoder = joblib.load(f"processed_data/text&soft/pca_True/labelsencoder.pkl")
        else:
            encoder = joblib.load(f"processed_data/{recognition_type}/pca_True/labelsencoder.pkl")
        test_labels = labels[:, 0]
        test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1).astype(int))
    elif recognition_type == "softness":
        if text_soft:
            encoder = joblib.load(f"processed_data/text&soft/pca_True/softnessencoder.pkl")
        else:
            encoder = joblib.load(f"processed_data/{recognition_type}/pca_True/softnessencoder.pkl")
        test_labels = labels[:, 1]
        test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))

    return test_windows, test_labels_encoded, encoder.categories_

def run_trial(recognition_type, text_soft=False):
    # Lists to store results
    accuracy_scores = []
    rec_scores = []
    prec_scores = []
    f1_scores = []
    fold_histories = [None]
    all_y_true = []
    all_y_pred = []

    test_windows, test_labels_encoded, label_categories = prepare_data_for_evaluation(recognition_type, text_soft)
    data_folds = time_divide_data([[np.empty(0), [None], test_windows, test_labels_encoded]])

    # # Test on all data as we trained on separate unimodal data
    test_windows = np.array(data_folds[0][2])
    test_labels = data_folds[0][3]

    # Load model
    if text_soft:
        model_path = os.path.join(
            "results",
            "uni_multi_factor_recog",
            f"text&soft_{recognition_type}_pcaTrue_CNN-LSTM",
            f"text&soft_{recognition_type}_pcaTrue_CNN-LSTM_Model.keras",
        )
    else:
        model_path = os.path.join(
            "results",
            "uni_multi_factor_recog",
            f"{recognition_type}_pcaTrue_CNN-LSTM",
            f"{recognition_type}_{recognition_type}_pcaTrue_CNN-LSTM_Model.keras",
        )
    model = load_model(model_path, compile=False)

    # Evaluate on Test data
    print(f"Input test data shape: {test_windows.shape}")
    y_test_pred = model.predict(test_windows)
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


def setup_and_run_trial(recognition_type, text_soft=False, plot_results=True, save_folder_app=""):
    if text_soft:
        save_folder = f"results/uni_multi_factor_recog/text&soft_{recognition_type}"
    else:
        save_folder = f"results/uni_multi_factor_recog/{recognition_type}"
    # Append optional save folder appendix
    if save_folder_app:
        save_folder = save_folder + "_" + save_folder_app

    os.makedirs(save_folder, exist_ok=True)
    results, categories = run_trial(recognition_type, text_soft)
    
    # Plot confusion matrix
    if plot_results:
        if recognition_type == "texture":
            categories = [c+1 for c in categories]
        plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories, save_dir=save_folder)

    hparam_hist = []
    save_results(results, 1, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, save_dir=save_folder, file_appendix=f'_uni_multi_factor_recog')
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/evaluation with configurable options.")
    parser.add_argument("--recognition-type", default="texture", choices=["texture", "softness"], help="Recognition target (overrides default for combined dataset)")
    parser.add_argument("--no-plot", dest="plot_results", action="store_false", help="Disable plotting of results")
    parser.set_defaults(plot_results=True)
    parser.add_argument("--text_soft", dest="text_soft", action="store_false", help="Texture and Softness base model")
    parser.set_defaults(text_soft=True)
    parser.add_argument("--save-folder-app", dest="save_folder_app", default="", help="Optional appendix to append to the results save folder")

    args = parser.parse_args()

    recognition_type = args.recognition_type
    plot_results = args.plot_results if hasattr(args, "plot_results") else True
    text_soft = args.text_soft if hasattr(args, "text_soft") else False

    save_folder_app = args.save_folder_app if hasattr(args, "save_folder_app") else ""
    setup_and_run_trial(recognition_type, text_soft, plot_results, save_folder_app)

    print("Done")