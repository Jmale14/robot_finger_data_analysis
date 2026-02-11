import set_env_opts

import os
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils.plot_training_results import plot_training_results
from utils.plot_confusion_matrix import plot_confusion_matrix
import datetime
from hparams import hp_dict
from utils.model_training_utils import time_divide_data, load_data, save_results
import utils.model_definitions as model_defs

def run_trial(dataset, recognition_type, hparams, folds, verbose=0, plot=False, outputModel=False, use_pca=True, model_type="CNN-LSTM"):
    # Lists to store results
    accuracy_scores = []
    rec_scores = []
    prec_scores = []
    f1_int_scores = []
    f1_scores = []
    fold_histories = []
    all_y_true = []
    all_y_pred = []

    normalized_folds, window_size, num_classes, encoder = load_data("processed_data/"+dataset+f"/pca_{use_pca}", recognition_type)
    if model_type == "CNN-LSTM": data_folds = time_divide_data(normalized_folds)
    elif model_type in ["CNN", "LSTM", "SVM", "RF", "LR", "KNN", "NB", "DT"]: data_folds = normalized_folds
    else: raise ValueError("model_type must be either 'CNN-LSTM', 'CNN', 'LSTM', 'SVM', 'RF', 'LR', 'KNN', 'NB', or 'DT'")

    # Train the model on each fold
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(data_folds):
        if i >= folds:
            break
        else:
            print(f'Training on fold {i+1}')
            if recognition_type == "texture":
                train_labels = train_labels[:, 0]
                test_labels = test_labels[:, 0]
            elif recognition_type == "softness":
                train_labels = train_labels[:, 1]
                test_labels = test_labels[:, 1]
            else:
                raise ValueError("recognition_type must be either 'texture' or 'softness'")

            # Shuffle data
            shuffle_indices = np.random.permutation(len(train_windows))
            train_windows = np.array(train_windows)[shuffle_indices]
            shuffled_labels = np.array(train_labels)[shuffle_indices]

            test_windows = np.array(test_windows)
            if recognition_type == "texture":
                train_labels_encoded = encoder.transform(shuffled_labels.reshape(-1, 1).astype(int))
                test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1).astype(int))
            elif recognition_type == "softness":
                train_labels_encoded = encoder.transform(shuffled_labels.reshape(-1, 1))
                test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))

            window_size = train_windows[0].shape
            assert window_size == test_windows[0].shape, "Train and Test windows must have the same shape"
            if use_pca: assert window_size[-1] == 5, f"PCA transformed data must have 5 features, got {window_size}"
            else: assert window_size[-1] == 7, f"Non-PCA data must have 7 features, got {window_size}"
            
            if model_type == "CNN-LSTM":
                assert window_size[0] == 19, f"CNN-LSTM model requires time-divided windows of size 19, got {window_size[0]}"
                model = model_defs.create_cnnlstm_model(window_size, num_classes, hparams)
            elif model_type == "CNN":
                assert window_size[0] == 100, f"Window size must be 100, got {window_size[0]}"
                model = model_defs.create_cnn_model(window_size, num_classes, hparams)
            elif model_type == "LSTM":
                assert window_size[0] == 100, f"Window size must be 100, got {window_size[0]}"
                model = model_defs.create_lstm_model(window_size, num_classes, hparams)
            elif model_type == "SVM":
                model = model_defs.create_svm_model()
            elif model_type == "RF":
                model = model_defs.create_random_forest_model()
            elif model_type == "LR":
                model = model_defs.create_logistic_regression_model()
            elif model_type == "KNN":
                model = model_defs.create_knn_model()
            elif model_type == "NB":
                model = model_defs.create_naive_bayes_model()
            elif model_type == "DT":
                model = model_defs.create_decision_tree_model()
            else: raise ValueError("model_type must be either 'CNN-LSTM', 'CNN', 'LSTM', 'SVM', 'RF', 'LR', 'KNN', 'NB', or 'DT'")
            
            print(f"Input train data shape: {train_windows.shape}")
            if model_type in ["CNN", "CNN-LSTM", "LSTM"]:
                history = model.fit(train_windows, train_labels_encoded, epochs=hparams["HP_EPOCHS"], batch_size=hparams["HP_BATCH"], validation_data=(test_windows, test_labels_encoded), shuffle=True, verbose=verbose)
            elif model_type in ["SVM", "RF", "LR", "KNN", "NB", "DT"]:
                # Reshape data for SVM
                num_samples, time_steps, num_features = train_windows.shape
                train_windows_reshaped = train_windows.reshape(num_samples, time_steps * num_features)
                model.fit(train_windows_reshaped, np.argmax(train_labels_encoded, axis=1))
                history = None


            if outputModel:
                os.makedirs('models', exist_ok=True)
                model.save(f"models/{dataset}_{recognition_type}_pca{use_pca}_{model_type}_Model.keras")

            # Evaluate on Test data
            print(f"Input test data shape: {test_windows.shape}")
            # test_loss, test_accuracy, test_prec, test_rec, test_f1_int = model.evaluate(test_windows, test_labels_encoded, verbose=0)
            if model_type in ["CNN", "CNN-LSTM", "LSTM"]:
                y_test_pred = model.predict(test_windows)
                y_test_pred = np.argmax(y_test_pred, axis=1)
            elif model_type in ["SVM", "RF", "LR", "KNN", "NB", "DT"]:
                num_samples, time_steps, num_features = test_windows.shape
                test_windows_reshaped = test_windows.reshape(num_samples, time_steps * num_features)
                y_test_pred = model.predict(test_windows_reshaped)
            
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
            fold_histories.append(history)

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

    return results, encoder.categories_



if __name__ == "__main__":   
    dataset = "texture" # "texture", "softness", "text&soft" <== Choose dataset to evaluate
    if dataset == "text&soft":
        recognition_type = "softness" # "texture", "softness" <== Choose one for combined text&soft dataset
    else:
        recognition_type = dataset

    hparams = hp_dict[f"{dataset}_{recognition_type}"]
    use_pca = True
    model_type = "CNN-LSTM" # "CNN-LSTM", "CNN", "LSTM", "SVM", "RF", "LR", "KNN", "NB", "DT" <== Choose model type to evaluate
    hparams["HP_EPOCHS"] = 500 if model_type == "LSTM" else hparams["HP_EPOCHS"] # Train CNN and LSTM for more epochs to ensure convergence
    folds2Test = 5
    outputModel=False
    plot_results = True
    if outputModel:
        folds2Test = 1

    if dataset == "text&soft":
        save_folder = f"results/{dataset}_{recognition_type}_pca{use_pca}_{model_type}"
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = f"results/{recognition_type}_pca{use_pca}_{model_type}"
        os.makedirs(save_folder, exist_ok=True)

    results, categories = run_trial(dataset, recognition_type, hparams, folds2Test, verbose=1, plot=True, outputModel=outputModel, use_pca=use_pca, model_type=model_type)

    # Plot confusion matrix
    if plot_results:
        if recognition_type == "texture":
            categories = [c+1 for c in categories]
        plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories, save_dir=save_folder)

    # Plot average training history across folds
    if plot_results and model_type in ["CNN", "CNN-LSTM", "LSTM"]:
        plot_training_results(results["hist"], save_dir=save_folder)

    hparam_hist = []
    hparam_hist = [[hprm for hprm in hparams.keys()]]
    hparam_hist.append([hprm for hprm in hparams.values()])

    save_results(results, folds2Test, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, save_dir=save_folder, file_appendix=f'{dataset}_{recognition_type}_pca{use_pca}_{model_type}')

    print("Done")