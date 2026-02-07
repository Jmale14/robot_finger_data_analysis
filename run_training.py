import set_env_opts

import os
import numpy as np
import random
from sklearn.metrics import f1_score
from utils.plot_training_results import plot_training_results
from utils.plot_confusion_matrix import plot_confusion_matrix
import datetime
from hparams import hp_dict
from utils.model_training_utils import create_cnnlstm_model, create_cnn_model, time_divide_data, load_data, save_results


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
    elif model_type == "CNN": data_folds = normalized_folds
    else: raise ValueError("model_type must be either 'CNN-LSTM' or 'CNN'")

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
                model = create_cnnlstm_model(window_size, num_classes, hparams)
            elif model_type == "CNN":
                assert window_size[0] == 100, f"Window size must be 100, got {window_size[0]}"
                model = create_cnn_model(window_size, num_classes, hparams)
            else: raise ValueError("model_type must be either 'CNN-LSTM' or 'CNN'")
            
            print(f"Input train data shape: {train_windows.shape}")
            history = model.fit(train_windows, train_labels_encoded, epochs=hparams["HP_EPOCHS"], batch_size=hparams["HP_BATCH"], validation_data=(test_windows, test_labels_encoded), shuffle=True, verbose=verbose)
            
            if outputModel:
                os.makedirs('models', exist_ok=True)
                model.save(f"models/{dataset}_{recognition_type}_pca{use_pca}_{model_type}_Model.keras")

            # Evaluate on Test data
            print(f"Input test data shape: {test_windows.shape}")
            test_loss, test_accuracy, test_prec, test_rec, test_f1_int = model.evaluate(test_windows, test_labels_encoded, verbose=0)
            y_test_pred = model.predict(test_windows)
            y_test_true = np.argmax(test_labels_encoded, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)

            # Accumulate predictions and true labels for confusion matrix
            all_y_true.append(y_test_true)
            all_y_pred.append(y_test_pred)

            # Calculate F1 score for test
            f1 = f1_score(y_test_true, y_test_pred, average='macro')

            accuracy_scores.append(test_accuracy)
            rec_scores.append(test_rec)
            prec_scores.append(test_prec)
            f1_scores.append(f1)
            f1_int_scores.append(test_f1_int)
            fold_histories.append(history)

    # Print results
    print(f"Test Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Test F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Test F1 Internal Score: {np.mean(f1_int_scores):.4f} ± {np.std(f1_int_scores):.4f}")
    print(f"Test Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print(f"Test Recall: {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

    results = {"acc"  : np.mean(accuracy_scores),
               "f1"   : np.mean(f1_scores), 
               "prec" : np.mean(prec_scores), 
               "rec"  : np.mean(rec_scores), 
               "yTrue": all_y_true, 
               "yPred": all_y_pred, 
               "hist" : fold_histories}

    return results, encoder.categories_



if __name__ == "__main__":   
    dataset = "text&soft" # "texture", "softness", "text&soft" <== Choose dataset to evaluate
    if dataset == "text&soft":
        recognition_type = "softness" # "texture", "softness" <== Choose one for combined text&soft dataset
    else:
        recognition_type = dataset

    hparams = hp_dict[f"{dataset}_{recognition_type}"]
    use_pca = True
    model_type = "CNN" # "CNN-LSTM" or "CNN"
    folds2Test = 5
    outputModel=False
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
    plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories, save_dir=save_folder)

    # Plot average training history across folds
    plot_training_results(results["hist"], save_dir=save_folder)

    hparam_hist = []
    hparam_hist = [[hprm for hprm in hparams.keys()]]
    hparam_hist.append([hprm for hprm in hparams.values()])

    save_results(results, folds2Test, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, save_dir=save_folder, file_appendix=f'{dataset}_{recognition_type}_pca{use_pca}_{model_type}')

    print("Done")