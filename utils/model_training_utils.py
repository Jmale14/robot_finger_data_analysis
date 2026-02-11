import os
import csv
import numpy as np
import joblib


def save_results(results, folds2Test, startTimeStamp, hparam_hist, save_dir='results', file_appendix=''):
    if results["hist"][0] is not None:
        hist = {'loss': [],
                'accuracy': [],
                'f1_score': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1_score': []}
        for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'f1_score', 'val_f1_score']:
            for fold in range(folds2Test):
                hist[metric].append([f'{metric}_{fold+1}']+ results["hist"][fold].history[metric])
            
            hist[metric].append([f'average'] + list(np.mean([history.history[metric] for history in results["hist"]], axis=0)))
            hist[metric].append([f'std'] + list(np.std([history.history[metric] for history in results["hist"]], axis=0)))
            hist[metric].append([])

        os.makedirs(save_dir, exist_ok=True)
        for m in ['loss', 'accuracy', 'f1_score']:
            name = f"{save_dir}/train_hist_{m}_{file_appendix}_" + startTimeStamp
            with open(f'{name}.csv', 'a') as out:
                for row in hist[m]:
                    for col in row:
                        out.write('{0},'.format(col))
                    out.write('\n')
                
                for row in hist["val_"+m]:
                    for col in row:
                        out.write('{0},'.format(col))
                    out.write('\n')

    name = f"{save_dir}/trial_details_{file_appendix}_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(hparam_hist)
    
    name = f"{save_dir}/trial_summary_{file_appendix}_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows([["accuracy", "f1_score", "precision", "recall"],
                         [results["acc"], results["f1"], results["prec"], results["rec"]],
                         [results["std_acc"], results["std_f1"], results["std_prec"], results["std_rec"]]])


def load_data(data_dir: str, data_type: str):
    assert data_type in ['softness', 'texture'], "data_type must be either 'softness' or 'texture'"
    # To load normalized folds and scalers
    normalized_folds = joblib.load(data_dir+'/normalized_folds.pkl')
    scalers = joblib.load(data_dir+'/scalers.pkl')
    if data_type == 'texture':
        encoded_labels = joblib.load(data_dir+'/encoded_texture.pkl')
        encoder = joblib.load(data_dir+'/labelsencoder.pkl')
    elif data_type == 'softness':
        encoded_labels = joblib.load(data_dir+'/encoded_softness.pkl')
        encoder = joblib.load(data_dir+'/softnessencoder.pkl')

    # Define window size and number of classes
    window_size = normalized_folds[0][0][0].shape[0]  # Assuming all windows have the same size
    num_classes = encoded_labels.shape[1]

    return normalized_folds, window_size, num_classes, encoder

def time_divide_data(normalized_folds, win_size=10):
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(normalized_folds):
        for j, win in enumerate(train_windows):
            startIdx = 0
            stopIdx = win_size
            new_win = []
            while stopIdx <= win.shape[0]:
                new_win.append(win[startIdx:stopIdx, :])
                startIdx = int(startIdx+(win_size/2))
                stopIdx = int(stopIdx+(win_size/2))
            train_windows[j] = np.array(new_win)
        
        for j, win in enumerate(test_windows):
            startIdx = 0
            stopIdx = win_size
            new_win = []
            while stopIdx <= win.shape[0]:
                new_win.append(win[startIdx:stopIdx, :])
                startIdx = int(startIdx+(win_size/2))
                stopIdx = int(stopIdx+(win_size/2))
            test_windows[j] = np.array(new_win)
        
    return normalized_folds


