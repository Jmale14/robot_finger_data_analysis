import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, LSTM, TimeDistributed, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, F1Score
import joblib


def save_results(results, folds2Test, startTimeStamp, hparam_hist):
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

    for m in ['loss', 'accuracy', 'f1_score']:
        name = f"train_hist_{m}_" + startTimeStamp
        with open(f'{name}.csv', 'a') as out:
            for row in hist[m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')
            
            for row in hist["val_"+m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')

    name = f"trial_details_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(hparam_hist)


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

    # plot_example_data(normalized_folds)

    # Define window size and number of classes
    window_size = normalized_folds[0][0][0].shape[0]  # Assuming all windows have the same size
    num_classes = encoded_labels.shape[1]

    return normalized_folds, window_size, num_classes, encoder

# Define the LSTM model for multi-class classification
def create_cnnlstm_model(input_size, num_classes, hparams):
    modelInput = Input(shape=input_size)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same"))(modelInput)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same"))(model)
    model = TimeDistributed(MaxPooling1D(hparams["HP_POOL"]))(model)
    model = TimeDistributed(Flatten())(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)  # Output layer with softmax for multi-class

    model = Model(modelInput, model)

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(average='macro')])
    
    print(model.summary())
    return model

def time_divide_data(normalized_folds):
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(normalized_folds):
        win_size = 10
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