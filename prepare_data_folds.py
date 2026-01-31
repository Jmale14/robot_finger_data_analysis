import set_env_opts

import os
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

from utils.prepare_data_utils import load_csv_files, split_into_folds, fit_pca, plot_pca, create_windows, normalize_windows
from utils.plot_example_data import plot_example_data

dataset_type = "texture" # "texture", "softness", "text&soft"
sampling_freq = 50 # Hz
window_size = 2 # Window size in seconds
n_splits = 5 # Number of folds
use_pca = False
plot_example_seq = False

# Specify the directory containing your CSV files
root_dir = "raw_data"
if dataset_type == "texture":
    directory = [root_dir+"/texture_only"]
elif dataset_type == "softness":
    directory = [root_dir+"/softness_only"]
elif dataset_type == "text&soft":
    directory = [root_dir+"/softness&texture"]
else:
    raise ValueError("Invalid dataset type. Choose from 'texture', 'softness', or 'text&soft'.")

data = load_csv_files(directory, sampling_freq)
windows, labels = create_windows(data, window_size*sampling_freq, overlap=0)
folds = split_into_folds(windows, labels, n_splits)

if plot_example_seq:
    plot_example_data(folds)

# Normalise data and apply PCA
normalized_folds = []
scalers = []
pcas = []
for train_windows, train_labels, test_windows, test_labels in folds:
    train_windows, train_scaler = normalize_windows(train_windows)
    test_windows, _ = normalize_windows(test_windows, scaler=train_scaler)

    if use_pca:
        pca = fit_pca(train_windows)
        train_windows = [pca.transform(window) for window in train_windows]
        test_windows = [pca.transform(window) for window in test_windows]
        pcas.append(pca)

    normalized_folds.append((train_windows, train_labels, test_windows, test_labels))
    scalers.append(train_scaler)
    
if use_pca:
    plot_pca(pcas)

# One-hot encode textures and softness
labels_encoder = OneHotEncoder(sparse_output=False)
texture_labels = np.array(labels[:, 0].astype(int)).reshape(-1, 1)
labels_encoder.fit(np.sort(np.unique(texture_labels)).reshape(-1,1))
encoded_texture = labels_encoder.transform(texture_labels)

softness_encoder = OneHotEncoder(sparse_output=False)
softness = np.array(labels[:, 1]).reshape(-1, 1)
softness_encoder.fit(np.sort(np.unique(softness)).reshape(-1,1))
encoded_softness = softness_encoder.transform(softness)


# Save normalized folds and scalers
save_dir = "processed_data"
os.makedirs(f'{save_dir}/{dataset_type}/pca_{use_pca}', exist_ok=True)
joblib.dump(normalized_folds, f'{save_dir}/{dataset_type}/pca_{use_pca}/normalized_folds.pkl')
joblib.dump(scalers, f'{save_dir}/{dataset_type}/pca_{use_pca}/scalers.pkl')
joblib.dump(encoded_texture, f'{save_dir}/{dataset_type}/pca_{use_pca}/encoded_texture.pkl')
joblib.dump(labels_encoder, f'{save_dir}/{dataset_type}/pca_{use_pca}/labelsencoder.pkl')
joblib.dump(encoded_softness, f'{save_dir}/{dataset_type}/pca_{use_pca}/encoded_softness.pkl')
joblib.dump(softness_encoder, f'{save_dir}/{dataset_type}/pca_{use_pca}/softnessencoder.pkl')
if use_pca:
    joblib.dump(pcas, f'{save_dir}/{dataset_type}/pca_{use_pca}/pcas.pkl')

print("Done preparing data")