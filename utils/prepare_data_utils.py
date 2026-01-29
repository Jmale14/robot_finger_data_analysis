import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def getSoftness(filename):
    if (filename.__contains__('dragon')) or (filename.__contains__('Dragon')):
        softness = 'dragonskin30'
    elif (filename.__contains__('flex20')) or (filename.__contains__('Flex20')):
        softness = 'echoflex20'
    elif (filename.__contains__('flex30')) or (filename.__contains__('Flex30')):
        softness = 'echoflex30'
    elif (filename.__contains__('foam')) or (filename.__contains__('Foam')):
        softness = 'foam'
    else:
        print(f"Unknown Softness Value: {filename}")
        raise ValueError
    return softness

def trim_to_peaks(data, sampling_freq, plot=False):
    pressure_data = -(data['pressure']-np.mean(data['pressure']))
    peaks = []
    prom_min = 100

    while (max(pressure_data) > 500):
        new_start = np.argmax(pressure_data)+1
        pressure_data = pressure_data[new_start:].reset_index(drop=True)
        data = data[new_start:].reset_index(drop=True)

    while (len(peaks) < 20):
        prom_min = prom_min - 2
        peaks, _ = find_peaks(pressure_data, prominence=[prom_min, 300])
    
    start_idx = max(peaks[0]-sampling_freq, 0)
    end_idx = min(peaks[-1]+sampling_freq, data.shape[0])

    if plot:
        plt.plot(pressure_data)
        plt.plot(peaks, pressure_data[peaks], "x")
        plt.axvline(x = start_idx, color = 'b')
        plt.axvline(x = end_idx, color = 'b')
        #plt.ylim((-200, 200))
        plt.show()

    return data[start_idx:end_idx]

def fit_pca(train_windows):
    pca = PCA(n_components=5)
    all_data = np.vstack(train_windows)
    pca.fit(all_data)
    # print(pca.explained_variance_ratio_)
    return pca

def plot_pca(pcas):
    plt.figure(figsize=(10, 6))
    cumulative_explained_variance = []
    for pca in pcas:
        cumulative_explained_variance.append(np.cumsum(pca.explained_variance_ratio_))
        plt.plot(cumulative_explained_variance[-1], marker='o', linestyle='--', color='b')
    print("Average explained variance ratio for PCA components:")
    print(np.mean(cumulative_explained_variance, axis=0))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA')
    plt.grid(True)
    plt.show()

def create_windows(data, window_size, overlap=0):
    step_size = int(window_size * (1 - overlap))
    windows = []
    labels = []
    for label in data['label'].unique():
        label_data = data[data['label'] == label]
        for i in range(0, len(label_data) - window_size + 1, step_size):
            windows.append(label_data.iloc[i:i + window_size].drop(columns=['label', 'softness']).values)
            labels.append([int(label), label_data.iloc[i].softness])
    return np.array(windows), np.array(labels)

def normalize_windows(windows, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        # Stack all windows to fit the scaler on the entire dataset
        all_data = np.vstack(windows)
        scaler.fit(all_data)
    
    # Transform each window using the fitted scaler
    normalized_windows = [scaler.transform(window) for window in windows]
    
    return normalized_windows, scaler

def load_csv_files(directories, sampling_freq=50):
    data_frames = []
    required_columns = ["accx", "accy", "accz", "gx", "gy", "gz", "pressure"]
    total_files = 0

    def process_new_data(label, softness, df):
        df = trim_to_peaks(df, sampling_freq, False)
        df = df[required_columns]
        df['label'] = label
        df['softness'] = softness
        return df

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.startswith("material_"):
                for fName in os.listdir(directory+"/"+filename):
                    if (fName.startswith("M") and fName.endswith(".csv")) or (fName.startswith("EXP") and fName.endswith(".csv")):
                        # Extract label from the filename
                        label = filename.split('_')[1]
                        softness = "None"
                        df = pd.read_csv(os.path.join(directory, filename, fName))
                        data_frames.append(process_new_data(label, softness, df))
                        total_files = total_files+1
            if filename.endswith("_just_softness"):
                for fName in os.listdir(directory+"/"+filename):
                    if fName.endswith("delay150.csv"):
                        label = 18 # Add tape as material 18
                        softness = getSoftness(filename)
                        df = pd.read_csv(os.path.join(directory, filename, fName))
                        data_frames.append(process_new_data(label, softness, df))
                        total_files = total_files+1
            if directory.endswith("softness&texture"):
                for fabric in os.listdir(directory+"/"+filename):
                    for fName in os.listdir(directory+"/"+filename+"/"+fabric):
                        if fName.endswith("delay100.csv"):
                            label =  fabric.split('fabric')[1]
                            softness = getSoftness(filename)
                            df = pd.read_csv(os.path.join(directory, filename, fabric, fName))
                            data_frames.append(process_new_data(label, softness, df))
                            total_files = total_files+1

    print(f"Total files used: {total_files}")
    return pd.concat(data_frames, ignore_index=True)

def split_into_folds(windows, labels, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in kf.split(windows):
        folds.append((windows[train_index], labels[train_index], windows[test_index], labels[test_index]))
        _, b = np.unique(labels[train_index, 0], return_counts=True)
        print("Train label distribution:")
        print(b)
        _, b = np.unique(labels[train_index, 1], return_counts=True)
        print("Train softness distribution:")
        print(b)
        _, b = np.unique(labels[test_index, 0], return_counts=True)
        print("Test label distribution:")
        print(b)
        _, b = np.unique(labels[test_index, 1], return_counts=True)
        print("Test softness distribution:")
        print(b)
    return folds