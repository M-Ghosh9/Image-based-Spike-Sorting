# data_loader.py

import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

def read_file_with_label(file_path, label):
    file_data = []
    with open(file_path, 'r') as f:
        file_data = [list(map(float, line.split())) for line in f if line.strip()]
    return file_data, label

def load_data_with_labels_optimized(directory):
    all_signals = []
    all_labels = []
    max_length = 0
    file_paths = []

    current_label = 0
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for subdir_root, subdirs, files in os.walk(dir_path):
                for file in sorted(files):
                    if file.endswith('.txt'):
                        file_path = os.path.join(subdir_root, file)
                        label = current_label
                        file_paths.append((file_path, label))
                        current_label += 1

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda fp: read_file_with_label(fp[0], fp[1]), file_paths))

    for file_data, label in results:
        if file_data:
            all_signals.extend(file_data)
            all_labels.extend([label] * len(file_data))
            max_length = max(max_length, max(len(signal) for signal in file_data))

    padded_signals = [signal + [0] * (max_length - len(signal)) for signal in all_signals]
    return np.array(padded_signals), np.array(all_labels)

def preprocess_data(signals):
    scaler = StandardScaler()
    signals_normalized = scaler.fit_transform(signals)
    signals_reshaped = torch.tensor(signals_normalized, dtype=torch.float32).unsqueeze(1)
    return signals_reshaped, scaler

if __name__ == "__main__":
    DATA_DIR = '/home/Guest/Downloads/ImageSpike Sorter/TrainingData'
    signals, labels = load_data_with_labels_optimized(DATA_DIR)
    signals_reshaped, scaler = preprocess_data(signals)
    torch.save((signals_reshaped, labels), 'signals.pt')
    print(f"Data loaded and preprocessed. Shape: {signals_reshaped.shape}")
    print(f"Labels loaded. Shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)[:10]}")
    print(f"Number of unique labels: {len(np.unique(labels))}")


