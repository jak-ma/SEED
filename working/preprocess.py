import scipy.io as sio
from scipy.signal import butter, filtfilt
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

# 带通滤波
def bandpass(data, low=4.0, high=45.0, fs=200, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

# 微分熵-手工特征
def compute_DE_band(X):
    bands = {'delta':(1, 4), 'theta':(4, 8),
             'alpha':(8, 14), 'beta':(14, 31), 'gamma':(31, 50)}
    features = []
    fs = 200
    win_len = 1 * fs
    overlap = 0.5  
    
    for band_name, (low, high) in bands.items():
        filtered = bandpass(X, low, high)
        step = int(win_len * (1 - overlap))
        n_windows = (filtered.shape[1] - win_len) // step + 1
        
        band_features = []
        for i in range(n_windows):
            start = i * step
            end = start + win_len
            seg = filtered[:, start:end]
            
            power = np.mean(seg**2, axis=-1)
            power = np.clip(power, 1e-6, None)
            de = 0.5 * np.log(2 * np.pi * np.e * power)
            band_features.append(de)
        
        # 取时间维度上的平均值
        mean_de = np.mean(band_features, axis=0)
        features.append(mean_de)
    
    return np.concatenate(features)

# 数据预处理
def process_data(X_dict, is_dl=False):
    eeg_keys = [k for k in X_dict.keys() if re.search('_eeg\d+', k)]
    eeg_keys = sorted(eeg_keys, key=lambda x: int(re.search(r'eeg(\d+)', x).group(1)))
    X_trials = []
    
    for k in eeg_keys:
        X_raw = X_dict[k]
        
        # 先截取再滤波
        start, end = 30*200, 120*200
        if X_raw.shape[1] <= start:
            raise ValueError('Signal is too short.')
        end = min(end, X_raw.shape[1])
        X_cut = X_raw[:, start:end]
        
        # 重参考（CAR）
        X_ref = X_cut - np.mean(X_cut, axis=0, keepdims=True)
        
        # 带通滤波
        low, high = 4.0, 45.0
        if is_dl:
            low, high = 0.5, 50.0
        X_filtered = bandpass(X_ref, low=low, high=high)
        
        # DE特征提取
        de = compute_DE_band(X_filtered)
        X_trials.append(de)
    
    return np.array(X_trials)


# 数据加载
def load_data(data_path='input/*_*.mat', is_dl=False):
    # 路径索引
    paths = glob(data_path)
    subject_paths = []
    for i in range(1, 16):
        path = f'[\\\]{i}_'
        subject_paths.append([k for k in paths if re.search(path, k)])
    # subject_paths -> [15, 3]
    # 数据处理
    all_data = []
    for i in tqdm(range(15), desc='Processing data'):
        subject_data = []
        for j in range(3):
            path = subject_paths[i][j]
            X = sio.loadmat(path)
            processed_X = process_data(X, is_dl)   # [15, 62, T] -> [15, 310]
            subject_data.append(processed_X)    # [3, 15, 310]
        all_data.append(subject_data)
    all_data = np.array(all_data)   # [15, 3, 15, 310]

    return all_data

# 可视化
def visualize_subjects(subject_acc, model_name):

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(subject_acc) + 1), subject_acc, marker='o')
    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} | Accuracy per Subject")
    plt.grid(True)
    plt.savefig(f"templates/subject_acc_line({model_name}).png")
    plt.show()