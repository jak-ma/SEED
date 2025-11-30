import scipy.io as sio
from scipy.signal import butter, filtfilt
import numpy as np
import re
from glob import glob
from tqdm import tqdm


# 带通滤波
def bandpass(data, low=4.0, high=45.0, fs=200, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

# 微分熵-手工特征
def compute_DE_band(X):
    # X -> [62, T]
    # 定义相关的频段
    bands = {'delta':(1, 4), 'theta':(4, 8),
    'alpha':(8, 14), 'beta':(14, 31), 'gamma':(31, 50)}
    features = []
    for _, (low, high) in bands.items():
        # 每个频段滤波
        filtered = bandpass(X, low, high)
        # 计算方差
        var = np.var(filtered, axis=-1)  # [62,]
        # 防止乘0错误
        var = np.clip(var, 1e-6, None)
        # 计算DE
        de = 0.5*np.log(2*np.pi*np.e*var)   # [62,]
        features.append(de)
    # features -> [5, 62]
    return np.concatenate(features) # [310,]

# 数据预处理
def process_data(X_dict, is_dl=False):
    eeg_keys = [k for k in X_dict.keys() if re.search('_eeg\d+', k)]
    eeg_keys = sorted(eeg_keys, key=lambda x: int(re.search(r'eeg(\d+)', x).group(1)))
    X_trials = []
    for k in eeg_keys:
        X_raw = X_dict[k]   # [62, T]
        # 重参考（CAR）
        X = X_raw - np.mean(X_raw, axis=0, keepdims=True)    # [62, T]
        # 主带通滤波
        low, high = 4.0, 45.0
        if is_dl:
            low, high = 0.5, 50.0  
        X = bandpass(X, low=low, high=high)
        # 截取稳定段: 30s 到 120s
        start, end = 30*200, 120*200
        if X.shape[1] <= start:
            raise ValueError('Signal is too short.')
        end = min(end, X.shape[1])
        X = X[:, start:end]
        # DE
        de = compute_DE_band(X)     # [310,]
        X_trials.append(de)     # [15, 310]

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
    print(all_data.shape)

    return all_data