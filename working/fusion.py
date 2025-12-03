import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
import scipy.io as sio
from tqdm import tqdm
import re
from glob import glob

class MultiFeatureExtractor:
    def __init__(self, fs=200):
        self.fs = fs
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
    
    # 特征1: DE (微分熵)
    def extract_de_features(self, X):
        features = []
        
        for low, high in self.bands.values():
            # 带通滤波
            nyq = self.fs / 2
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            X_band = filtfilt(b, a, X, axis=-1)
            
            # 计算每个通道的方差
            variances = np.var(X_band, axis=-1)
            
            # 微分熵: DE = 0.5 * log(2πe * σ²)
            de = 0.5 * np.log(2 * np.pi * np.e * variances + 1e-15)
            features.append(de)
        
        return np.concatenate(features)  # (310,)
    
    # 特征2: PSD (功率谱)
    def extract_psd_features(self, X):
        features = []
        
        for low, high in self.bands.values():
            band_powers = []
            
            for ch in range(X.shape[0]):
                freqs, psd = welch(
                    X[ch, :],
                    fs=self.fs,
                    nperseg=512,
                    noverlap=256
                )
                
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.mean(psd[mask])  
                band_powers.append(band_power)
            
            features.append(np.array(band_powers))
        
        return np.concatenate(features)  # (310,)
    
    # 特征3: 时域特征
    def extract_time_domain_features(self, X):
        features = []
        
        # 特征1: 均值
        mean_vals = np.mean(X, axis=-1)  # (62,)
        features.append(mean_vals)
        
        # 特征2: 方差
        var_vals = np.var(X, axis=-1)  # (62,)
        features.append(var_vals)
        
        # 特征3: 斜度 (偏度)
        skew_vals = skew(X, axis=-1)  # (62,)
        features.append(skew_vals)
        
        # 特征4: 峰度 (Kurtosis)
        kurt_vals = kurtosis(X, axis=-1)  # (62,)
        features.append(kurt_vals)
        
        return np.concatenate(features)  # (248,)
    
    # 特征4: 频域特征
    def extract_frequency_domain_features(self, X):
        features = []
        
        for low, high in self.bands.values():
            band_features = []
            
            for ch in range(X.shape[0]):
                # 计算功率谱
                freqs, psd = welch(
                    X[ch, :],
                    fs=self.fs,
                    nperseg=512,
                    noverlap=256
                )
                
                # 只看指定频段
                mask = (freqs >= low) & (freqs <= high)
                band_psd = psd[mask]
                
                # 归一化功率谱作为概率分布
                band_psd_norm = band_psd / (np.sum(band_psd) + 1e-15)
                
                concentration = np.max(band_psd_norm)
                band_features.append(concentration)
            
            features.append(np.array(band_features))
        
        return np.concatenate(features)  # (310,)
    
    # 高级时域特征
    def extract_advanced_time_features(self, X):
        features = []
        
        # 特征1: 一阶差分
        X_diff1 = np.diff(X, axis=-1)
        var_diff1 = np.var(X_diff1, axis=-1)
        features.append(var_diff1)
        
        # 特征2: 二阶差分
        X_diff2 = np.diff(X_diff1, axis=-1)
        var_diff2 = np.var(X_diff2, axis=-1)
        features.append(var_diff2)
        
        # 特征3: 零交叉率
        zero_crossings = np.sum(np.diff(np.sign(X), axis=-1) != 0, axis=-1)
        zcr = zero_crossings / X.shape[-1]
        features.append(zcr)
        
        return np.concatenate(features)  # (186,)
    
    # 特征融合
    def extract_all_features(self, X):

        de_features = self.extract_de_features(X)
        
        psd_features = self.extract_psd_features(X)

        time_features = self.extract_time_domain_features(X)

        freq_features = self.extract_frequency_domain_features(X)
        
        advanced_time_features = self.extract_advanced_time_features(X)
        
        all_features = np.concatenate([
            de_features,
            # psd_features,
            # time_features,
            freq_features,
            advanced_time_features
        ])
        
        return all_features

def load_data_with_fusion(data_path='input/*_*.mat'):

    paths = glob(data_path)
    subject_paths = []
    for i in range(1, 16):
        path = f'[\\\]{i}_'
        subject_paths.append([k for k in paths if re.search(path, k)])

    extractor = MultiFeatureExtractor(fs=200)
    
    all_data = []
    
    for i in tqdm(range(15), desc='Processing data'):
        subject_data = []
        for j in range(3):
            path = subject_paths[i][j]
            data_dict = sio.loadmat(path)
        
            # 提取 EEG 数据
            eeg_keys = [k for k in data_dict.keys() if re.search('_eeg\d+', k)]
            eeg_keys = sorted(eeg_keys, key=lambda x: int(re.search(r'eeg(\d+)', x).group(1)))
            
            session_data = []
            for eeg_key in eeg_keys:
                X_raw = data_dict[eeg_key]  # (62, T)
                
                # 带通滤波 (4-45 Hz)
                nyq = 200 / 2
                b, a = butter(4, [4/nyq, 45/nyq], btype='band')
                X_filtered = filtfilt(b, a, X_raw, axis=-1)
                
                # 截取 30-120s
                start, end = 30*200, 120*200
                if X_filtered.shape[1] > start:
                    end = min(end, X_filtered.shape[1])
                    X_cut = X_filtered[:, start:end]
                else:
                    X_cut = X_filtered
                
                features = extractor.extract_all_features(X_cut)
                
                session_data.append(features)
            
            subject_data.append(np.array(session_data))
    
        all_data.append(subject_data)

    return np.array(all_data)