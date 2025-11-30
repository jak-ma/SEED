import mne
import scipy.io as sio
import re

path = 'input/2_20140404.mat'

if __name__ == '__main__':
    X = sio.loadmat(path)
    # print(X['jl_eeg7'].shape)
    eeg_keys = [k for k in X.keys() if re.search('_eeg\d+', k)]
    print(eeg_keys)
