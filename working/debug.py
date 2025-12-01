import mne
import scipy.io as sio
import re
import numpy as np
from glob import glob
from preprocess import load_data

path = 'input/2_20140404.mat'
label_path = 'input/label.mat'

if __name__ == '__main__':
    # all_data = load_data().reshape(-1, 310)
    # print(all_data[:,:15])
    pass