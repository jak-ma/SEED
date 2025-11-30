from sklearn import svm
from preprocess import load_data, visualize_subjects
from sklearn.model_selection import LeaveOneOut
import scipy.io as sio
import numpy as np
import time
from tqdm import tqdm

def evaluate_model(all_data, label):
    y = np.array([list(label)*3 for _ in range(15)])
    model = svm.SVC(C=10, kernel='linear', probability=True)
    loo = LeaveOneOut()
    acc_results = []
    
    for train_idx, test_idx in tqdm(loo.split(all_data), desc='[Training]'):
        X_train, X_test = all_data[train_idx], all_data[test_idx]   # [14, 3, 15, 310], [1, 3, 15, 310]
        y_train, y_test = y[train_idx], y[test_idx]     # [14, 45], [1, 45]
        X_train = X_train.reshape(-1, 310)
        X_test = X_test.reshape(-1, 310)
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        acc_results.append(acc)

    return acc_results

def print_results(acc_results):
    print(f'SVM model evaluate mean score: {np.mean(acc_results):.3f}')

def main():
    print('Run Start!')
    print('Load data...')
    all_data = load_data()
    print('Load label...')
    label = sio.loadmat('input/label.mat')['label'].squeeze(0)
    print('Train model...')
    acc_results = evaluate_model(all_data, label)
    print('\nVisualize Test Results...')
    visualize_subjects(acc_results, 'SVM')
    print_results(acc_results)
    print('Run Done!')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time {end-start:.3f} seconds')
    