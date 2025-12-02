from sklearn import svm
from preprocess import load_data, visualize_subjects
from sklearn.model_selection import LeaveOneOut, GridSearchCV
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from feature_project import apply_pca, apply_anova

def evaluate_svm(all_data, label):
    y = np.array([list(label+1)*3 for _ in range(15)])
    loo = LeaveOneOut()
    acc_results = []
    
    for train_idx, test_idx in tqdm(loo.split(all_data), desc='[SVM Training]'):
        X_train, X_test = all_data[train_idx], all_data[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]     
        X_train = X_train.reshape(-1, 310)
        X_test = X_test.reshape(-1, 310)
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # 归一化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # # PCA|K=60
        # X_train, pca = apply_pca(X_train, n_components=60)
        # X_test = pca.transform(X_test)

        # ANOVA|K=80
        X_train, anova = apply_anova(X_train, y_train, n_features=80)
        X_test = anova.transform(X_test)

        # 网格搜索参数
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear',  'rbf', 'poly'],
            'gamma': ['scale', 'auto'] + [0.01, 0.1, 1]
        }
        grid = GridSearchCV(
            svm.SVC(probability=True),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
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
    model_name = 'SVM_ANOVA80'
    acc_results = evaluate_svm(all_data, label)
    print_results(acc_results)
    print('\nVisualize Test Results...')
    visualize_subjects(acc_results, model_name, 'SVM')
    print('Run Done!')

if __name__ == '__main__':
    main()

    