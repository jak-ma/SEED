from sklearn.decomposition import PCA
import numpy as np
from scipy import stats

def apply_pca(data, n_components=100):
    original_shape = data.shape
    X_flat = data
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_flat)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"[PCA] {n_components} components explain {explained_var:.3f} variance")
    print(f"[PCA] Shape: {original_shape} → {X_reduced.shape}")
    
    X_reduced = X_reduced.reshape(*original_shape[:-1], n_components)
    
    return X_reduced, pca


class ANOVASelector:
    def __init__(self, n_features=60):
        self.n_features = n_features
        self.selected_indices_ = None
        self.f_scores_ = None
    
    def fit(self, X, y):
        """
        在训练集上计算 F-score 并选择特征
        X: (n_samples, 310)
        y: (n_samples,)
        """
        n_total_features = X.shape[1]
        f_scores = np.zeros(n_total_features)
        
        # 对每个特征计算 F-score
        for feat_idx in range(n_total_features):
            groups = []
            for class_label in np.unique(y):
                group_data = X[y == class_label, feat_idx]
                groups.append(group_data)
            
            # 计算 F 值
            f_stat, _ = stats.f_oneway(*groups)
            f_scores[feat_idx] = f_stat
        
        # 保存 F-score
        self.f_scores_ = f_scores
        
        # 选择 F-score 最高的 n_features 个
        self.selected_indices_ = np.argsort(f_scores)[-self.n_features:][::-1]
        
        return self
    
    def transform(self, X):
        if self.selected_indices_ is None:
            raise ValueError("ANOVASelector must be fitted before transform!")
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_indices(self):
        return self.selected_indices_
    
    def get_f_scores(self):
        return self.f_scores_

def apply_anova(X_train, y_train, n_features=60):
    anova = ANOVASelector(n_features=n_features)
    X_train_selected = anova.fit_transform(X_train, y_train)
    return X_train_selected, anova
