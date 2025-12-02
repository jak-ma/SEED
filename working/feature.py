from sklearn.decomposition import PCA

def apply_pca(all_data, n_components=100):
    original_shape = all_data.shape
    X_flat = all_data.reshape(-1, 310)  # (15*3*15, 310)
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_flat)
    
 
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"[PCA] {n_components} components explain {explained_var:.3f} variance")
    print(f"[PCA] Shape: {original_shape} → {X_reduced.shape}")
    
    X_reduced = X_reduced.reshape(*original_shape[:-1], n_components)
    
    return X_reduced
