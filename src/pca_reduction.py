import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X, n_components=20):
    """
    Apply PCA for dimensionality reduction
    """

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return X_pca, pca
