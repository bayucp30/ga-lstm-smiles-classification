from sklearn.decomposition import PCA
import pandas as pd

def apply_pca(df, feature_columns, n_components=10):
    """
    Apply PCA to normalized feature data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing normalized features
    feature_columns : list
        List of feature columns
    n_components : int
        Number of principal components

    Returns
    -------
    pca_df : pandas.DataFrame
        DataFrame containing principal components
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    """
    X = df[feature_columns].values

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        principal_components,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pca_df, pca
