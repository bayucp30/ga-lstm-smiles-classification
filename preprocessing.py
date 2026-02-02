import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_features(X: pd.DataFrame):
    """
    Standardize features using StandardScaler
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
