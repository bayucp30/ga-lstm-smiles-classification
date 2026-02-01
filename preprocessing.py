def normalize_features(df, feature_columns, smiles_column="SMILES"):
    """
    Normalize extracted features by SMILES length.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing extracted features
    feature_columns : list
        List of feature column names
    smiles_column : str
        Column name for SMILES strings

    Returns
    -------
    pandas.DataFrame
        Normalized feature DataFrame
    """
    smiles_length = df[smiles_column].apply(len)

    for col in feature_columns:
        df[col] = (df[col] / smiles_length * 100).astype(int)

    return df
