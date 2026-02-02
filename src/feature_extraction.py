def extract_features(smiles, features):
    """
    Extract chemical structure features from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule
    features : list
        List of feature symbols to count

    Returns
    -------
    dict
        Dictionary of feature counts
    """
    feature_counts = {feature: 0 for feature in features}
    open_paren_count = 0
    i = 0

    while i < len(smiles):
        char = smiles[i]

        # Single-character features
        if char in feature_counts:
            feature_counts[char] += 1
        elif char.isalnum():
            feature_counts["Z"] += 1

        # Multi-character features
        if i + 1 < len(smiles):
            pair = smiles[i:i+2]
            if pair in feature_counts:
                feature_counts[pair] += 1
                i += 1
                continue

        if i + 2 < len(smiles):
            triple = smiles[i:i+3]
            if triple in feature_counts:
                feature_counts[triple] += 1
                i += 2
                continue

        # Parenthesis handling
        if char == "(":
            open_paren_count += 1
        elif char == ")" and open_paren_count > 0:
            open_paren_count -= 1
            feature_counts["()"] += 1

        i += 1

    # Correct carbon count
    all_c = (
        2 * feature_counts.get("COC", 0)
        + 2 * feature_counts.get("C=C", 0)
        + feature_counts.get("C=O", 0)
    )
    feature_counts["C"] -= all_c

    return feature_counts
