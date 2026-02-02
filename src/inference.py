import numpy as np


def run_inference(
    model,
    X_new
):
    """
    Run inference on new unseen data
    """

    # reshape for LSTM
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    predictions = model.predict(X_new)

    return predictions
