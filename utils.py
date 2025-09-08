import pandas as pd
import numpy as np

def print_coefficients(feature_names, coefficients):
    return pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients[0],
        'odds_ratio': np.exp(coefficients[0])
    })


def coefficients_to_prediction(intercept, coefficients, X):
    log_odds = intercept + np.dot(X, coefficients.T)
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


def predict_with_threshold(model, data, threshold=0.5) -> list[int]:
    probabilities = model.predict_proba(data)[:, 1]
    return [int(prob > threshold) for prob in probabilities]