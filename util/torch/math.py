import torch

def r2_score(y_true, y_pred):
    """
    Calculates R-squared score of y_true and y_pred.
    """
    numerator = torch.sum((y_true - y_pred)**2)
    denominator = torch.sum((y_true - torch.mean(y_true))**2)
    r2 = 1 - numerator/denominator
    return r2