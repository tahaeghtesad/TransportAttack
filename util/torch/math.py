import torch


def r2_score(y_true, y_pred):
    """
    Calculates R-squared score of y_true and y_pred.
    """
    numerator = torch.sum((y_true - y_pred)**2)
    denominator = torch.sum((y_true - torch.mean(y_true))**2)
    r2 = 1 - numerator/denominator
    return r2


def explained_variance(y_true, y_pred):
    """
    Calculates explained variance of y_true and y_pred.
    """
    y_var = y_true.var()
    diff_var = (y_true - y_pred).var()
    return 1.0 - (diff_var / y_var + 1e-8)


def sigmoid(x, c1, c2):
    """
    Sigmoid function with parameters c1 and c2.
    """
    return 1/(1 + torch.exp(-c1*(x - c2)))
