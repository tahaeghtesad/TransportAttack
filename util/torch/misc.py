import torch


def allocate_best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.backends.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')