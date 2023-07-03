import torch


def allocate_best_device(default_device=None):
    if default_device is not None:
        return torch.device(default_device)
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
