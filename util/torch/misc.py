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


def soft_sync(target, source, tau):
    """
    Soft update of target network parameters.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_sync(target, source):
    """
    Hard update of target network parameters.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)