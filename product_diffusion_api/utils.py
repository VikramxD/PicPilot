import torch



def accelerator():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else :
        device = 'cpu'
    return device 