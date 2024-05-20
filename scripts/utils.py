import gc
import torch




def flush():
    gc.collect()
    torch.cuda.empty_cache()