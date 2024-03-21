import gc
import torch
from logger import rich_logger as l

def clear_memory():
    """
    Clears the memory by collecting garbage and emptying the CUDA cache.

    This function is useful when dealing with memory-intensive operations in Python, especially when using libraries like PyTorch.

    Note:
        This function requires the `gc` and `torch` modules to be imported.

    """
    gc.collect()
    torch.cuda.empty_cache()
    l.info("Memory Cleared")
    